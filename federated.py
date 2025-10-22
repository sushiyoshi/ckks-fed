# =========================================================

# 🔒 Secure Federated Averaging with OpenFHE CKKS (full script)
#   - The original Concrete-ML (TFHE-like) parts are fully replaced.
#   - External class interface of FHEModelAggregator is preserved:
#       * __init__(model_structure, num_clients, ...)
#       * encrypt_model_weights(model_weights_dict)  -> returns per-layer "encrypted" payloads
#       * aggregate_encrypted_models(client_weights_list) -> returns aggregated torch state_dict
#   - Internals now use OpenFHE CKKS to pack/encrypt/add/scale/decrypt.
#   - Minimal changes to the rest of the federated-learning code.
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import threading
import time
import copy
from math import log2, ceil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIなし環境対応

# [OPENFHE-CKKS] 追加
from openfhe import *


# =========================
# 小ユーティリティ
# =========================
def _next_pow2(n: int) -> int:
    return 1 << ceil(log2(max(1, n)))


# =========================================================
# 🔒 CKKS-based Model Aggregator (インタフェースは維持)
#   - 以前は Concrete-ML の回路コンパイル & 暗号実行
#   - いまは OpenFHE CKKS で:
#       1) サーバ側: CryptoContext + keys を一度用意
#       2) クライアント: レイヤごとに packed plaintext → Encrypt
#       3) サーバ: レイヤごとに EvalAdd（和）→ (1/N) の平文スカラー乗算 → Decrypt
# =========================================================
class FHEModelAggregator:
    """
    役割:
      - 各レイヤ重みの形状を記録
      - CKKSコンテキストを 1 つ準備（BatchSize は最大要素数に合わせる）
      - クライアント重みをレイヤごとに flatten → CKKS で pack & encrypt
      - サーバ側で暗号加算 & 定数乗算（1/num_clients）→ 復号 → 形状に戻して返す
    """

    def __init__(
        self,
        model_structure,
        num_clients=5,
        # 以下2つは Concrete-ML 版の残置引数（外部インタフェース維持のため受け取るが無視）
        scale_factor=100,
        max_value=50,
        # CKKS パラメータ（必要最小限）
        mult_depth: int = 1,
        scale_mod_size: int = 50,
        security_level: SecurityLevel = SecurityLevel.HEStd_128_classic,
    ):
        self.num_clients = num_clients

        # モデル構造から重みテンソルの形状を取得し辞書に保存
        self.weight_shapes = {}
        max_elems = 0
        for name, param in model_structure.named_parameters():
            self.weight_shapes[name] = param.shape
            max_elems = max(max_elems, int(np.prod(param.shape)))

        # CKKS の BatchSize は 2 の冪に（最大要素数に基づき設定）
        self.batch_size = _next_pow2(max_elems)

        # ---- CKKS 文脈を初期化（Concrete-ML の回路コンパイルは完全撤去） ----
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(mult_depth)     # 和 + 平文スカラー乗算だけなら 1 で十分
        params.SetScalingModSize(scale_mod_size)      # 近似精度の指標（必要に応じて個別調整）
        params.SetBatchSize(self.batch_size)
        params.SetSecurityLevel(security_level)

        self.cc = GenCryptoContext(params)
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)

        # 鍵生成（乗算は平文定数なので必須ではないが、安全側で生成）
        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)

        print(f"🔐 CKKS ready: ring dimension = {self.cc.GetRingDimension()}, batch_size = {self.batch_size}")

        # 以前の Concrete-ML 回路置き場はダミーで残す（外部参照されないが互換のため）
        self.circuits = {}

    # [REPLACED] Concrete-ML の回路コンパイルは削除（互換のため空メソッドは残さない）

    def _flatten_to_len(self, tensor: torch.Tensor, length: int):
        """Tensor → 1D float list（必要なら 0 パディング）"""
        v = tensor.detach().cpu().float().numpy().reshape(-1).tolist()
        if len(v) > length:
            raise ValueError(f"vector length {len(v)} exceeds batch_size {length}")
        if len(v) < length:
            v = v + [0.0] * (length - len(v))
        return v

    def encrypt_model_weights(self, model_weights_dict):
        """
        各クライアントのモデル重みを「暗号化入力形式」に整形して返す（インタフェースは維持）。
        Concrete-ML 版との違い:
          - 以前は整数化した flatten 配列（平文）を返していた
          - いまは CKKS の Ciphertext を返す
        戻り値:
          { layer_name: CiphertextCKKSRNS }
        """
        encrypted_weights = {}
        for layer_name, shape in self.weight_shapes.items():
            w_tensor: torch.Tensor = model_weights_dict[layer_name]
            flat = self._flatten_to_len(w_tensor, self.batch_size)  # 0パディングあり
            pt = self.cc.MakeCKKSPackedPlaintext(flat)
            ct = self.cc.Encrypt(self.keys.publicKey, pt)
            encrypted_weights[layer_name] = ct
        return encrypted_weights

    def aggregate_encrypted_models(self, client_weights_list):
        """
        各レイヤごとに暗号文のまま平均を計算し、PyTorchテンソルに戻す。
        Concrete-ML 版との互換:
          - 引数 client_weights_list は「各クライアントの state_dict」
          - 内部で encrypt_model_weights を呼び、レイヤごとに暗号和→定数乗算→復号
        戻り値:
          aggregated_state_dict (torch state_dict)
        """
        print("\n🔒 Starting CKKS model aggregation...")

        # 1) 各クライアントの重みを CKKS 暗号化
        encrypted_weights_list = []
        for i, client_state_dict in enumerate(client_weights_list):
            print(f"  🔑 Encrypting weights from Client {i+1}...")
            enc = self.encrypt_model_weights(client_state_dict)
            encrypted_weights_list.append(enc)

        # 2) レイヤごとに暗号集約
        aggregated_weights = {}
        inv_n = 1.0 / len(encrypted_weights_list)

        for layer_name, shape in self.weight_shapes.items():
            # クライアントの同じレイヤの暗号文を集める
            layer_cts = [enc[layer_name] for enc in encrypted_weights_list]
            # 和
            c_sum = layer_cts[0]
            for ct in layer_cts[1:]:
                c_sum = self.cc.EvalAdd(c_sum, ct)
            # 平均（平文定数との乗算）
            c_avg = self.cc.EvalMult(c_sum, inv_n)
            # 復号
            pt_avg = self.cc.Decrypt(c_avg, self.keys.secretKey)
            pt_avg.SetLength(self.batch_size)
            # 先頭 shape 要素ぶんのみ取り出し（パディング除去）
            num_elems = int(np.prod(shape))
            # openfhe-python の Plaintext から値リストを得る一般形:
            #vals = pt_avg.GetCKKSPackedValue()  # 実数配列（Pythonバインディングでの一般 API）
            vals =pt_avg.GetRealPackedValue()
            trimmed = np.array(vals[:num_elems], dtype=np.float32).reshape(shape)
            aggregated_weights[layer_name] = torch.from_numpy(trimmed)

            print(f"  ✅ Aggregated {layer_name} with {len(layer_cts)} clients")

        print("✅ CKKS model aggregation completed!")
        return aggregated_weights


# =========================================================
# モデル／データセットの簡易実装（説明目的：元コードのまま）
# =========================================================

class CustomDataset(Dataset):
    """featuresとlabelsを受け取る薄いDataset"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class NeuralNetwork(nn.Module):
    """
    1層の全結合のみを持つ極小ネット（説明・検証用）。
    初期重みは固定して、挙動追跡を容易に。
    """
    def __init__(self, input_size=3, output_size=2):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        with torch.no_grad():
            self.fc.weight.data = torch.tensor([[0.1, 0.2, 0.3],
                                                [0.4, 0.5, 0.6]], dtype=torch.float32)
            self.fc.bias.data = torch.tensor([0.1, 0.2], dtype=torch.float32)

    def forward(self, x):
        return self.fc(x)


class LocalTrainer:
    """
    各クライアント側の学習・評価・重み取得を担当するヘルパ
    """
    def __init__(self, model, dataset, optimizer, criterion, device):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, epochs):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for data, target in self.dataset:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                predicted = torch.argmax(output, dim=1)
                epoch_correct += (predicted == target).sum().item()
                epoch_samples += target.size(0)
                epoch_loss += loss.item()

            epoch_accuracy = epoch_correct / epoch_samples * 100
            print(f"    Epoch {epoch + 1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")
            total_loss += epoch_loss
            correct += epoch_correct
            total_samples += epoch_samples

        overall_accuracy = correct / total_samples * 100
        print(f"  Local training completed: Overall Accuracy = {overall_accuracy:.2f}%")
        return overall_accuracy

    def evaluate(self, test_dataset=None):
        """与えられたデータローダで精度と損失を評価"""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        dataset = test_dataset if test_dataset else self.dataset
        with torch.no_grad():
            for data, target in dataset:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                total_loss += loss.item()

        accuracy = correct / total * 100
        avg_loss = total_loss / len(dataset)
        print(f"  Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss

    def get_model_weights(self):
        """モデル状態をディープコピーで返す（破壊的変更を防ぐ）"""
        return copy.deepcopy(self.model.state_dict())

    def set_model_weights(self, weights):
        """外部から与えられた重みでモデルを上書き"""
        self.model.load_state_dict(weights)


def create_simple_data(batch_size, num_samples=100, seed=None):
    """
    線形分離可能な簡易データを生成（説明・検証用）
    ルール: x1 + x2 - x3 > 0 なら 1、それ以外は 0
    """
    if seed is not None:
        np.random.seed(seed)
    features = np.random.randn(num_samples, 3).astype(np.float32)
    labels = ((features[:, 0] + features[:, 1] - features[:, 2]) > 0).astype(np.int64)
    dataset = CustomDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Client:
    """
    各クライアント:
      - 手元データでローカル学習
      - 学習前後の精度を出力
      - 学習後の重みをサーバへ返す
    """
    def __init__(self, client_id, data_seed=None):
        self.client_id = client_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNetwork(input_size=3, output_size=2).to(self.device)
        self.train_loader = create_simple_data(batch_size=16, num_samples=100, seed=data_seed)
        self.test_loader = create_simple_data(batch_size=16, num_samples=50, seed=data_seed+1000 if data_seed else None)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)  # 低LRで安定化
        self.criterion = nn.CrossEntropyLoss()
        self.trainer = LocalTrainer(self.model, self.train_loader, self.optimizer, self.criterion, self.device)

    def local_update(self, global_weights, epochs=2):
        """グローバル重みで初期化→学習→評価→重み返却"""
        print(f"Client {self.client_id}: Starting local update")
        self.trainer.set_model_weights(global_weights)
        print(f"Client {self.client_id}: Pre-training evaluation:")
        pre_accuracy, _ = self.trainer.evaluate(self.test_loader)
        print(f"Client {self.client_id}: Starting local training...")
        train_accuracy = self.trainer.train(epochs=epochs)
        print(f"Client {self.client_id}: Post-training evaluation:")
        post_accuracy, _ = self.trainer.evaluate(self.test_loader)
        print(f"Client {self.client_id} Summary:")
        print(f"  Pre-training accuracy: {pre_accuracy:.2f}%")
        print(f"  Training accuracy: {train_accuracy:.2f}%")
        print(f"  Post-training accuracy: {post_accuracy:.2f}%")
        print(f"  Accuracy improvement: {post_accuracy - pre_accuracy:.2f}%")
        return self.trainer.get_model_weights()


# =========================================================
# サーバ: グローバルモデル & FHE集約を管理（Aggregator を CKKS 版に）
# =========================================================
class FHEServer:
    """
    役割:
      - グローバルモデルの保持と評価
      - CKKS 集約（FHEModelAggregator: CKKS 版）を呼び出し
    """
    def __init__(self, num_clients=5):
        self.num_clients = num_clients
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = NeuralNetwork(input_size=3, output_size=2).to(self.device)

        # [REPLACED WITH OPENFHE-CKKS] ここで CKKS 版の Aggregator を使用
        self.fhe_aggregator = FHEModelAggregator(
            self.global_model,
            num_clients=num_clients,
            scale_factor=100,   # 引数互換のため受け渡しはするが、CKKS 版では未使用
            max_value=50        # 同上
        )

        self.test_loader = create_simple_data(batch_size=32, num_samples=200, seed=9999)
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_global_model(self):
        """サーバでグローバルモデルの精度を評価"""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = self.criterion(output, target)
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                total_loss += loss.item()
        accuracy = correct / total * 100
        avg_loss = total_loss / len(self.test_loader)
        print(f"Global Model - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss

    def aggregate_models_with_fhe(self, client_weights_list):
        """
        CKKS でクライアント重みを平均し、グローバルへ適用。
        """
        print("🔒 FHE-based model aggregation (OpenFHE CKKS)...")
        aggregated_weights = self.fhe_aggregator.aggregate_encrypted_models(client_weights_list)
        self.global_model.load_state_dict(aggregated_weights)
        print("🔒 FHE Global model updated:")
        for name, param in self.global_model.named_parameters():
            print(f"  {name}: {param.data}")

    def get_global_weights(self):
        """クライアントに配布するためのグローバル重みを返す"""
        return copy.deepcopy(self.global_model.state_dict())


# =========================================================
# 平文サーバ: 暗号化なしの通常の連合学習
# =========================================================
class PlainServer:
    """
    役割:
      - グローバルモデルの保持と評価
      - 平文でのモデル集約（単純平均）
    """
    def __init__(self, num_clients=5):
        self.num_clients = num_clients
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = NeuralNetwork(input_size=3, output_size=2).to(self.device)
        self.test_loader = create_simple_data(batch_size=32, num_samples=200, seed=9999)
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_global_model(self):
        """サーバでグローバルモデルの精度を評価"""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = self.criterion(output, target)
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                total_loss += loss.item()
        accuracy = correct / total * 100
        avg_loss = total_loss / len(self.test_loader)
        print(f"Global Model - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss

    def aggregate_models_plain(self, client_weights_list):
        """
        平文でクライアント重みを平均し、グローバルへ適用。
        """
        print("📊 Plain model aggregation (simple averaging)...")

        # 全クライアントの重みを平均
        aggregated_weights = {}
        for layer_name in client_weights_list[0].keys():
            layer_weights = torch.stack([w[layer_name] for w in client_weights_list])
            aggregated_weights[layer_name] = torch.mean(layer_weights, dim=0)

        self.global_model.load_state_dict(aggregated_weights)
        print("📊 Plain Global model updated:")
        for name, param in self.global_model.named_parameters():
            print(f"  {name}: {param.data}")

    def get_global_weights(self):
        """クライアントに配布するためのグローバル重みを返す"""
        return copy.deepcopy(self.global_model.state_dict())


# =========================================================
# 比較実験関数
# =========================================================
def run_federated_learning_comparison(num_clients=10, num_rounds=10):
    """
    平文とCKKS暗号化の連合学習を実行し、精度と時間を比較
    """
    print("="*80)
    print("🔬 FEDERATED LEARNING COMPARISON: Plain vs CKKS Encrypted")
    print("="*80)

    # 結果格納用
    results = {
        'plain': {'accuracy': [], 'time': []},
        'ckks': {'accuracy': [], 'time': [], 'key_gen_time': 0}
    }

    # =========================================================
    # 1. 平文での連合学習
    # =========================================================
    print("\n" + "="*80)
    print("📊 PLAIN FEDERATED LEARNING")
    print("="*80)

    plain_server = PlainServer(num_clients=num_clients)
    plain_clients = [Client(client_id=i+1, data_seed=i*100) for i in range(num_clients)]

    print("\nInitial Global Model Evaluation (Plain):")
    initial_accuracy_plain, _ = plain_server.evaluate_global_model()
    results['plain']['accuracy'].append(initial_accuracy_plain)

    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"📊 PLAIN ROUND {round_num + 1}")
        print(f"{'='*60}")

        round_start = time.time()

        global_weights = plain_server.get_global_weights()
        client_weights_list = []

        for client in plain_clients:
            print(f"\n--- Client {client.client_id} Local Update ---")
            local_weights = client.local_update(global_weights, epochs=2)
            client_weights_list.append(local_weights)

        print(f"\n--- 📊 Plain Server Aggregation ---")
        plain_server.aggregate_models_plain(client_weights_list)

        round_end = time.time()
        round_time = round_end - round_start

        print(f"\nRound {round_num + 1} Global Model Evaluation (Plain):")
        current_accuracy, _ = plain_server.evaluate_global_model()

        results['plain']['accuracy'].append(current_accuracy)
        results['plain']['time'].append(round_time)

        print(f"📊 Plain Round {round_num + 1} completed in {round_time:.2f} seconds!")

    # =========================================================
    # 2. CKKS暗号化での連合学習
    # =========================================================
    print("\n" + "="*80)
    print("🔒 CKKS ENCRYPTED FEDERATED LEARNING")
    print("="*80)

    # 鍵生成時間を計測
    key_gen_start = time.time()
    ckks_server = FHEServer(num_clients=num_clients)
    key_gen_end = time.time()
    key_gen_time = key_gen_end - key_gen_start
    results['ckks']['key_gen_time'] = key_gen_time

    print(f"🔑 Key generation time: {key_gen_time:.2f} seconds")

    ckks_clients = [Client(client_id=i+1, data_seed=i*100) for i in range(num_clients)]

    print("\nInitial Global Model Evaluation (CKKS):")
    initial_accuracy_ckks, _ = ckks_server.evaluate_global_model()
    results['ckks']['accuracy'].append(initial_accuracy_ckks)

    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"🔒 CKKS ENCRYPTED ROUND {round_num + 1}")
        print(f"{'='*60}")

        round_start = time.time()

        global_weights = ckks_server.get_global_weights()
        client_weights_list = []

        for client in ckks_clients:
            print(f"\n--- Client {client.client_id} Local Update ---")
            local_weights = client.local_update(global_weights, epochs=2)
            client_weights_list.append(local_weights)

        print(f"\n--- 🔒 CKKS Server Aggregation ---")
        ckks_server.aggregate_models_with_fhe(client_weights_list)

        round_end = time.time()
        round_time = round_end - round_start

        print(f"\nRound {round_num + 1} Global Model Evaluation (CKKS):")
        current_accuracy, _ = ckks_server.evaluate_global_model()

        results['ckks']['accuracy'].append(current_accuracy)
        results['ckks']['time'].append(round_time)

        print(f"🔒 CKKS Round {round_num + 1} completed in {round_time:.2f} seconds!")

    return results


def plot_comparison_results(results, num_rounds, num_clients):
    """
    比較結果をグラフ化して保存
    """
    key_gen_time = results['ckks']['key_gen_time']

    # 図の作成
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # ラウンド番号（0はInitial、1以降は各ラウンド）
    rounds = list(range(num_rounds + 1))

    # =========================================================
    # 1. 精度の比較グラフ
    # =========================================================
    ax1 = axes[0]
    ax1.plot(rounds, results['plain']['accuracy'], 'o-', label='Plain', linewidth=2, markersize=8)
    ax1.plot(rounds, results['ckks']['accuracy'], 's-', label='CKKS Encrypted', linewidth=2, markersize=8)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Federated Learning Accuracy Comparison: Plain vs CKKS', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)
    ax1.set_xticklabels(['Initial'] + [f'R{i}' for i in range(1, num_rounds + 1)])

    # =========================================================
    # 2. 実行時間の比較グラフ
    # =========================================================
    ax2 = axes[1]
    round_nums = list(range(1, num_rounds + 1))

    bar_width = 0.35
    x = np.arange(len(round_nums))

    bars1 = ax2.bar(x - bar_width/2, results['plain']['time'], bar_width,
                    label='Plain', alpha=0.8)
    bars2 = ax2.bar(x + bar_width/2, results['ckks']['time'], bar_width,
                    label='CKKS Encrypted', alpha=0.8)

    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Federated Learning Execution Time Comparison: Plain vs CKKS', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Round {i}' for i in round_nums])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # 鍵生成時間をテキストで表示
    key_gen_text = f'CKKS Key Generation Time: {key_gen_time:.2f} seconds\n(Not included in round times)'
    ax2.text(0.02, 0.98, key_gen_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # グラフを保存
    output_file = f'federated_learning_comparison_clients{num_clients}_rounds{num_rounds}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison graph saved to: {output_file}")

    plt.close()


def print_comparison_summary(results, num_rounds):
    """
    比較結果のサマリーを標準出力
    """
    print("\n" + "="*80)
    print("📈 COMPARISON SUMMARY")
    print("="*80)

    print(f"\n🔑 CKKS Key Generation Time: {results['ckks']['key_gen_time']:.2f} seconds")
    print("   (This time is not included in round execution times)")

    print("\n" + "-"*80)
    print("📊 Accuracy Comparison (per round):")
    print("-"*80)
    print(f"{'Round':<15} {'Plain (%)':<15} {'CKKS (%)':<15} {'Difference':<15}")
    print("-"*80)

    for i in range(num_rounds + 1):
        round_name = "Initial" if i == 0 else f"Round {i}"
        plain_acc = results['plain']['accuracy'][i]
        ckks_acc = results['ckks']['accuracy'][i]
        diff = ckks_acc - plain_acc
        print(f"{round_name:<15} {plain_acc:>8.2f}%      {ckks_acc:>8.2f}%      {diff:>+8.2f}%")

    print("\n" + "-"*80)
    print("⏱️  Execution Time Comparison (per round):")
    print("-"*80)
    print(f"{'Round':<15} {'Plain (s)':<15} {'CKKS (s)':<15} {'Overhead':<15}")
    print("-"*80)

    for i in range(num_rounds):
        plain_time = results['plain']['time'][i]
        ckks_time = results['ckks']['time'][i]
        overhead = ((ckks_time / plain_time) - 1) * 100
        print(f"Round {i+1:<8} {plain_time:>8.2f}s      {ckks_time:>8.2f}s      {overhead:>+8.1f}%")

    # 平均値
    avg_plain_time = np.mean(results['plain']['time'])
    avg_ckks_time = np.mean(results['ckks']['time'])
    avg_overhead = ((avg_ckks_time / avg_plain_time) - 1) * 100

    print("-"*80)
    print(f"{'Average':<15} {avg_plain_time:>8.2f}s      {avg_ckks_time:>8.2f}s      {avg_overhead:>+8.1f}%")
    print("-"*80)

    print("\n" + "="*80)


# =========================================================
# 実行エントリポイント
# =========================================================
def main():
    """
    比較実験のメイン関数
    """
    num_clients = 10
    num_rounds = 10

    # 比較実験を実行
    results = run_federated_learning_comparison(num_clients=num_clients, num_rounds=num_rounds)

    # 結果のサマリーを出力
    print_comparison_summary(results, num_rounds)

    # グラフを作成
    plot_comparison_results(results, num_rounds, num_clients)

    print("\n✅ All comparisons completed!")


if __name__ == "__main__":
    main()

