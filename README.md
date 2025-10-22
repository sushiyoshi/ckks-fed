# CKKS Federated Learning

OpenFHE CKKSスキームを使用した完全準同型暗号による連合学習の実装。

## 環境構築

```bash
# 仮想環境の作成
python -m venv .

# 仮想環境の有効化
source bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 実行例

```bash
# CKKS基本サンプル
python ckks_sample.py

# ブートストラップサンプル
python simple_ckks_bootstrap.py

# 連合学習（平文とCKKS暗号化の比較）
python federated.py
```
