from math import log2, ceil
import random
from openfhe import *


def _next_pow2(n: int) -> int:
    return 1 << ceil(log2(max(1, n)))


def setup_ckks(
    mult_depth: int = 1,
    scale_mod_size: int = 50,
    batch_size: int | None = None,
    security_level: SecurityLevel = SecurityLevel.HEStd_128_classic,
):
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(scale_mod_size)
    params.SetBatchSize(batch_size if batch_size is not None else _next_pow2(8))
    params.SetSecurityLevel(security_level)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    return cc, keys


def encrypt_weights(cc, public_key, weights: list[float]):
    pt = cc.MakeCKKSPackedPlaintext(weights)
    return cc.Encrypt(public_key, pt)


def federated_average_encrypted(cc, ciphertexts: list):
    if not ciphertexts:
        raise ValueError("ciphertexts is empty")
    c_sum = ciphertexts[0]
    for ct in ciphertexts[1:]:
        c_sum = cc.EvalAdd(c_sum, ct)
    inv_n = 1.0 / len(ciphertexts)
    return cc.EvalMult(c_sum, inv_n)


def decrypt_to_plaintext(cc, secret_key, ct, length: int, precision: int = 8) -> str:
    pt = cc.Decrypt(ct, secret_key)
    pt.SetLength(length)
    return pt.GetFormattedValues(precision)


def _gen_random_clients(num_clients: int, vec_len: int, lo: float = -3.0, hi: float = 3.0):
    return [[random.uniform(lo, hi) for _ in range(vec_len)] for _ in range(num_clients)]


def _plaintext_average(all_weights: list[list[float]]) -> list[float]:
    n = len(all_weights)
    vec_len = len(all_weights[0])
    s = [0.0] * vec_len
    for w in all_weights:
        for i, v in enumerate(w):
            s[i] += v
    return [sv / n for sv in s]


if __name__ == "__main__":
    # 固定長ベクトル（スロット数）：8。必要なら変更可。
    vec_len = 8
    batch_size = _next_pow2(vec_len)

    # CKKS 文脈（深さ1: 和 + 平文スカラー乗算のみ）
    cc, keys = setup_ckks(mult_depth=1, scale_mod_size=50, batch_size=batch_size)

    random.seed(42)  # 再現性のため（任意）

    # 10 クライアント & 100 クライアントの両方を実行
    for num_clients in (10, 1000):
        print(f"\n=== Federated average with N={num_clients} clients (vec_len={vec_len}) ===")

        # ランダム重みを生成
        all_weights = _gen_random_clients(num_clients, vec_len)

        # 平文の正解
        plain_avg = _plaintext_average(all_weights)
        plain_avg_str = "[" + ", ".join(f"{x:.8f}" for x in plain_avg) + "]"

        # 暗号化して平均（N=10 のときは内部計算もトレース表示）
        ciphertexts = []
        if num_clients == 10:
            # 逐次和の中間平均を「暗号化のまま」出して復号して確認（ノイズの進み方観察用）
            c_sum = None
            running_plain_sum = [0.0] * vec_len
            for idx, w in enumerate(all_weights, start=1):
                ct = encrypt_weights(cc, keys.publicKey, w)
                ciphertexts.append(ct)
                # sum を更新（暗号）
                if c_sum is None:
                    c_sum = ct
                else:
                    c_sum = cc.EvalAdd(c_sum, ct)

                # トレース: 現在の暗号平均（c_sum * 1/idx）を一時的に作って復号
                ct_avg_now = cc.EvalMult(c_sum, 1.0 / idx)
                dec_avg_now = decrypt_to_plaintext(cc, keys.secretKey, ct_avg_now, length=vec_len, precision=8)

                # 平文側の逐次平均（正解）も更新・表示
                for i, v in enumerate(w):
                    running_plain_sum[i] += v
                running_plain_avg = [sv / idx for sv in running_plain_sum]
                running_plain_avg_str = "[" + ", ".join(f"{x:.8f}" for x in running_plain_avg) + "]"

                print(f"[TRACE N=10] step {idx:2d}/{num_clients}: "
                      f"plain_avg_now={running_plain_avg_str} | dec_encrypted_avg_now={dec_avg_now}")
        else:
            # N=100 は最終結果のみ（内部トレースなし）
            ciphertexts = [encrypt_weights(cc, keys.publicKey, w) for w in all_weights]

        # 平均（暗号文）
        assert len(ciphertexts) == num_clients, "暗号文の数がクライアント数と一致していません"
        c_avg = federated_average_encrypted(cc, ciphertexts)

        # 復号結果
        dec_avg_str = decrypt_to_plaintext(cc, keys.secretKey, c_avg, length=vec_len, precision=8)

        # 正解 vs 復号の比較
        print("Plaintext correct avg =", plain_avg_str)
        print("Decrypted CKKS  avg   =", dec_avg_str)

    # メモ: 大きな N（例: 100）では和が大きくなるため、CKKS の丸め誤差が見えやすくなります。
    # 精度が気になる場合は scale_mod_size を上げる/値のレンジを狭くする等を検討してください。

