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


def decrypt_to_plaintext(cc, secret_key, ct_avg, length: int, precision: int = 8) -> str:
    pt = cc.Decrypt(ct_avg, secret_key)
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

    # CKKS 文脈を一度用意（深さ1: 和 + 平文スカラー乗算のみ）
    cc, keys = setup_ckks(mult_depth=1, scale_mod_size=50, batch_size=batch_size)

    # 乱数の再現性が欲しければ seed を固定（任意）
    random.seed(42)

    # 10クライアント と 100クライアント のケースをそれぞれ評価
    for num_clients in (10, 100):
        print(f"\n=== Federated average with N={num_clients} clients (vec_len={vec_len}) ===")

        # ランダム重みを生成（平文での正解計算に使用）
        all_weights = _gen_random_clients(num_clients, vec_len)
        plain_avg = _plaintext_average(all_weights)

        # 暗号化して送られてきた想定で平均（暗号化のまま）
        cts = [encrypt_weights(cc, keys.publicKey, w) for w in all_weights]
        c_avg = federated_average_encrypted(cc, cts)

        # 復号して表示（CKKS の近似結果）
        dec_avg_str = decrypt_to_plaintext(cc, keys.secretKey, c_avg, length=vec_len, precision=8)

        # 平文での正解も表示（比較用）
        # ここでは見やすさのために丸めた文字列を出す（必要ならそのままのリストを print してもよい）
        plain_avg_str = "[" + ", ".join(f"{x:.8f}" for x in plain_avg) + "]"

        print("Plaintext correct avg =", plain_avg_str)
        print("Decrypted CKKS  avg   =", dec_avg_str)

    # 参考: 大きな N で「和」が大きくなるほど CKKS のノイズ/丸め誤差の影響が見えやすくなります。
    # 必要に応じて scale_mod_size を上げる / 初期スケールを調整する / 再スケール戦略を見直すと精度改善が見込めます。

