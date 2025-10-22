from math import log2, ceil
from openfhe import *


def _next_pow2(n: int) -> int:
    # CKKS の BatchSize は 2 の冪が扱いやすい。未指定なら最小の 2^k に切り上げる。
    return 1 << ceil(log2(max(1, n)))


def setup_ckks(
    mult_depth: int = 1,
    scale_mod_size: int = 50,
    batch_size: int | None = None,
    security_level: SecurityLevel = SecurityLevel.HEStd_128_classic,
):
    """
    CKKS用のCryptoContextと鍵を用意して返す（最小限）。
    戻り値: (cc, keys)
    """
    if batch_size is None:
        batch_size = _next_pow2(8)  # デフォルト8スロット相当

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(scale_mod_size)
    params.SetBatchSize(batch_size)
    params.SetSecurityLevel(security_level)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)  # 平文スカラー乗算のみでも安全側で用意
    return cc, keys


def encrypt_weights(cc, public_key, weights: list[float]):
    """
    1 クライアントの重みベクトルをパックして暗号化して返す。
    戻り値: Ciphertext
    """
    pt = cc.MakeCKKSPackedPlaintext(weights)
    return cc.Encrypt(public_key, pt)


def federated_average_encrypted(cc, ciphertexts: list):
    """
    暗号文のまま平均（和→1/N のスカラー乗算）。
    戻り値: 平均の Ciphertext
    """
    if not ciphertexts:
        raise ValueError("ciphertexts is empty")
    c_sum = ciphertexts[0]
    for ct in ciphertexts[1:]:
        c_sum = cc.EvalAdd(c_sum, ct)
    inv_n = 1.0 / len(ciphertexts)
    return cc.EvalMult(c_sum, inv_n)


def decrypt_to_plaintext(cc, secret_key, ct_avg, length: int, precision: int = 8) -> str:
    """
    平均の暗号文を復号し、表示用の整形済み文字列を返す。
    （プログラムで配列値が必要なら、Plaintext から取得する実装に差し替えてください）
    """
    pt = cc.Decrypt(ct_avg, secret_key)
    pt.SetLength(length)
    return pt.GetFormattedValues(precision)


# --- 使い方サンプル（最小限の関数呼び出しで動くデモ） ---
if __name__ == "__main__":
    # サンプル重み（長さは揃える）——実運用では各クライアント側で encrypt_weights を呼び出し、暗号文のみをサーバへ送る
    w1 = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
    w2 = [5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25]
    w3 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    w4 = [1.5, 1.25, 1.0, 0.75, 0.5, 0.4, 0.3, 0.2]
    w5 = [2.0, 2.1, 2.2, 2.3, 0.0, -0.1, -0.2, -0.3]
    all_weights = [w1, w2, w3, w4, w5]

    # BatchSize はベクトル長に合わせて 2 の冪へ（必要最小限の自動調整）
    vec_len = len(all_weights[0])
    batch_size = _next_pow2(vec_len)

    cc, keys = setup_ckks(mult_depth=1, scale_mod_size=50, batch_size=batch_size)

    # 各クライアント（ここでは同一プロセス内で代用）が公開鍵で暗号化
    cts = [encrypt_weights(cc, keys.publicKey, w) for w in all_weights]

    # サーバ側: 暗号文のまま平均
    c_avg = federated_average_encrypted(cc, cts)

    # （評価用）平均の復号・表示
    avg_str = decrypt_to_plaintext(cc, keys.secretKey, c_avg, length=vec_len, precision=8)
    print(f"Federated average over {len(cts)} clients:")
    print("avg(x) =", avg_str)

