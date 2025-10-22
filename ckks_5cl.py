from openfhe import *


def main():
    mult_depth = 1
    scale_mod_size = 50
    batch_size = 8

    parameters = CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(mult_depth)
    parameters.SetScalingModSize(scale_mod_size)
    parameters.SetBatchSize(batch_size)

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    print("The CKKS scheme is using ring dimension: " + str(cc.GetRingDimension()))

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, [1, -2])  # 最小変更: 残しておく（未使用でも支障なし）

    # --- 複数クライアントの重みベクトル（全て長さ batch_size） ---
    x1 = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
    x2 = [5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25]
    x3 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    x4 = [1.5, 1.25, 1.0, 0.75, 0.5, 0.4, 0.3, 0.2]
    x5 = [2.0, 2.1, 2.2, 2.3, 0.0, -0.1, -0.2, -0.3]

    client_weights = [x1, x2, x3, x4, x5]

    # 平文→暗号化（packed）
    encrypted_clients = []
    for w in client_weights:
        pt = cc.MakeCKKSPackedPlaintext(w)
        encrypted_clients.append(cc.Encrypt(keys.publicKey, pt))

    # ===== 連合平均 (encrypted) =====
    c_sum = encrypted_clients[0]
    for ct in encrypted_clients[1:]:
        c_sum = cc.EvalAdd(c_sum, ct)

    inv_n = 1.0 / len(encrypted_clients)
    c_avg = cc.EvalMult(c_sum, inv_n)

    # 復号・表示
    pt_avg = cc.Decrypt(c_avg, keys.secretKey)
    pt_avg.SetLength(batch_size)
    print("\nFederated average over", len(encrypted_clients), "clients:")
    print("avg(x) = " + pt_avg.GetFormattedValues(8))


if __name__ == "__main__":
    main()

