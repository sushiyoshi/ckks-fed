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
    cc.EvalRotateKeyGen(keys.secretKey, [1, -2])

    # --- sample client weights (各クライアントの重みベクトルの例) ---
    x1 = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
    x2 = [5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25]

    # 平文→CKKSパック
    ptx1 = cc.MakeCKKSPackedPlaintext(x1)
    ptx2 = cc.MakeCKKSPackedPlaintext(x2)

    print("Input x1: " + str(ptx1))
    print("Input x2: " + str(ptx2))

    # 暗号化
    c1 = cc.Encrypt(keys.publicKey, ptx1)
    c2 = cc.Encrypt(keys.publicKey, ptx2)

    # ====== ここから最小追加：暗号化されたまま重み平均（連合学習用） ======
    # 複数クライアントの暗号文リスト（ここでは c1, c2）
    encrypted_clients = [c1, c2]  # ← 他クライアントがあればここに追加するだけ

    # 1) 和を計算（暗号文のまま加算を畳み込み）
    c_sum = encrypted_clients[0]
    for ct in encrypted_clients[1:]:
        c_sum = cc.EvalAdd(c_sum, ct)

    # 2) 1/N を平文スカラーとして掛けて平均化
    inv_n = 1.0 / len(encrypted_clients)
    c_avg = cc.EvalMult(c_sum, inv_n)

    # 3) 復号して表示
    pt_avg = cc.Decrypt(c_avg, keys.secretKey)
    pt_avg.SetLength(batch_size)
    print("\nFederated average (encrypted):")
    print("avg(x) = " + pt_avg.GetFormattedValues(8))
if __name__ == "__main__":
    main()

