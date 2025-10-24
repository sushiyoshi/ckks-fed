# CKKS Federated Learning

OpenFHE CKKSによる連合学習実装

## 概要

このプロジェクトは、準同型暗号（CKKS方式）を用いた連合学習システムです。線形分離可能な2クラス分類問題（ルール: x₁ - x₂ + x₃ - x₄ + ... ± xₙ > 0）を学習するニューラルネットワークの重みを、暗号化したまま集約します。各クライアントがローカルで学習した重みを暗号化してサーバに送信し、サーバは暗号文のまま平均化を行うことでプライバシーを保護します。平文での連合学習との精度・実行時間の比較も実装されています。

## セットアップ

```bash
python -m venv .
source bin/activate
pip install -r requirements.txt
```

## 実行

```bash
# デフォルト設定で実行
python federated.py

# 引数を指定して実行
python federated.py --input-size 5 --num-clients 20 --num-rounds 15
```

### 引数

- `--input-size`: 入力特徴量の数（デフォルト: 3）
- `--num-clients`: クライアント数（デフォルト: 10）
- `--num-rounds`: 学習ラウンド数（デフォルト: 10）
