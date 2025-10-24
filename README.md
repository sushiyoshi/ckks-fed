# CKKS Federated Learning

OpenFHE CKKSによる連合学習実装

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
