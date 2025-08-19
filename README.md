# part-trait-regression-mit
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

このリポジトリは、**車のパーツが各イメージワード（Trait）に与える影響を回帰分析で評価し、結果を可視化するサンプル**をまとめたものです。

主に以下のことができます：

- 車ごとのイメージワードの影響を回帰分析
- 回帰係数（Coef）と有意性（logP = -log10(P値)）の算出
- ヒートマップ・散布図・棒グラフ・レーダーチャートの生成
- CSV 出力（横長形式・レーダー用形式）による結果整理

> 少ないサンプルでも傾向をつかむことができ、マーケティングや商品企画の意思決定に役立てることができます。

---

## 環境
- Python 3.x
- 必要ライブラリ: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `seaborn`, `matplotlib`, `japanize_matplotlib`

---

## フォルダ構成
```
├─ 1_flow/
│   └─ part-trait-regression.py       # 実行用スクリプト
├─ 2_data/
│   └─ area_image.csv           # 入力データ
├─ 3_output/                    # データ出力先（スクリプト実行時に自動作成）
```


---

## 入力データフォーマット例
`2_data/area_image.csv` を参照ください。

---

## 使い方

### 1. CSVを準備
`2_data/area_image.csv` にモニターの回答データを置きます。

### 2. スクリプト実行
```bash
python 1_flow/main.py
```

### 3. 処理内容
- 車ごとに Trait × Part の回帰分析
- 回帰係数（Coef）と有意性（logP = -log10(P値)）を算出
- ヒートマップ、散布図、棒グラフ、レーダーチャートを生成
- CSV 出力（横長形式やレーダー用形式）で整理
- すべての結果は 3_output/ に保存されます


# グラフ出力
- ヒートマップ
  - 行: Part / 列: Trait
  - 値: Coef / logP
  - 色の濃淡で係数の大小や信頼度を直感的に把握

- 散布図・棒グラフ
  - 散布図: Coef（X軸） vs logP（Y軸）、色は Coef
  - 棒グラフ: 各 Part ごとの Coef、色で logP を表現

- レーダーチャート
  - Trait ごとに Part の影響をレーダー形式で表示
  - 上段: Coef、下段: logP


# 今後の拡張予定
- データ整形や集計精度の向上
- 可視化のカスタマイズ性向上
- 他カテゴリの分析対応

# 貢献方法
- バグ報告や機能追加の提案は Issues で
- コード改善や新機能追加は Pull Request で
- ドキュメント改善や翻訳も歓迎


## LICENSE
MIT License（詳細はLICENSEファイルをご参照ください）

#### 開発者： iwakazusuwa(Swatchp)
<img width="254" height="126" alt="image" src="https://github.com/user-attachments/assets/fd2c55e2-7d50-4fb4-8610-50ea29baee42" />
