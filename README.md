# Shape_of_time_flow

IMGTrans GUI v2 (`Shape_of_time_flow.py`)。PyQt5 製の GUI で、サンプル画像生成パネル
（4方向グラデーション / 50%均一 / ランダムの6パターン）や、映像情報からの
パラメータ自動初期化などを備える。

## 依存

サードパーティ:

```bash
pip install -r requirements.txt
```

加えて、`imgtrans`（`drawManeuver`）が必要。これは公開 PyPI パッケージではなく
ローカルの imgtrans コードを指す。インストール例:

```bash
# imgtrans_cloude を pip でインストール
pip install git+https://github.com/ryufurusawa/imgtrans.git
```

## 実行

```bash
python Shape_of_time_flow.py
```
