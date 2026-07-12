# imgtrans-gui

IMGTrans GUI v2 (`test20260526.py`)。PyQt5 製の GUI で、サンプル画像生成パネル
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
# imgtrans_cloude 側を pip インストール可能にした上で
pip install -e /path/to/imgtrans_cloude
```

## 実行

```bash
python test20260526.py
```
