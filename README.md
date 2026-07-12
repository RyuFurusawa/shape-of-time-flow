# Shape_of_time_flow

IMGTrans GUI v2 (`Shape_of_time_flow.py`)。PyQt5 製の GUI で、サンプル画像生成パネル
（4方向グラデーション / 50%均一 / ランダムの6パターン）や、映像情報からの
パラメータ自動初期化などを備える。

## セットアップ（venv 推奨）

クリーンな仮想環境を作ってから依存をインストールする:

```bash
# 仮想環境を作成・有効化
python -m venv .venv
source .venv/bin/activate        # Windows は .venv\Scripts\activate

# サードパーティ依存
pip install -r requirements.txt

# imgtrans（drawManeuver）— 公開 PyPI パッケージではなくローカルの imgtrans コード。
# `pip install imgtrans` は別人の無関係なパッケージなので使わないこと。
pip install git+https://github.com/ryufurusawa/imgtrans.git
```

> 注意: `imgtrans` は numba / av(PyAV) / librosa などの重い依存を含み、
> システムに FFmpeg（`ffmpeg` と `ffprobe`）が PATH 上にある必要がある。

## 実行

```bash
python Shape_of_time_flow.py
```
