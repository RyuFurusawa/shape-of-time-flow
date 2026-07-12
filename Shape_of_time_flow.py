"""IMGTrans GUI v2 (2026-05-26)

test20251105.py からの主な変更:

- サンプル画像生成パネルを追加
  - 6種類のパターン (4方向グラデーション + 50%均一 + ランダム)
  - サイズはスキャン方向 (映像と自動一致) / 時間方向 (カスタム) の 2 ボックス
  - Slit 方向に応じて自動で正しい向きの 16bit PNG を出力
  - 生成後に該当する Space/Time/Rate スロットへ自動セット
  - ファイル名は img_to_maneuver の規約 (space_W.png / time_VMIN-VMAX.png / rate_DEV.png) に従う
- Initialize 直後に各パラメータ欄を映像情報から賢く初期化
"""

import sys
import os
import re
import time
import subprocess
from pathlib import Path

# Continue normal imports
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,
    QComboBox, QTextEdit, QCheckBox, QMessageBox, QSpinBox, QHBoxLayout,
    QFrame, QDoubleSpinBox, QGroupBox, QTabWidget, QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl
from PyQt5.QtGui import QImage, QPixmap, QMovie, QImageReader

# 動画の内蔵再生 (QtMultimedia) は環境により無い場合があるため防御的に import
try:
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    from PyQt5.QtMultimediaWidgets import QVideoWidget
    HAS_MULTIMEDIA = True
except Exception:
    HAS_MULTIMEDIA = False
import numpy as np
import cv2
from PIL import Image

from imgtrans import drawManeuver


# ======== Sample image generator ========
PATTERN_LABELS = [
    "上→下: 白→黒",
    "上→下: 黒→白",
    "左→右: 白→黒",
    "左→右: 黒→白",
    "50% グレー均一",
    "ランダムノイズ",
    "波形 (Wave) ※ 振幅/周期/位相 編集",
]
PATTERN_IDS = [
    "v_w2b", "v_b2w", "h_w2b", "h_b2w", "solid_gray", "random", "wave",
]

# 各セクション (space/time/rate) の「通常再生」に相当するパターン。
#   space: 左→右 黒→白グラデーション (h_b2w) = 空間を素通し (等倍マッピング)
#   time : 上→下 黒→白グラデーション (v_b2w) = 時間が線形に流れる (等速)
#   rate : 50% グレー均一 (solid_gray)        = 再生レート一定 (等速)
# この pattern をセクションの選択肢の先頭に置き、ラベル末尾に「（通常再生）」を付す。
SECTION_NORMAL_PATTERN = {
    "space": "h_b2w",
    "time": "v_b2w",
    "rate": "solid_gray",
}


def section_pattern_order(type_name):
    """セクション {type_name} 用の (pattern_ids, labels) を返す。

    「通常再生」に相当する pattern を先頭 (index 0) に移動し、そのラベル末尾に
    「（通常再生）」を付与する。残りは元の PATTERN_IDS 順を維持。
    """
    normal = SECTION_NORMAL_PATTERN.get(type_name, PATTERN_IDS[0])
    ordered_ids = [normal] + [pid for pid in PATTERN_IDS if pid != normal]
    labels = []
    for pid in ordered_ids:
        base = PATTERN_LABELS[PATTERN_IDS.index(pid)]
        if pid == normal:
            base = f"{base}（通常再生）"
        labels.append(base)
    return ordered_ids, labels


def render_pattern(h_pix, w_pix, pattern_id, **wave_params):
    """16bit uint16 (H, W) のグレースケール画像を生成する。

    pattern_id="wave" の場合は wave_params で:
        direction  : "v" (上下方向に変化) / "h" (左右方向に変化)
        amplitude  : 0.0 - 1.0 (full-range の割合, 1.0 で 0..65535 振り切る)
        period     : 1サイクルのピクセル数 (例: H==period でちょうど1周期)
        phase_deg  : 開始位相 (度, 0..360)
    """
    h, w = int(h_pix), int(w_pix)
    if pattern_id == "v_w2b":
        col = np.linspace(65535, 0, h, dtype=np.float32)
        img = np.broadcast_to(col[:, None], (h, w)).astype(np.uint16)
    elif pattern_id == "v_b2w":
        col = np.linspace(0, 65535, h, dtype=np.float32)
        img = np.broadcast_to(col[:, None], (h, w)).astype(np.uint16)
    elif pattern_id == "h_w2b":
        row = np.linspace(65535, 0, w, dtype=np.float32)
        img = np.broadcast_to(row[None, :], (h, w)).astype(np.uint16)
    elif pattern_id == "h_b2w":
        row = np.linspace(0, 65535, w, dtype=np.float32)
        img = np.broadcast_to(row[None, :], (h, w)).astype(np.uint16)
    elif pattern_id == "solid_gray":
        img = np.full((h, w), 32767, dtype=np.uint16)
    elif pattern_id == "random":
        rng = np.random.default_rng()
        img = rng.integers(0, 65536, size=(h, w), dtype=np.uint16)
    elif pattern_id == "wave":
        direction = wave_params.get("direction", "v")
        amp = float(wave_params.get("amplitude", 1.0))      # 0..1
        period = max(1.0, float(wave_params.get("period", max(h, 1))))
        phase = np.deg2rad(float(wave_params.get("phase_deg", 0.0)))
        # 0..1 正規化された sin 波 → 16bit
        mid = 32767.5
        amp_scaled = amp * 32767.5
        if direction == "v":
            axis = np.arange(h, dtype=np.float64)
            wave1d = mid + amp_scaled * np.sin(2 * np.pi * axis / period + phase)
            col = np.clip(wave1d, 0, 65535)
            img = np.broadcast_to(col[:, None], (h, w)).astype(np.uint16)
        else:  # "h"
            axis = np.arange(w, dtype=np.float64)
            wave1d = mid + amp_scaled * np.sin(2 * np.pi * axis / period + phase)
            row = np.clip(wave1d, 0, 65535)
            img = np.broadcast_to(row[None, :], (h, w)).astype(np.uint16)
    else:
        raise ValueError(f"Unknown pattern_id: {pattern_id}")
    return img


def generate_sample_image(out_dir, image_type, pattern_id,
                          scan_size, time_size,
                          scan_direction,
                          space_range=None, time_vmin=None, time_vmax=None,
                          rate_maxdev=None,
                          wave_direction="v", wave_amplitude=1.0,
                          wave_period=None, wave_phase_deg=0.0):
    """サンプル画像を生成してパスを返す。

    image_type: "space" / "time" / "rate"
    scan_direction: 1=vertical slit, 0=horizontal slit
        - vertical:   file shape (H, W) = (time_size, scan_size)
        - horizontal: file shape (H, W) = (scan_size, time_size)  ※img_to_maneuver が .T するため

    pattern_id == "wave" の場合の追加パラメータ:
        wave_direction : "v"(上下) or "h"(左右)
        wave_amplitude : 0.0 - 1.0
        wave_period    : ピクセル数 (None なら該当軸サイズと同じ → 1周期)
        wave_phase_deg : 度 (0..360)
    """
    # ファイル名は img_to_maneuver の extract_params_from_filename 規約に従う
    if image_type == "space":
        if space_range is None:
            space_range = scan_size
        fname = f"sample_space_{int(space_range)}.png"
    elif image_type == "time":
        if time_vmin is None: time_vmin = 0
        if time_vmax is None: time_vmax = 100
        fname = f"sample_time_{int(time_vmin)}-{int(time_vmax)}.png"
    elif image_type == "rate":
        if rate_maxdev is None: rate_maxdev = 0.5
        fname = f"sample_rate_{rate_maxdev}.png"
    else:
        raise ValueError(f"image_type must be space/time/rate, got {image_type!r}")

    # Slit 方向に応じてファイル形状を決定
    if int(scan_direction) == 1:
        h_pix, w_pix = int(time_size), int(scan_size)   # (time, scan)
    else:
        h_pix, w_pix = int(scan_size), int(time_size)   # (scan, time) — .T される

    # Wave のデフォルト period (該当軸サイズ)
    if pattern_id == "wave" and wave_period is None:
        wave_period = h_pix if wave_direction == "v" else w_pix

    img16 = render_pattern(
        h_pix, w_pix, pattern_id,
        direction=wave_direction,
        amplitude=wave_amplitude,
        period=wave_period,
        phase_deg=wave_phase_deg,
    )

    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, img16)
    return out_path


# ======== Rendered video preview widget ========
class VideoPreview(QWidget):
    """レンダリング結果の動画を内蔵再生するウィジェット。

    - QtMultimedia がある場合: QVideoWidget + QMediaPlayer で再生 (縦横比保持・ループ再生)
    - 無い場合: パス表示 + 「外部プレイヤーで開く」ボタンにフォールバック
    """
    def __init__(self, base_title):
        super().__init__()
        self._base_title = base_title
        self.path = None
        self.loaded = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(base_title)
        self.title_label.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(self.title_label)

        if HAS_MULTIMEDIA:
            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(220)
            # 縦横比を崩さない (レターボックス表示)
            self.video_widget.setAspectRatioMode(Qt.KeepAspectRatio)
            v.addWidget(self.video_widget)

            self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
            self.player.setVideoOutput(self.video_widget)
            self.player.setMuted(True)
            self.player.mediaStatusChanged.connect(self._on_status)

            ctl = QHBoxLayout()
            self.play_btn = QPushButton("⏸ 一時停止")
            self.play_btn.clicked.connect(self._toggle)
            ctl.addWidget(self.play_btn)
            self.open_btn = QPushButton("外部プレイヤーで開く")
            self.open_btn.clicked.connect(self._open_external)
            ctl.addWidget(self.open_btn)
            ctl.addStretch()
            v.addLayout(ctl)
        else:
            self.info_label = QLabel("(QtMultimedia が無いため内蔵再生できません)")
            self.info_label.setWordWrap(True)
            self.info_label.setStyleSheet("color: #a66; font-size: 11px;")
            v.addWidget(self.info_label)
            self.open_btn = QPushButton("外部プレイヤーで開く")
            self.open_btn.clicked.connect(self._open_external)
            v.addWidget(self.open_btn)

        self.setVisible(False)

    def load(self, path):
        """path の動画を読み込み、あれば表示 + 自動再生。無ければ非表示。"""
        self.path = path
        if not (path and os.path.exists(path)):
            self.loaded = False
            self.setVisible(False)
            return
        self.loaded = True
        self.title_label.setText(f"{self._base_title}: {os.path.basename(path)}")
        if HAS_MULTIMEDIA:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(path))))
            self.player.play()
            self.play_btn.setText("⏸ 一時停止")
        self.setVisible(True)

    def stop(self):
        if HAS_MULTIMEDIA:
            try:
                self.player.stop()
            except Exception:
                pass

    def _on_status(self, status):
        # 末尾まで再生したら先頭へ戻してループ
        if status == QMediaPlayer.EndOfMedia:
            self.player.setPosition(0)
            self.player.play()

    def _toggle(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.play_btn.setText("▶ 再生")
        else:
            self.player.play()
            self.play_btn.setText("⏸ 一時停止")

    def _open_external(self):
        if not (self.path and os.path.exists(self.path)):
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", self.path])
            elif os.name == "nt":
                os.startfile(self.path)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", self.path])
        except Exception:
            pass


# ======== Worker thread ========
class RenderWorker(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str, str)  # success, video_path, anim_path

    def __init__(self, dm, mode, animout,
                 space_img, time_img, rate_img,
                 duration,
                 space_set=None, time_vmin=None, time_vmax=None, rate_maxdev=None,
                 anim_only=False, rate_baseline=None, rate_startpoint=None):
        super().__init__()
        self.dm = dm
        self.mode = mode
        self.animout = animout
        self.space_img = space_img
        self.time_img = time_img
        self.rate_img = rate_img
        self.duration = duration
        self.space_set = space_set
        self.time_vmin = time_vmin
        self.time_vmax = time_vmax
        self.rate_maxdev = rate_maxdev
        self.rate_baseline = rate_baseline
        self.rate_startpoint = rate_startpoint
        self.anim_only = anim_only

    def run(self):
        try:
            bm = self.dm
            if self.anim_only:
                self.emit("=== Animation-only mode started ===")
                anim_path = self.run_animation_only(bm)
                self.done_signal.emit(True, "", anim_path or "")
                return

            self.emit("=== Rendering process started ===")

            if self.mode == "time to data":
                bm.img_to_maneuver(
                    space_img_path=self.space_img,
                    time_img_path=self.time_img,
                    space_set=self.space_set,
                    vrange=[self.time_vmin, self.time_vmax]
                )
            elif self.mode == "rate to data":
                bm.img_to_maneuver_rate_based(
                    time_rate_path=self.rate_img,
                    space_img_path=self.space_img,
                    space_set=self.space_set,
                    rate_range=self.rate_maxdev,
                    rate_baseline=self.rate_baseline,
                    rate_startpoint=self.rate_startpoint,
                )
            else:
                self.emit("[ERROR] Invalid mode.")
                self.done_signal.emit(False, "", "")
                return

            bm.zPointCheck()
            bm.maneuver_imgplot("all")

            video_path = ""
            anim_path = ""
            if self.animout:
                bm.new_transprocess(del_data=False)
                video_path = self._resolve_video_path(bm)
                out_fps = 10
                dynamic_frames = int(self.duration * out_fps)
                self.emit(f"out_framenums={dynamic_frames} )")
                ts_anim = time.time() - 0.5
                bm.animationout_custome(
                    zRangeFix=False,
                    out_fps=out_fps,
                    aspect_ratio=(16, 50, 9),
                    colormode='white',
                    transparent=False,
                    gridplot=True,
                    drawLineNum=bm.width // 10,
                    dpi=300,
                    out_framenums=dynamic_frames
                )
                anim_path = self._find_anim_output(ts_anim)
            else:
                bm.new_transprocess(del_data=False)
                video_path = self._resolve_video_path(bm)

            self.done_signal.emit(True, video_path, anim_path)

        except Exception as e:
            self.emit(f"[ERROR] Rendering failed: {str(e)}")
            self.done_signal.emit(False, "", "")

    def run_animation_only(self, bm):
        try:
            out_fps = 10
            dynamic_frames = int(self.duration * out_fps)
            self.emit(f"out_framenums={dynamic_frames} )")

            ts_anim = time.time() - 0.5
            bm.animationout_custome(
                zRangeFix=False,
                out_fps=out_fps,
                aspect_ratio=(16, 50, 9),
                colormode='white',
                transparent=False,
                gridplot=True,
                drawLineNum=bm.width // 10,
                dpi=300,
                out_framenums=dynamic_frames
            )
            return self._find_anim_output(ts_anim)
        except Exception as e:
            self.emit(f"[ERROR] Animation output failed: {e}")
            return ""

    @staticmethod
    def _resolve_video_path(bm):
        """new_transprocess が設定する out_videopath を絶対パスで返す。"""
        p = getattr(bm, "out_videopath", "") or ""
        if p and not os.path.isabs(p):
            p = os.path.abspath(p)
        return p if (p and os.path.exists(p)) else ""

    @staticmethod
    def _find_anim_output(since_ts):
        """animationout_custome が出力した *_img_3d-pixelMap.mp4 を mtime で検出。"""
        cwd = os.getcwd()
        best = ("", -1.0)
        for f in os.listdir(cwd):
            if not f.lower().endswith("_img_3d-pixelmap.mp4"):
                continue
            full = os.path.join(cwd, f)
            if not os.path.isfile(full):
                continue
            mt = os.path.getmtime(full)
            if mt >= since_ts and mt > best[1]:
                best = (os.path.abspath(full), mt)
        return best[0]

    def emit(self, text):
        safe_text = str(text).encode("ascii", "ignore").decode("ascii")
        self.log_signal.emit(safe_text)


# ======== Maneuver preview worker (2D plot + 3D anim GIF) ========
class ManeuverPreviewWorker(QThread):
    """Time+Space または Rate+Space が揃った時点で軽量プレビューを生成する。

    流れ:
      1. img_to_maneuver / img_to_maneuver_rate_based で data を構築
      2. zPointCheck で検証
      3. maneuver_2dplot で 2D PNG 生成
      4. maneuver_3dplot で短尺 3D MP4 生成 (out_framenums/dpi を低めに)
      5. ffmpeg で MP4 → GIF 変換
      6. 出力 (PNG path, GIF path) を done_signal で通知
    """
    progress_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str, str)  # success, plot2d_path, gif_path

    def __init__(self, dm, mode, space_img, time_img, rate_img,
                 space_set, time_vmin, time_vmax,
                 rate_maxdev, rate_baseline, rate_startpoint,
                 anim_frames=20, anim_fps=10, anim_dpi=80):
        super().__init__()
        self.dm = dm
        self.mode = mode  # "time" or "rate"
        self.space_img = space_img
        self.time_img = time_img
        self.rate_img = rate_img
        self.space_set = space_set
        self.time_vmin = time_vmin
        self.time_vmax = time_vmax
        self.rate_maxdev = rate_maxdev
        self.rate_baseline = rate_baseline
        self.rate_startpoint = rate_startpoint
        self.anim_frames = anim_frames
        self.anim_fps = anim_fps
        self.anim_dpi = anim_dpi

    @staticmethod
    def _latest_file(cwd, suffixes, since_ts):
        """cwd の中で suffixes のいずれかに合致し、mtime >= since_ts のうち最新のフルパスを返す。
        無ければ "" を返す。同名ファイルの上書きケースでも mtime が更新されているため検出される。
        """
        suffixes = tuple(s.lower() for s in suffixes)
        candidates = []
        for f in os.listdir(cwd):
            full = os.path.join(cwd, f)
            if not os.path.isfile(full):
                continue
            if not f.lower().endswith(suffixes):
                continue
            mt = os.path.getmtime(full)
            if mt >= since_ts:
                candidates.append((mt, full))
        if not candidates:
            return ""
        candidates.sort(reverse=True)
        return candidates[0][1]

    def run(self):
        try:
            cwd = os.getcwd()

            self.progress_signal.emit("img_to_maneuver: ロード中…")
            if self.mode == "time":
                self.dm.img_to_maneuver(
                    space_img_path=self.space_img,
                    time_img_path=self.time_img,
                    space_set=self.space_set,
                    vrange=[self.time_vmin, self.time_vmax],
                )
            else:
                self.dm.img_to_maneuver_rate_based(
                    time_rate_path=self.rate_img,
                    space_img_path=self.space_img,
                    space_set=self.space_set,
                    rate_range=self.rate_maxdev,
                    rate_baseline=self.rate_baseline,
                    rate_startpoint=self.rate_startpoint,
                )

            self.progress_signal.emit("zPointCheck…")
            self.dm.zPointCheck()

            # 2D プロット生成: mtime で「呼び出し後に変更されたファイル」を検出
            # (同じファイル名で上書きされるケースに対応するため set 差分は使わない)
            ts_2d = time.time() - 0.5  # 小さなクロックスラックを許容
            self.progress_signal.emit("maneuver_2dplot: 2D プロット生成中…")
            self.dm.maneuver_2dplot()
            plot2d = self._latest_file(cwd, (".png",), ts_2d)

            # 3D アニメ生成: 同じく mtime で検出
            ts_3d = time.time() - 0.5
            self.progress_signal.emit(
                f"maneuver_3dplot: 3D アニメ生成中 ({self.anim_frames} frames @ {self.anim_dpi} dpi)…"
            )
            self.dm.maneuver_3dplot(
                out_framenums=self.anim_frames,
                out_fps=self.anim_fps,
                dpi=self.anim_dpi,
            )
            mp4 = self._latest_file(cwd, (".mp4", ".mov"), ts_3d)

            gif = ""
            if mp4 and os.path.exists(mp4):
                gif = os.path.splitext(mp4)[0] + "_preview.gif"
                self.progress_signal.emit("ffmpeg で GIF 変換…")
                # 横幅 400 にスケール (高さは自動)、ループ無限
                cmd = ["ffmpeg", "-y", "-i", mp4,
                       "-vf", f"fps={self.anim_fps},scale=400:-1:flags=lanczos",
                       "-loop", "0", gif]
                proc = subprocess.run(cmd, capture_output=True)
                if proc.returncode != 0:
                    self.progress_signal.emit(f"[WARN] GIF 変換失敗: {proc.stderr.decode('utf-8', 'ignore')[:200]}")
                    gif = ""

            self.progress_signal.emit("完了")
            self.done_signal.emit(True, plot2d, gif)
        except Exception as e:
            self.progress_signal.emit(f"[ERROR] {e}")
            self.done_signal.emit(False, "", "")


# ======== Main GUI ========
class IMGTransApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMGTrans GUI v2 (2026-05-26)")
        self.resize(760, 820)
        self.setMinimumSize(640, 480)

        self.videopath = None
        self.space_img_path = None
        self.time_img_path = None
        self.rate_img_path = None
        self.dm = None
        self.worker = None
        self.render_completed = False
        self._preview_stale = False

        self.init_ui()
        self.update_ui_state("initial")

    # --- UI Setup ---
    def _wrap_scroll(self, widget):
        """タブのコンテンツ widget を QScrollArea で包む (縦に長くてもスクロール可能)"""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def init_ui(self):
        # --- Video file ---
        self.video_label = QLabel("No video file selected")
        self.video_btn = QPushButton("Select Video File")
        self.video_btn.clicked.connect(self.select_video)

        # --- Slit toggle ---
        self.slit_toggle = QCheckBox("Vertical (check for vertical)")
        self.slit_label = QLabel("Slit direction: horizontal")
        self.slit_toggle.stateChanged.connect(self.update_slit_label)

        # --- Initialize ---
        self.init_btn = QPushButton("Initialize")
        self.init_btn.clicked.connect(self.initialize_drawmaneuver)
        self.info_label = QLabel("Video info: (not initialized)")

        # ===== 共通サイズ設定 (Space/Time/Rate で共有) =====
        # img_to_maneuver は space と time/rate 画像の形状一致を要求するため、サイズは共有。
        # パターン/波形/プレビューは各セクション (Space/Time/Rate) に個別配置。
        self.gen_group = QGroupBox("共通サイズ設定 (Shared Image Size)")
        gen_v = QVBoxLayout(self.gen_group)

        s_layout = QHBoxLayout()
        s_layout.addWidget(QLabel("スキャン方向サイズ:"))
        self.gen_scan_size = QSpinBox()
        self.gen_scan_size.setRange(16, 32768)
        self.gen_scan_size.setValue(1920)
        s_layout.addWidget(self.gen_scan_size)
        s_layout.addWidget(QLabel("(映像幅から自動)"))
        gen_v.addLayout(s_layout)

        t2_layout = QHBoxLayout()
        t2_layout.addWidget(QLabel("時間方向サイズ:"))
        self.gen_time_size = QSpinBox()
        self.gen_time_size.setRange(2, 32768)
        self.gen_time_size.setValue(120)
        t2_layout.addWidget(self.gen_time_size)
        t2_layout.addWidget(QLabel("(任意のフレーム数)"))
        gen_v.addLayout(t2_layout)

        self.gen_hint = QLabel("")
        self.gen_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.gen_hint.setWordWrap(True)
        gen_v.addWidget(self.gen_hint)

        self.gen_group.setVisible(False)

        # サイズ変更時は全セクションのプレビューを更新
        self.gen_scan_size.valueChanged.connect(self._update_gen_hint)
        self.gen_time_size.valueChanged.connect(self._update_gen_hint)
        self.gen_scan_size.valueChanged.connect(self._update_all_section_previews)
        self.gen_time_size.valueChanged.connect(self._update_all_section_previews)

        # 各セクション (Space/Time/Rate) の独立ジェネレータ widget bundle を保持
        self._section_gens = {}

        # --- Space image (select + per-section generator が下部) ---
        # 生成ボタンはセクション内のジェネレータパネルに統合
        space_btn_row = QHBoxLayout()
        self.space_btn = QPushButton("Select Space Image")
        self.space_btn.clicked.connect(lambda: self.select_image('space'))
        space_btn_row.addWidget(self.space_btn)
        self.space_btn_row = space_btn_row
        self.space_label = QLabel("No space image selected")

        sp_layout = QHBoxLayout()
        sp_label = QLabel("space range:")
        self.space_set_value = QSpinBox()
        self.space_set_value.setRange(0, 999999)
        sp_layout.addWidget(sp_label)
        sp_layout.addWidget(self.space_set_value)

        self.space_info_label = QLabel("")
        self.space_info_label.setStyleSheet("color: gray; font-size: 12px;")

        self.space_param_frame = QFrame()
        sp_vbox = QVBoxLayout(self.space_param_frame)
        sp_vbox.addLayout(sp_layout)
        sp_vbox.addWidget(self.space_info_label)
        self.space_param_frame.setVisible(False)

        # Space 用のジェネレータパネル (パターン / 波形エディタ / プレビュー)
        self.space_gen_frame = self._build_section_gen('space')

        # --- Time image (select + per-section generator が下部) ---
        time_btn_row = QHBoxLayout()
        self.time_btn = QPushButton("Select Time Image")
        self.time_btn.clicked.connect(lambda: self.select_image('time'))
        time_btn_row.addWidget(self.time_btn)
        self.time_btn_row = time_btn_row
        self.time_label = QLabel("No time image selected")

        time_layout = QHBoxLayout()
        self.time_vmin_spin = QSpinBox()
        self.time_vmax_spin = QSpinBox()
        self.time_vmin_spin.setRange(-999999, 999999)
        self.time_vmax_spin.setRange(-999999, 999999)
        time_layout.addWidget(QLabel("vmin:"))
        time_layout.addWidget(self.time_vmin_spin)
        time_layout.addWidget(QLabel("vmax:"))
        time_layout.addWidget(self.time_vmax_spin)

        self.time_info_label = QLabel("")
        self.time_info_label.setStyleSheet("color: gray; font-size: 12px;")

        self.time_param_frame = QFrame()
        time_vbox = QVBoxLayout(self.time_param_frame)
        time_vbox.addLayout(time_layout)
        time_vbox.addWidget(self.time_info_label)
        self.time_param_frame.setVisible(False)

        # Time 用のジェネレータパネル
        self.time_gen_frame = self._build_section_gen('time')

        # --- Rate image (select + per-section generator が下部) ---
        rate_btn_row = QHBoxLayout()
        self.rate_btn = QPushButton("Select Rate Image")
        self.rate_btn.clicked.connect(lambda: self.select_image('rate'))
        rate_btn_row.addWidget(self.rate_btn)
        self.rate_btn_row = rate_btn_row
        self.rate_label = QLabel("No rate image selected")

        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("baseline:"))
        self.rate_baseline_spin = QDoubleSpinBox()
        self.rate_baseline_spin.setRange(0.0, 999999.0)
        self.rate_baseline_spin.setDecimals(3)
        rate_layout.addWidget(self.rate_baseline_spin)
        rate_layout.addWidget(QLabel("max_range:"))
        self.rate_maxdev_spin = QDoubleSpinBox()
        self.rate_maxdev_spin.setRange(0.0, 999999.0)
        self.rate_maxdev_spin.setDecimals(3)
        rate_layout.addWidget(self.rate_maxdev_spin)
        rate_layout.addWidget(QLabel("start frame:"))
        self.rate_startpoint_spin = QDoubleSpinBox()
        self.rate_startpoint_spin.setRange(-999999, 999999)
        rate_layout.addWidget(self.rate_startpoint_spin)

        self.rate_info_label = QLabel("")
        self.rate_info_label.setStyleSheet("color: gray; font-size: 12px;")
        self.rate_param_frame = QFrame()
        rate_vbox = QVBoxLayout(self.rate_param_frame)
        rate_vbox.addLayout(rate_layout)
        rate_vbox.addWidget(self.rate_info_label)
        self.rate_param_frame.setVisible(False)

        # Rate 用のジェネレータパネル
        self.rate_gen_frame = self._build_section_gen('rate')

        # ===== マニューバ プレビュー (Time+Space or Rate+Space 揃った時点で確認) =====
        self.preview_group = QGroupBox("マニューバ プレビュー (Maneuver Preview)")
        prev_v = QVBoxLayout(self.preview_group)
        prev_hint = QLabel("Space + (Time または Rate) を設定後、軌道データを生成して 2D/3D で確認できます")
        prev_hint.setStyleSheet("color: gray; font-size: 11px;")
        prev_hint.setWordWrap(True)
        prev_v.addWidget(prev_hint)

        # データ生成方法の選択 (time to data / rate to data)
        pmode_layout = QHBoxLayout()
        pmode_layout.addWidget(QLabel("データ生成方法 / Generation method:"))
        self.preview_mode_select = QComboBox()
        self.preview_mode_select.addItems(["time to data", "rate to data"])
        self.preview_mode_select.currentIndexChanged.connect(self._update_preview_btn_state)
        pmode_layout.addWidget(self.preview_mode_select)
        pmode_layout.addStretch()
        prev_v.addLayout(pmode_layout)

        # Settings row: anim frame count + dpi for quick preview
        pset_layout = QHBoxLayout()
        pset_layout.addWidget(QLabel("3D frames:"))
        self.preview_frames_spin = QSpinBox()
        self.preview_frames_spin.setRange(5, 200)
        self.preview_frames_spin.setValue(20)
        pset_layout.addWidget(self.preview_frames_spin)
        pset_layout.addWidget(QLabel("dpi:"))
        self.preview_dpi_spin = QSpinBox()
        self.preview_dpi_spin.setRange(40, 300)
        self.preview_dpi_spin.setValue(80)
        pset_layout.addWidget(self.preview_dpi_spin)
        pset_layout.addStretch()
        prev_v.addLayout(pset_layout)

        self.preview_btn = QPushButton("プレビュー生成 (2D Plot + 3D GIF)")
        self.preview_btn.clicked.connect(self.start_maneuver_preview)
        prev_v.addWidget(self.preview_btn)

        self.preview_status_label = QLabel("Status: idle")
        self.preview_status_label.setStyleSheet("color: gray; font-size: 11px;")
        prev_v.addWidget(self.preview_status_label)

        prev_v.addWidget(QLabel("2D Plot:"))
        self.preview_2dplot_label = QLabel("(プレビュー生成後に表示)")
        self.preview_2dplot_label.setAlignment(Qt.AlignCenter)
        self.preview_2dplot_label.setMinimumSize(400, 250)
        # 2D プロットは透過 PNG (黒文字/黒線) なので背景を白にして視認性を確保
        self.preview_2dplot_label.setStyleSheet(
            "QLabel { background: #ffffff; color: #888; border: 1px solid #555; }"
        )
        prev_v.addWidget(self.preview_2dplot_label)

        prev_v.addWidget(QLabel("3D Animation (GIF):"))
        self.preview_3d_label = QLabel("(プレビュー生成後に表示)")
        self.preview_3d_label.setAlignment(Qt.AlignCenter)
        self.preview_3d_label.setMinimumSize(400, 300)
        self.preview_3d_label.setStyleSheet(
            "QLabel { background: #222; color: #888; border: 1px solid #555; }"
        )
        prev_v.addWidget(self.preview_3d_label)

        self.preview_group.setVisible(False)
        self._preview_movie = None  # QMovie の生存維持用

        # --- Mode selection ---
        self.mode_select = QComboBox()
        self.mode_select.addItems(["Select mode", "time to data", "rate to data"])
        self.mode_select.currentIndexChanged.connect(self.on_mode_selected)
        mode_label = QLabel("Select trajectory data generation method")

        # --- Animation toggle ---
        anim_label = QLabel("Animation Output Settings")
        self.anim_toggle = QCheckBox("Enable animation output")
        self.anim_toggle.stateChanged.connect(self.on_anim_toggle_changed)

        self.anim_settings_container = QFrame()
        anim_settings_layout = QVBoxLayout(self.anim_settings_container)
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Animation Duration (seconds):")
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 120)
        self.duration_spin.setValue(10)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_spin)
        anim_settings_layout.addLayout(duration_layout)
        self.anim_settings_container.setVisible(False)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Rendering")
        self.start_btn.clicked.connect(self.start_rendering)
        self.animonly_btn = QPushButton("Animation Only")
        self.animonly_btn.clicked.connect(self.start_animation_only)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.animonly_btn)

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)

        # ===== タブ構造でレイアウト組み立て =====
        tabs = QTabWidget()

        # --- Tab 1: 入力 (Setup) ---
        t1 = QWidget(); t1_l = QVBoxLayout(t1)
        for w in [self.video_btn, self.video_label,
                  self.slit_toggle, self.slit_label,
                  self.init_btn, self.info_label]:
            t1_l.addWidget(w)
        t1_l.addStretch()
        tabs.addTab(self._wrap_scroll(t1), "1. 入力 / Setup")

        # --- Tab 2: 画像生成 + 選択 (Images) ---
        # 各セクション (Space/Time/Rate) は独立した QGroupBox にまとめる:
        #   [Select / Auto Generate ボタン] + label + パラメータ + 専用ジェネレータ (パターン/波形/プレビュー)
        t2 = QWidget(); t2_l = QVBoxLayout(t2)
        t2_l.addWidget(self.gen_group)  # 共通サイズ設定

        space_box = QGroupBox("Space Image")
        space_v = QVBoxLayout(space_box)
        space_v.addLayout(self.space_btn_row)
        space_v.addWidget(self.space_label)
        space_v.addWidget(self.space_param_frame)
        space_v.addWidget(self.space_gen_frame)
        t2_l.addWidget(space_box)

        time_box = QGroupBox("Time Image")
        time_v = QVBoxLayout(time_box)
        time_v.addLayout(self.time_btn_row)
        time_v.addWidget(self.time_label)
        time_v.addWidget(self.time_param_frame)
        time_v.addWidget(self.time_gen_frame)
        t2_l.addWidget(time_box)

        rate_box = QGroupBox("Rate Image")
        rate_v = QVBoxLayout(rate_box)
        rate_v.addLayout(self.rate_btn_row)
        rate_v.addWidget(self.rate_label)
        rate_v.addWidget(self.rate_param_frame)
        rate_v.addWidget(self.rate_gen_frame)
        t2_l.addWidget(rate_box)

        t2_l.addStretch()
        tabs.addTab(self._wrap_scroll(t2), "2. 画像 / Images")

        # --- Tab 3: マニューバ プレビュー (Preview) ---
        t3 = QWidget(); t3_l = QVBoxLayout(t3)
        t3_l.addWidget(self.preview_group)
        t3_l.addStretch()
        tabs.addTab(self._wrap_scroll(t3), "3. プレビュー / Preview")

        # --- Tab 4: 出力 (Render) ---
        t4 = QWidget(); t4_l = QVBoxLayout(t4)
        for w in [mode_label, self.mode_select,
                  anim_label, self.anim_toggle,
                  self.anim_settings_container]:
            t4_l.addWidget(w)
        t4_l.addLayout(btn_layout)

        # レンダリング結果プレビュー (完了後に動画を上下に並べて再生)
        self.result_group = QGroupBox("レンダリング結果プレビュー (Rendered Preview)")
        result_v = QVBoxLayout(self.result_group)
        self.rendered_preview = VideoPreview("レンダリング動画 (Rendered Video)")
        self.anim_preview = VideoPreview("アニメーション (3D Animation)")
        result_v.addWidget(self.rendered_preview)
        result_v.addWidget(self.anim_preview)
        self.result_group.setVisible(False)
        t4_l.addWidget(self.result_group)

        t4_l.addStretch()
        tabs.addTab(self._wrap_scroll(t4), "4. 出力 / Render")

        # ===== ログは常時表示 (タブ外) =====
        log_label = QLabel("Log:")
        log_label.setStyleSheet("color: gray; font-size: 11px; margin-top: 4px;")
        self.log_window.setMinimumHeight(80)
        self.log_window.setMaximumHeight(160)

        # Splitter で「タブ」と「ログ」のサイズを可変に
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(tabs)
        log_box = QWidget()
        log_box_l = QVBoxLayout(log_box)
        log_box_l.setContentsMargins(0, 0, 0, 0)
        log_box_l.addWidget(log_label)
        log_box_l.addWidget(self.log_window)
        splitter.addWidget(log_box)
        splitter.setStretchFactor(0, 5)  # tabs 側を広く
        splitter.setStretchFactor(1, 1)

        outer = QVBoxLayout()
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(splitter)
        self.setLayout(outer)

    # --- UI Control ---
    def update_ui_state(self, stage):
        self.video_btn.setEnabled(True)
        self.anim_settings_container.setVisible(False)
        # 各セクションの「Generate & Apply」ボタン (各 _section_gens 内の generate_btn)
        gen_btns = [self._section_gens[t]['generate_btn']
                    for t in self._section_gens
                    if 'generate_btn' in self._section_gens.get(t, {})]
        if stage == "initial":
            for b in [self.init_btn, self.space_btn, self.time_btn, self.rate_btn,
                      self.mode_select, self.anim_toggle, self.start_btn,
                      self.animonly_btn, *gen_btns]:
                b.setEnabled(False)
        elif stage == "video_selected":
            self.init_btn.setEnabled(True)
        elif stage == "initialized":
            for b in [self.space_btn, self.time_btn, self.rate_btn,
                      self.mode_select, *gen_btns]:
                b.setEnabled(True)
            self.gen_group.setVisible(True)
            self.preview_group.setVisible(True)
            self.preview_btn.setEnabled(False)  # 画像が揃うまで無効
            self._apply_video_defaults()
        elif stage == "rendered":
            self.animonly_btn.setEnabled(True)
            self.anim_settings_container.setVisible(True)
            self.log("Animation-only rendering is now available.")

    def _apply_video_defaults(self):
        """drawManeuver 初期化直後に、スピンボックスの既定値を映像情報から賢く設定する"""
        if not self.dm:
            return
        # 共通サイズ
        self.gen_scan_size.setValue(int(self.dm.scan_nums))
        self.gen_time_size.setValue(120)
        # 各 type の既定パラメータ
        self.space_set_value.setValue(int(self.dm.scan_nums))
        self.time_vmin_spin.setValue(0)
        self.time_vmax_spin.setValue(int(self.dm.count))
        self.rate_baseline_spin.setValue(1.0)
        self.rate_maxdev_spin.setValue(0.5)
        # 各セクションジェネレータの波形周期既定値 = 時間方向サイズ (= 全体で 1 周期)
        for t in self._section_gens:
            self._section_gens[t]['wave_period'].setValue(120)
        # ヒントラベル更新 + マニューバプレビュー stale マーク用シグナル接続
        for sp in (self.space_set_value, self.time_vmin_spin, self.time_vmax_spin,
                   self.rate_maxdev_spin, self.rate_baseline_spin, self.rate_startpoint_spin):
            for cb in (self._update_gen_hint, self._mark_preview_stale):
                try:
                    sp.valueChanged.disconnect(cb)
                except Exception:
                    pass
                sp.valueChanged.connect(cb)
        # 3D プレビュー枠を出力映像のフォーマット (アスペクト比) に合わせる
        self._apply_3d_preview_aspect()
        self._update_gen_hint()
        self._update_all_section_previews()  # 全セクションの初回プレビュー

    def _apply_3d_preview_aspect(self):
        """3D プレビューラベルの枠を、出力映像 (dm.width×dm.height) のアスペクト比に合わせる。
        GIF はこの枠内にアスペクト比を保ったまま収める (歪ませない)。
        """
        if not self.dm:
            return
        try:
            vw, vh = int(self.dm.width), int(self.dm.height)
            base_w = 400
            box_h = max(120, int(round(base_w * vh / max(vw, 1))))
            self.preview_3d_label.setMinimumSize(base_w, box_h)
        except Exception:
            pass

    def _update_gen_hint(self):
        """共通サイズと、生成されるファイル形状/各セクションのファイル名を表示"""
        if not self.dm:
            self.gen_hint.setText("")
            return
        sd = int(getattr(self.dm, "scan_direction", 1))
        scan_size = self.gen_scan_size.value()
        time_size = self.gen_time_size.value()
        if sd == 1:
            file_dim = f"{scan_size}(W) × {time_size}(H)  → Width=scan, Height=time"
        else:
            file_dim = f"{time_size}(W) × {scan_size}(H)  → Width=time, Height=scan"
        self.gen_hint.setText(
            f"出力ファイル形状: {file_dim}\n"
            f"(各セクションでパターン/波形を個別に設定 → そのセクションの Auto Generate ボタンで生成)"
        )

    # --- Events ---
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video file", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.videopath = path
            self.video_label.setText(f"Selected: {path}")
            self.log(f"Video selected: {path}")
            self.update_ui_state("video_selected")

    def initialize_drawmaneuver(self):
        if not self.videopath:
            QMessageBox.warning(self, "Error", "Select a video first.")
            return
        sd = bool(self.slit_toggle.isChecked())
        self.log("Initializing drawManeuver...")
        try:
            self.dm = drawManeuver(videopath=self.videopath, sd=sd)
            info = (f"Video info: {self.dm.width}x{self.dm.height}, "
                    f"Frames: {self.dm.count}, FPS: {self.dm.recfps:.2f}")
            self.info_label.setText(info)
            self.log(info)
            self.update_ui_state("initialized")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.update_ui_state("video_selected")

    def _build_section_gen(self, type_name):
        """セクション ({type_name}=space/time/rate) 専用のジェネレータパネルを生成。

        含む widget:
          - pattern (QComboBox)
          - wave_frame (QFrame, 波形パターン選択時のみ表示)
              * wave_dir / wave_amp / wave_period / wave_phase
          - preview_label (240×140 サムネイル)

        widget は self._section_gens[type_name] に dict として保存。
        Returns: 構築済の QFrame (Tab2 のセクション内に addWidget する用)
        """
        g = {}
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        v = QVBoxLayout(frame)

        head = QLabel(f"▼ サンプル生成設定 ({type_name})")
        head.setStyleSheet("font-weight: bold; color: #555; margin-top: 4px;")
        v.addWidget(head)

        # Pattern (セクションごとに「通常再生」を先頭に並べ替えた選択肢)
        g['pattern_ids'], pattern_labels = section_pattern_order(type_name)
        pl = QHBoxLayout()
        pl.addWidget(QLabel("パターン:"))
        g['pattern'] = QComboBox()
        g['pattern'].addItems(pattern_labels)
        # 既定は先頭 = そのセクションの「通常再生」パターン
        g['pattern'].setCurrentIndex(0)
        pl.addWidget(g['pattern'])
        v.addLayout(pl)

        # Wave editor (collapsed unless wave pattern selected)
        g['wave_frame'] = QFrame()
        g['wave_frame'].setFrameShape(QFrame.StyledPanel)
        wv = QVBoxLayout(g['wave_frame'])

        wd = QHBoxLayout()
        wd.addWidget(QLabel("方向:"))
        g['wave_dir'] = QComboBox()
        g['wave_dir'].addItems(["上下方向 (vertical)", "左右方向 (horizontal)"])
        wd.addWidget(g['wave_dir'])
        wv.addLayout(wd)

        wa = QHBoxLayout()
        wa.addWidget(QLabel("振幅:"))
        g['wave_amp'] = QDoubleSpinBox()
        g['wave_amp'].setRange(0.0, 1.0)
        g['wave_amp'].setDecimals(3); g['wave_amp'].setSingleStep(0.05)
        g['wave_amp'].setValue(1.0)
        wa.addWidget(g['wave_amp'])
        wv.addLayout(wa)

        wp = QHBoxLayout()
        wp.addWidget(QLabel("周期:"))
        g['wave_period'] = QSpinBox()
        g['wave_period'].setRange(1, 32768); g['wave_period'].setValue(120)
        wp.addWidget(g['wave_period'])
        wp.addWidget(QLabel("px"))
        wv.addLayout(wp)

        wph = QHBoxLayout()
        wph.addWidget(QLabel("位相:"))
        g['wave_phase'] = QDoubleSpinBox()
        g['wave_phase'].setRange(-360.0, 720.0)
        g['wave_phase'].setDecimals(1); g['wave_phase'].setSingleStep(15.0)
        g['wave_phase'].setValue(0.0)
        wph.addWidget(g['wave_phase'])
        wph.addWidget(QLabel("°"))
        wv.addLayout(wph)

        g['wave_frame'].setVisible(False)
        v.addWidget(g['wave_frame'])

        # Preview (パターン編集中はパターン、画像読み込み後はその画像を表示)
        g['preview_label'] = QLabel("(Initialize 後に表示)")
        g['preview_label'].setAlignment(Qt.AlignCenter)
        g['preview_label'].setMinimumSize(320, 180)
        g['preview_label'].setStyleSheet(
            "QLabel { background: #222; color: #888; border: 1px solid #555; }"
        )
        v.addWidget(g['preview_label'])

        # Generate ボタン (Auto Generate を統合)
        g['generate_btn'] = QPushButton(f"▶ 生成して {type_name.capitalize()} に適用 / Generate & Apply")
        g['generate_btn'].setToolTip(
            "上のパターン設定で画像を生成し、このセクションの画像として自動セット"
        )
        g['generate_btn'].clicked.connect(lambda *_, t=type_name: self.generate_sample_image_action(t))
        g['generate_btn'].setEnabled(False)  # Initialize 前は無効
        v.addWidget(g['generate_btn'])

        # Wire updates: pattern change → wave_frame 可視性 + preview 更新
        def on_pattern(idx, t=type_name):
            ids = self._section_gens[t]['pattern_ids']
            pid = ids[idx] if 0 <= idx < len(ids) else ""
            self._section_gens[t]['wave_frame'].setVisible(pid == "wave")
            self._update_section_preview(t)
        g['pattern'].currentIndexChanged.connect(on_pattern)
        # 各パラメータ変更で preview 再描画
        for spinbox in (g['wave_amp'], g['wave_period'], g['wave_phase']):
            spinbox.valueChanged.connect(lambda *_, t=type_name: self._update_section_preview(t))
        g['wave_dir'].currentIndexChanged.connect(lambda *_, t=type_name: self._update_section_preview(t))

        self._section_gens[type_name] = g
        return frame

    def _make_preview_pixmap_for(self, type_name, max_w=240, max_h=160):
        """セクション {type_name} の現在設定でプレビュー QPixmap を生成。"""
        if not self.dm or type_name not in self._section_gens:
            return None
        g = self._section_gens[type_name]
        sd = int(getattr(self.dm, "scan_direction", 1))
        scan_size = self.gen_scan_size.value()
        time_size = self.gen_time_size.value()
        if sd == 1:
            h, w = time_size, scan_size
        else:
            h, w = scan_size, time_size

        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        ph = max(4, int(round(h * scale)))
        pw = max(4, int(round(w * scale)))
        sH = ph / max(h, 1)
        sW = pw / max(w, 1)

        pattern_id = g['pattern_ids'][g['pattern'].currentIndex()]
        wave_direction = "v" if g['wave_dir'].currentIndex() == 0 else "h"
        user_period = g['wave_period'].value()
        if wave_direction == "v":
            preview_period = max(1.0, user_period * sH)
        else:
            preview_period = max(1.0, user_period * sW)

        img16 = render_pattern(
            ph, pw, pattern_id,
            direction=wave_direction,
            amplitude=g['wave_amp'].value(),
            period=preview_period,
            phase_deg=g['wave_phase'].value(),
        )
        img8 = np.ascontiguousarray((img16 >> 8).astype(np.uint8))
        qimg = QImage(img8.tobytes(), pw, ph, pw, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def _update_section_preview(self, type_name):
        """セクション {type_name} のプレビューラベルを再描画"""
        if not self.dm or type_name not in self._section_gens:
            return
        pix = self._make_preview_pixmap_for(type_name)
        if pix is None:
            return
        self._section_gens[type_name]['preview_label'].setPixmap(pix)
        sd = int(getattr(self.dm, "scan_direction", 1))
        sc = self.gen_scan_size.value(); ts = self.gen_time_size.value()
        dim = f"{sc}(W) × {ts}(H)" if sd == 1 else f"{ts}(W) × {sc}(H)"
        self._section_gens[type_name]['preview_label'].setToolTip(f"ファイル形状: {dim}")

    def _update_all_section_previews(self, *_):
        """共通サイズ変更時に全セクションのプレビューを更新"""
        for t in self._section_gens:
            self._update_section_preview(t)

    def _show_loaded_image_in_preview(self, type_name, path):
        """読み込んだ (or 生成した) 画像をセクションのプレビューエリアに表示。
        パターンプレビューと同じ QLabel を使うことで「兼任」を実現。
        """
        if type_name not in self._section_gens:
            return
        label = self._section_gens[type_name]['preview_label']
        if not (path and os.path.exists(path)):
            label.setText("(画像なし)")
            return
        pix = QPixmap()
        ok = pix.load(path)
        if not ok or pix.isNull():
            label.setText(f"(画像 load 失敗: {os.path.basename(path)})")
            return
        # ラベルサイズに合わせて縮小 (アスペクト比保持)
        target_w = max(label.width(), label.minimumWidth())
        target_h = max(label.height(), label.minimumHeight())
        scaled = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        label.setToolTip(
            f"ロード画像: {os.path.basename(path)}\n"
            f"({pix.width()}×{pix.height()})"
        )

    # --- Maneuver preview (2D + 3D) ---
    def _selected_preview_mode(self):
        """プレビューパネルで選択された生成方法を "time" / "rate" で返す。"""
        sel = self.preview_mode_select.currentText() if hasattr(self, "preview_mode_select") else "time to data"
        return "rate" if sel == "rate to data" else "time"

    def _can_preview_mode(self):
        """選択された生成方法でプレビュー可能なら "time"/"rate"、不足があれば None。"""
        if not self.dm or not self.space_img_path:
            return None
        mode = self._selected_preview_mode()
        if mode == "time":
            return "time" if self.time_img_path else None
        else:
            return "rate" if self.rate_img_path else None

    def _update_preview_btn_state(self):
        """選択された生成方法と画像セット状態に応じてボタンの有効/無効を切り替え"""
        if not hasattr(self, "preview_btn"):
            return
        mode = self._can_preview_mode()
        self.preview_btn.setEnabled(mode is not None)
        if not self.dm or not self.space_img_path:
            self.preview_status_label.setText("Status: Space 画像が必要です")
        elif mode is not None:
            self.preview_status_label.setText(f"Status: ready ({mode} mode)")
        else:
            need = "Time" if self._selected_preview_mode() == "time" else "Rate"
            self.preview_status_label.setText(f"Status: {need} 画像が必要です")

    def start_maneuver_preview(self):
        mode = self._can_preview_mode()
        if mode is None:
            QMessageBox.warning(self, "Error",
                                "Space + (Time または Rate) 画像が必要です")
            return
        self.preview_btn.setEnabled(False)
        self.preview_status_label.setText("Status: running…")
        self.log(f"[preview] starting in {mode} mode")

        self._preview_worker = ManeuverPreviewWorker(
            self.dm, mode,
            self.space_img_path, self.time_img_path, self.rate_img_path,
            self.space_set_value.value(),
            self.time_vmin_spin.value(), self.time_vmax_spin.value(),
            self.rate_maxdev_spin.value(),
            self.rate_baseline_spin.value(),
            self.rate_startpoint_spin.value(),
            anim_frames=self.preview_frames_spin.value(),
            anim_fps=10,
            anim_dpi=self.preview_dpi_spin.value(),
        )
        self._preview_worker.progress_signal.connect(self._on_preview_progress)
        self._preview_worker.done_signal.connect(self._on_preview_done)
        self._preview_worker.start()

    def _on_preview_progress(self, msg):
        self.preview_status_label.setText(f"Status: {msg}")
        self.log(f"[preview] {msg}")

    def _on_preview_done(self, success, plot2d, gif):
        self.preview_btn.setEnabled(True)
        if not success:
            self.preview_status_label.setText("Status: failed (see log)")
            return

        # 前回の QMovie を停止 (上書きされたファイルに対する古いキャッシュを切る)
        if self._preview_movie is not None:
            try:
                self._preview_movie.stop()
            except Exception:
                pass
            self.preview_3d_label.setMovie(None)
            self._preview_movie = None

        # 2D PNG 表示 — QPixmap は path から毎回読むのでキャッシュなし
        if plot2d and os.path.exists(plot2d):
            pix = QPixmap()
            pix.load(plot2d)
            if not pix.isNull():
                scaled = pix.scaled(self.preview_2dplot_label.width(),
                                    self.preview_2dplot_label.height(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_2dplot_label.setPixmap(scaled)
                self.log(f"[preview] 2D: {os.path.basename(plot2d)}")
            else:
                self.preview_2dplot_label.setText(f"(2D plot load 失敗: {plot2d})")
        else:
            self.preview_2dplot_label.setText("(2D plot 出力なし)")

        # 3D GIF 表示 (QMovie で再生) — キャッシュ無効化して同名ファイル上書きにも対応
        if gif and os.path.exists(gif):
            movie = QMovie(gif)
            movie.setCacheMode(QMovie.CacheNone)
            if movie.isValid():
                # GIF のネイティブ寸法を取得し、ラベル枠内にアスペクト比を保って収める
                # (以前は label.size() へ強制スケールしていたため縦横比が崩れていた)
                native = QImageReader(gif).size()
                box = self.preview_3d_label.size()
                if native.width() > 0 and native.height() > 0:
                    scale = min(box.width() / native.width(),
                                box.height() / native.height())
                    movie.setScaledSize(QSize(
                        max(1, int(native.width() * scale)),
                        max(1, int(native.height() * scale)),
                    ))
                self.preview_3d_label.setMovie(movie)
                movie.start()
                self._preview_movie = movie  # 参照保持
                self.log(f"[preview] 3D GIF: {os.path.basename(gif)}")
            else:
                self.preview_3d_label.setText(f"(GIF 読み込み失敗: {gif})")
        else:
            self.preview_3d_label.setText("(3D GIF 出力なし — ffmpeg をチェック)")

        self.preview_status_label.setText("Status: done — 設定を変更したら「プレビュー生成」を再実行")
        self._preview_stale = False

    def _mark_preview_stale(self, *_):
        """Tab 2 の編集を検知してプレビュー側に「再生成が必要」と表示する。
        実プレビュー画像は残したまま、ステータスだけ更新 (古い表示の使い回し防止)。
        """
        if not hasattr(self, "preview_btn"):
            return
        if not self.preview_btn.isEnabled():
            return
        # 既に "running" 中などはスキップ
        cur = self.preview_status_label.text()
        if "running" in cur.lower():
            return
        self.preview_status_label.setText(
            "Status: ⚠ 設定が変更されました — 「プレビュー生成」を再実行してください"
        )
        self._preview_stale = True

    def generate_sample_image_action(self, type_name):
        """セクション {type_name} のジェネレータ設定でサンプル画像を生成 → 自動セット"""
        if not self.dm:
            QMessageBox.warning(self, "Error", "Initialize a video first.")
            return
        if type_name not in ("space", "time", "rate"):
            QMessageBox.warning(self, "Error", f"Unknown image type: {type_name}")
            return
        if type_name not in self._section_gens:
            QMessageBox.warning(self, "Error", f"Generator panel not built for {type_name}")
            return

        g = self._section_gens[type_name]
        pattern_id = g['pattern_ids'][g['pattern'].currentIndex()]
        scan_size = self.gen_scan_size.value()
        time_size = self.gen_time_size.value()
        sd = int(getattr(self.dm, "scan_direction", 1))
        out_dir = os.path.dirname(self.videopath) or "."

        # 波形パラメータ (このセクション固有)
        wave_direction = "v" if g['wave_dir'].currentIndex() == 0 else "h"
        wave_amp = g['wave_amp'].value()
        wave_period = g['wave_period'].value()
        wave_phase = g['wave_phase'].value()

        try:
            out_path = generate_sample_image(
                out_dir=out_dir,
                image_type=type_name,
                pattern_id=pattern_id,
                scan_size=scan_size,
                time_size=time_size,
                scan_direction=sd,
                space_range=self.space_set_value.value(),
                time_vmin=self.time_vmin_spin.value(),
                time_vmax=self.time_vmax_spin.value(),
                rate_maxdev=self.rate_maxdev_spin.value(),
                wave_direction=wave_direction,
                wave_amplitude=wave_amp,
                wave_period=wave_period,
                wave_phase_deg=wave_phase,
            )
        except Exception as e:
            QMessageBox.critical(self, "Generate Error", str(e))
            self.log(f"[ERROR] generate_sample_image: {e}")
            return

        if pattern_id == "wave":
            self.log(f"Sample {type_name} (wave dir={wave_direction} amp={wave_amp} "
                     f"period={wave_period}px phase={wave_phase}°): {out_path}")
        else:
            self.log(f"Sample {type_name} ({pattern_id}): {out_path}")
        setattr(self, f"{type_name}_img_path", out_path)
        self._wire_loaded_image(type_name, out_path)

    def _wire_loaded_image(self, img_type, path):
        """select_image() の "画像情報表示 + パラメータ抽出 + プレビュー" 共通処理"""
        getattr(self, f"{img_type}_label").setText(f"Selected: {path}")
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                is_grayscale = (mode == "L") or ("I;16" in mode)
                is_16bit = ("I;16" in mode)
                gscale = "Grayscale" if is_grayscale else "Color"
                bit = "16-bit" if is_16bit else "8-bit"
                info_text = f"Size: {width}x{height} | {gscale}, {bit}"
                getattr(self, f"{img_type}_info_label").setText(info_text)
                self.log(f"{img_type} info: {info_text}")
        except Exception as e:
            getattr(self, f"{img_type}_info_label").setText(f"[Error reading image info: {e}]")

        getattr(self, f"{img_type}_param_frame").setVisible(True)

        # ロードした画像をセクション内のプレビューエリアに表示 (パターン preview と兼任)
        self._show_loaded_image_in_preview(img_type, path)

        if self.dm:
            try:
                params = self.dm.extract_params_from_filename(Path(path))
                t = params.get("type")
                if t == "space":
                    self.space_set_value.setValue(params.get("range", self.dm.scan_nums))
                elif t == "time":
                    self.time_vmin_spin.setValue(params.get("vmin", 0))
                    self.time_vmax_spin.setValue(params.get("vmax", 0))
                elif t == "rate":
                    self.rate_maxdev_spin.setValue(params.get("max_dev", 0.0))
                self.log(f"Extracted params: {params}")
            except Exception as e:
                self.log(f"[WARN] Could not extract params: {e}")

        self.mode_select.blockSignals(True)
        self.mode_select.setCurrentText("Select mode")
        self.mode_select.blockSignals(False)
        self.start_btn.setEnabled(False)
        self.anim_toggle.setEnabled(False)
        self.anim_settings_container.setVisible(False)

        # マニューバプレビューボタンの有効/無効を更新 + stale マーク (画像が変わったため)
        self._update_preview_btn_state()
        self._mark_preview_stale()

    def select_image(self, img_type):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {img_type} image", "", "Images (*.png *.jpg *.bmp)")
        if not path:
            return
        setattr(self, f"{img_type}_img_path", path)
        self.log(f"{img_type} image selected: {path}")
        self._wire_loaded_image(img_type, path)

    def on_mode_selected(self, index):
        mode = self.mode_select.currentText()
        if mode in ["time to data", "rate to data"]:
            self.start_btn.setEnabled(True)
            self.anim_toggle.setEnabled(True)
            self.log(f"Mode selected: {mode}")
        else:
            self.start_btn.setEnabled(False)
            self.anim_toggle.setEnabled(False)
            self.anim_settings_container.setVisible(False)

    def on_anim_toggle_changed(self, state):
        if state == Qt.Checked:
            self.anim_settings_container.setVisible(True)
        else:
            self.anim_settings_container.setVisible(False)
        self.start_btn.setEnabled(True)

    def update_slit_label(self):
        if self.slit_toggle.isChecked():
            self.slit_label.setText("Slit direction: vertical")
        else:
            self.slit_label.setText("Slit direction: horizontal")

    def start_rendering(self):
        mode = self.mode_select.currentText()
        if mode not in ["time to data", "rate to data"]:
            QMessageBox.warning(self, "Error", "Please select a valid mode.")
            return
        animout = self.anim_toggle.isChecked()
        duration = self.duration_spin.value()

        space_set = self.space_set_value.value()
        vmin = self.time_vmin_spin.value()
        vmax = self.time_vmax_spin.value()
        maxdev = self.rate_maxdev_spin.value()
        baseline = self.rate_baseline_spin.value()
        startpoint = self.rate_startpoint_spin.value()

        self.worker = RenderWorker(
            self.dm, mode, animout,
            self.space_img_path, self.time_img_path, self.rate_img_path,
            duration,
            space_set=space_set, time_vmin=vmin, time_vmax=vmax,
            rate_maxdev=maxdev, rate_baseline=baseline, rate_startpoint=startpoint
        )
        self.worker.log_signal.connect(self.log)
        self.worker.done_signal.connect(self.on_render_done)
        self.worker.start()

    def start_animation_only(self):
        if not self.dm:
            QMessageBox.warning(self, "Error", "No drawManeuver instance found.")
            return
        if not hasattr(self.dm, "data") or self.dm.data is None:
            QMessageBox.warning(self, "Error", "No data to animate. Please run rendering first.")
            return

        duration = self.duration_spin.value()
        self.log("Starting animation-only rendering...")

        self.worker = RenderWorker(
            self.dm, None, True,
            None, None, None,
            duration,
            anim_only=True
        )
        self.worker.log_signal.connect(self.log)
        self.worker.done_signal.connect(self.on_render_done)
        self.worker.start()

    def on_render_done(self, success, video_path="", anim_path=""):
        if success:
            self.render_completed = True
            self.log(" Rendering completed.")
            self.update_ui_state("rendered")
            self._show_rendered_preview(video_path, anim_path)
            if hasattr(self.dm, "data") and self.dm.data is not None:
                self.animonly_btn.setEnabled(True)
                self.log("Animation-only rendering is now available.")
            else:
                self.log("[WARN] No data found in drawManeuver; Animation Only disabled.")
                self.animonly_btn.setEnabled(False)
        else:
            self.log("Rendering failed.")

    def _show_rendered_preview(self, video_path, anim_path):
        """レンダリング結果 (本編動画 / アニメーション) をプレビュー領域に読み込み再生。
        空パスの枠は既存表示を維持する (例: Animation Only は本編動画を残す)。
        """
        if video_path and os.path.exists(video_path):
            self.rendered_preview.stop()
            self.rendered_preview.load(video_path)
            self.log(f"[preview] rendered video: {os.path.basename(video_path)}")
        if anim_path and os.path.exists(anim_path):
            self.anim_preview.stop()
            self.anim_preview.load(anim_path)
            self.log(f"[preview] animation: {os.path.basename(anim_path)}")
        any_shown = self.rendered_preview.loaded or self.anim_preview.loaded
        self.result_group.setVisible(any_shown)
        if not any_shown:
            self.log("[preview] no output video found to preview.")

    def log(self, text):
        self.log_window.append(str(text))
        self.log_window.ensureCursorVisible()


# ======== Main entry ========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = IMGTransApp()
    win.show()
    sys.exit(app.exec_())
