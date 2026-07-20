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
    QFrame, QDoubleSpinBox, QGroupBox, QTabWidget, QScrollArea, QSplitter,
    QProgressBar, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl, QTimer
from PyQt5.QtGui import (QImage, QPixmap, QMovie, QImageReader,
                         QPainter, QPen, QColor)

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

# リアルタイム GPU プレビュー (任意依存: wgpu)。読み込めなくてもアプリは動く。
try:
    from realtime_preview import RealtimePreviewWidget
    _HAS_RT_PREVIEW = True
except Exception:
    RealtimePreviewWidget = None
    _HAS_RT_PREVIEW = False


# ======== i18n (Japanese / English UI) ========
# 起動時のデフォルト言語は環境変数 STF_LANG で切替可能 (ja / en)。既定は ja。
# GUI 上の「Language / 言語」セレクタでも実行中に切り替えられる。
LANG = os.environ.get("STF_LANG", "ja").strip().lower()
if LANG not in ("ja", "en"):
    LANG = "ja"

# key -> {"ja": ..., "en": ...}
TR = {
    # Window / tabs
    "window_title": {"ja": "Shape of Time Flow", "en": "Shape of Time Flow"},
    "tab_main":    {"ja": "1. 入力・画像 / Setup & Images", "en": "1. Setup & Images"},
    "tab_preview": {"ja": "2. プレビュー / Preview", "en": "2. Preview"},
    "tab_render":  {"ja": "3. 出力 / Render",     "en": "3. Render"},
    "grp_setup":   {"ja": "入力 (Setup)", "en": "Setup"},
    # Language selector
    "lang_label":  {"ja": "言語 / Language:",     "en": "Language / 言語:"},
    # Setup tab
    "btn_select_video": {"ja": "動画を選択 / Select Video File", "en": "Select Video File"},
    "no_video":    {"ja": "動画が未選択です",       "en": "No video file selected"},
    "chk_vertical":{"ja": "縦スリット (Vertical)", "en": "Vertical (check for vertical)"},
    "slit_h":      {"ja": "スリット方向: 横 (horizontal)", "en": "Slit direction: horizontal"},
    "slit_v":      {"ja": "スリット方向: 縦 (vertical)",   "en": "Slit direction: vertical"},
    "btn_initialize": {"ja": "初期化 / Initialize", "en": "Initialize"},
    "video_not_init": {"ja": "動画情報: (未初期化)", "en": "Video info: (not initialized)"},
    # Shared size
    "grp_shared_size": {"ja": "共通サイズ設定 (Shared Image Size)", "en": "Shared Image Size"},
    "lbl_scan_size":   {"ja": "スキャン方向サイズ:", "en": "Scan-direction size:"},
    "hint_scan_auto":  {"ja": "(映像幅から自動)",   "en": "(auto from video width)"},
    "lbl_time_size":   {"ja": "時間方向サイズ:",     "en": "Time-direction size:"},
    "hint_time_any":   {"ja": "(任意のフレーム数)",  "en": "(any frame count)"},
    "lbl_out_fps":     {"ja": "出力FPS:",            "en": "Output FPS:"},
    "hint_out_fps":    {"ja": "(最終映像の尺 = 時間方向サイズ ÷ 出力FPS)",
                        "en": "(final duration = time size ÷ output fps)"},
    "gen_hint_dur":    {"ja": "→ 出力映像の尺: {dur} 秒  ({ts} frames ÷ {fps} fps)",
                        "en": "→ Output duration: {dur} s  ({ts} frames ÷ {fps} fps)"},
    "gen_hint": {
        "ja": "出力ファイル形状: {dim}\n(各セクションでパターン/波形を個別に設定 → そのセクションの Generate ボタンで生成)",
        "en": "Output file shape: {dim}\n(Set pattern/wave per section → generate with that section's Generate button)",
    },
    # Image sections
    "grp_space_image": {"ja": "Space 画像 (Space Image)", "en": "Space Image"},
    "grp_time_image":  {"ja": "Time 画像 (Time Image)",   "en": "Time Image"},
    "grp_rate_image":  {"ja": "Rate 画像 (Rate Image)",   "en": "Rate Image"},
    "no_space_image":  {"ja": "Space 画像が未選択です", "en": "No space image selected"},
    "no_time_image":   {"ja": "Time 画像が未選択です",  "en": "No time image selected"},
    "no_rate_image":   {"ja": "Rate 画像が未選択です",  "en": "No rate image selected"},
    "lbl_space_range": {"ja": "space range:", "en": "space range:"},
    "lbl_vmin": {"ja": "vmin:", "en": "vmin:"},
    "lbl_vmax": {"ja": "vmax:", "en": "vmax:"},
    "lbl_baseline": {"ja": "baseline:", "en": "baseline:"},
    "lbl_max_range": {"ja": "max_range:", "en": "max_range:"},
    "lbl_start_frame": {"ja": "start frame:", "en": "start frame:"},
    # Section generator
    "gen_header": {"ja": "画像選択 / 生成設定 ({t})", "en": "Image / Generator settings ({t})"},
    "lbl_pattern": {"ja": "パターン:", "en": "Pattern:"},
    "lbl_wave_dir": {"ja": "方向:", "en": "Direction:"},
    "lbl_wave_amp": {"ja": "振幅:", "en": "Amplitude:"},
    "lbl_wave_period": {"ja": "周期:", "en": "Period:"},
    "lbl_wave_phase": {"ja": "位相:", "en": "Phase:"},
    "lbl_wave_angle": {"ja": "角度:", "en": "Angle:"},
    "hint_wave_angle": {"ja": "(0°=上下, 90°=左右, 30°/45° など任意)",
                         "en": "(0°=vertical, 90°=horizontal, any angle)"},
    # Layer compositing
    "lbl_layer": {"ja": "レイヤー {n}", "en": "Layer {n}"},
    "btn_add_layer": {"ja": "＋ レイヤーを追加 (合成)", "en": "+ Add layer (composite)"},
    "lbl_blend": {"ja": "合成:", "en": "Blend:"},
    "lbl_opacity": {"ja": "不透明度:", "en": "Opacity:"},
    "lbl_dot": {"ja": "ドットサイズ:", "en": "Dot size:"},
    "lbl_blur": {"ja": "ブラー:", "en": "Blur:"},
    "lbl_seed": {"ja": "シード:", "en": "Seed:"},
    "lbl_cell": {"ja": "スケール:", "en": "Scale:"},
    "lbl_octaves": {"ja": "オクターブ:", "en": "Octaves:"},
    "btn_layer_image": {"ja": "画像を選択…", "en": "Select image…"},
    "no_layer_image": {"ja": "(画像未選択 → 50%グレー扱い)",
                        "en": "(no image → treated as 50% gray)"},
    "wave_dir_v": {"ja": "上下方向 (vertical)", "en": "Vertical"},
    "wave_dir_h": {"ja": "左右方向 (horizontal)", "en": "Horizontal"},
    "preview_after_init": {"ja": "(Initialize 後に表示)", "en": "(shown after Initialize)"},
    "btn_generate_apply": {"ja": "▶ 生成して {t} に適用 / Generate & Apply",
                            "en": "▶ Generate & Apply to {t}"},
    # Apply mode (Tab2 bottom — required before Preview/Render unlock)
    "grp_apply_mode": {"ja": "適用方法の選択 (Apply Mode) ※必須",
                        "en": "Apply Mode (required)"},
    "apply_mode_hint": {
        "ja": "画像データをどう適用するかを選択してください:\n"
              "  time to data = Time 画像を「時間マップ」として適用\n"
              "  rate to data = Rate 画像を「再生レートマップ」として適用\n"
              "選択して必要な画像が揃うと「2. プレビュー」「3. 出力」タブが使えるようになります。",
        "en": "Choose how the image data is applied:\n"
              "  time to data = apply the Time image as a time map\n"
              "  rate to data = apply the Rate image as a playback-rate map\n"
              "Selecting this (with the required images set) unlocks the Preview / Render tabs.",
    },
    "grp_live3d": {"ja": "軌道プロット ライブプレビュー (3D / 2D 自動更新)",
                    "en": "Trajectory Plots Live Preview (3D / 2D, auto)"},
    "live3d_waiting": {"ja": "(画像と適用方法が揃うと自動生成されます)",
                        "en": "(auto-generates once images & apply mode are set)"},
    "live3d_updating": {"ja": "更新中…", "en": "updating…"},
    "lbl_apply_mode_info": {"ja": "適用方法: {m}   (変更は「1. 入力・画像」タブで)",
                             "en": "Apply mode: {m}   (change on the Setup & Images tab)"},
    "status_need_mode": {"ja": "Status: 適用方法が未選択です (「1. 入力・画像」タブで選択)",
                          "en": "Status: choose an apply mode (Setup & Images tab)"},
    "processing_wait": {"ja": "⏳ 演算中です — しばらくお待ちください…",
                         "en": "⏳ Processing — please wait…"},
    # Maneuver preview panel
    "grp_maneuver_preview": {"ja": "マニューバ プレビュー (Maneuver Preview)",
                              "en": "Maneuver Preview"},
    "grp_realtime": {"ja": "リアルタイム軸間変換プレビュー (GPU)",
                      "en": "Realtime axis-transform preview (GPU)"},
    "preview_hint": {"ja": "Space + (Time または Rate) を設定後、軌道データを生成して 2D/3D で確認できます",
                      "en": "After setting Space + (Time or Rate), generate trajectory data to check it in 2D/3D"},
    "lbl_gen_method": {"ja": "データ生成方法 / Generation method:",
                        "en": "Generation method:"},
    "lbl_3d_frames": {"ja": "3D frames:", "en": "3D frames:"},
    "lbl_dpi": {"ja": "dpi:", "en": "dpi:"},
    "btn_gen_preview": {"ja": "プレビュー生成 (2D Plot + 3D GIF)",
                         "en": "Generate Preview (2D Plot + 3D GIF)"},
    "lbl_2d_plot": {"ja": "2D Plot:", "en": "2D Plot:"},
    "lbl_3d_anim": {"ja": "3D Animation (GIF):", "en": "3D Animation (GIF):"},
    "preview_after_gen": {"ja": "(プレビュー生成後に表示)", "en": "(shown after generating preview)"},
    "status_idle": {"ja": "Status: idle", "en": "Status: idle"},
    "status_need_space": {"ja": "Status: Space 画像が必要です", "en": "Status: a Space image is required"},
    "status_ready": {"ja": "Status: ready ({m} mode)", "en": "Status: ready ({m} mode)"},
    "status_need_img": {"ja": "Status: {need} 画像が必要です", "en": "Status: a {need} image is required"},
    # Render tab
    "lbl_select_method": {"ja": "軌道データ生成方法を選択 / Select trajectory data generation method",
                           "en": "Select trajectory data generation method"},
    "lbl_anim_settings": {"ja": "アニメーション出力設定 / Animation Output Settings",
                           "en": "Animation Output Settings"},
    "chk_enable_anim": {"ja": "アニメーション出力を有効化 / Enable animation output",
                         "en": "Enable animation output"},
    "lbl_anim_duration": {"ja": "アニメーション長さ (秒) / Animation Duration (seconds):",
                           "en": "Animation Duration (seconds):"},
    "btn_start_render": {"ja": "レンダリング開始 / Start Rendering", "en": "Start Rendering"},
    "btn_anim_only": {"ja": "アニメーションのみ / Animation Only", "en": "Animation Only"},
    "grp_rendered_preview": {"ja": "レンダリング結果プレビュー (Rendered Preview)",
                              "en": "Rendered Preview"},
    "rendered_video_title": {"ja": "レンダリング動画 (Rendered Video)", "en": "Rendered Video"},
    "anim_title": {"ja": "アニメーション (3D Animation)", "en": "3D Animation"},
    # Log
    "lbl_log": {"ja": "Log:", "en": "Log:"},
    "mode_select_placeholder": {"ja": "Select mode", "en": "Select mode"},
    # VideoPreview
    "btn_pause": {"ja": "⏸ 一時停止", "en": "⏸ Pause"},
    "btn_play": {"ja": "▶ 再生", "en": "▶ Play"},
    "btn_open_external": {"ja": "外部プレイヤーで開く", "en": "Open in external player"},
    "no_multimedia": {"ja": "(QtMultimedia が無いため内蔵再生できません)",
                       "en": "(QtMultimedia not available — embedded playback disabled)"},
}


def tr(key, **fmt):
    """現在の言語 LANG に応じた訳文を返す。未知キーはキー名をそのまま返す。"""
    d = TR.get(key)
    s = (d.get(LANG) or d.get("ja")) if d else key
    return s.format(**fmt) if fmt else s


# ======== Sample image generator ========
# パターン表示ラベル (言語別)。ロジックは PATTERN_IDS を使うので翻訳しても安全。
# 並び順は全セクション共通の固定順 (黒→白 → 白→黒 → 左右 → グレー → ノイズ → 波形)。
PATTERN_LABELS_BY_LANG = {
    "ja": [
        "上→下: 黒→白",
        "上→下: 白→黒",
        "左→右: 黒→白",
        "左→右: 白→黒",
        "50% グレー均一",
        "ランダムノイズ",
        "波形 (Wave) ※ 振幅/周期/位相/角度 編集",
    ],
    "en": [
        "Top→Bottom: black→white",
        "Top→Bottom: white→black",
        "Left→Right: black→white",
        "Left→Right: white→black",
        "Solid 50% gray",
        "Random noise",
        "Wave ※ edit amplitude/period/phase/angle",
    ],
}
PATTERN_IDS = [
    "v_b2w", "v_w2b", "h_b2w", "h_w2b", "solid_gray", "random", "wave",
]

# 「通常再生」サフィックス (言語別)
NORMAL_SUFFIX = {"ja": "（通常再生）", "en": " (normal playback)"}

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


def normal_pattern_for(section, sd=1):
    """スリット方向 sd に応じた「通常再生」パターン id を返す。

    マップファイルの形状は sd=1: (time, scan) / sd=0: (scan, time)。
    space は常に「スキャン軸に沿ったランプ」、time は常に「時間軸に沿った
    ランプ」が通常再生なので、ファイル上の向きは sd で入れ替わる:
        sd=1: space=h_b2w (横=scan), time=v_b2w (縦=time)
        sd=0: space=v_b2w (縦=scan), time=h_b2w (横=time)
    """
    if section == "rate":
        return "solid_gray"
    if int(sd) == 1:
        return "h_b2w" if section == "space" else "v_b2w"
    return "v_b2w" if section == "space" else "h_b2w"


def section_pattern_order(type_name, lang=None, sd=1):
    """セクション {type_name} 用の (pattern_ids, labels) を現在の言語で返す。

    並び順は全セクション共通の PATTERN_IDS 固定順 (並べ替えなし)。
    「通常再生」に相当する pattern (スリット方向 sd に依存) のラベル末尾に
    だけサフィックスを付与する。
    """
    lang = lang or LANG
    pattern_labels = PATTERN_LABELS_BY_LANG.get(lang, PATTERN_LABELS_BY_LANG["ja"])
    normal = normal_pattern_for(type_name, sd)
    labels = []
    for pid, base in zip(PATTERN_IDS, pattern_labels):
        if pid == normal:
            base = f"{base}{NORMAL_SUFFIX.get(lang, NORMAL_SUFFIX['ja'])}"
        labels.append(base)
    return list(PATTERN_IDS), labels


def render_pattern(h_pix, w_pix, pattern_id, **wave_params):
    """16bit uint16 (H, W) のグレースケール画像を生成する。

    pattern_id="wave" の場合は wave_params で:
        amplitude  : 0.0 - 1.0 (full-range の割合, 1.0 で 0..65535 振り切る)
        period     : 1サイクルのピクセル数 (例: H==period でちょうど1周期)
        phase_deg  : 開始位相 (度, 0..360)
        angle_deg  : 波の進行方向の角度 (度)。0°=上下方向, 90°=左右方向,
                     30°/45° など任意の斜め波が作れる。
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
        amp = float(wave_params.get("amplitude", 1.0))      # 0..1
        period = max(1.0, float(wave_params.get("period", max(h, 1))))
        phase = np.deg2rad(float(wave_params.get("phase_deg", 0.0)))
        angle = float(wave_params.get("angle_deg", 0.0)) % 360.0
        th = np.deg2rad(angle)
        # 0..1 正規化された sin 波 → 16bit
        mid = 32767.5
        amp_scaled = amp * 32767.5
        # 波の座標 u = x·sinθ + y·cosθ (0°=上下方向, 90°=左右方向)。
        # 軸に沿う角度は 1D ブロードキャストで済ませ、斜めのときだけ 2D 計算。
        if angle % 180.0 == 0.0:
            sign = 1.0 if angle < 180.0 else -1.0
            axis = sign * np.arange(h, dtype=np.float64)
            wave1d = mid + amp_scaled * np.sin(2 * np.pi * axis / period + phase)
            col = np.clip(wave1d, 0, 65535)
            img = np.broadcast_to(col[:, None], (h, w)).astype(np.uint16)
        elif angle % 180.0 == 90.0:
            sign = 1.0 if angle < 180.0 else -1.0
            axis = sign * np.arange(w, dtype=np.float64)
            wave1d = mid + amp_scaled * np.sin(2 * np.pi * axis / period + phase)
            row = np.clip(wave1d, 0, 65535)
            img = np.broadcast_to(row[None, :], (h, w)).astype(np.uint16)
        else:
            xx = np.arange(w, dtype=np.float32)[None, :]
            yy = np.arange(h, dtype=np.float32)[:, None]
            u = xx * np.sin(th) + yy * np.cos(th)
            wave2d = mid + amp_scaled * np.sin(2 * np.pi * u / period + phase)
            img = np.clip(wave2d, 0, 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unknown pattern_id: {pattern_id}")
    return img


# ======== Layer compositing (パターンを何層でも重ねられる) ========

# レイヤーで追加選択できるパターン (基本パターンに加えて)
EXTRA_PATTERN_IDS = ["perlin", "image"]
EXTRA_PATTERN_LABELS = {
    "ja": ["パーリンノイズ", "画像ファイル…"],
    "en": ["Perlin noise", "Image file…"],
}

BLEND_IDS = ["normal", "add", "subtract", "multiply", "screen", "difference"]
BLEND_LABELS = {
    "ja": ["通常", "加算", "減算", "乗算", "スクリーン", "差の絶対値"],
    "en": ["Normal", "Add", "Subtract", "Multiply", "Screen", "Difference"],
}


def layer_pattern_order(type_name, lang=None, sd=1):
    """レイヤー用: 基本パターン + perlin + 画像ファイル の (ids, labels)。"""
    ids, labels = section_pattern_order(type_name, lang, sd=sd)
    lang = lang or LANG
    extra = EXTRA_PATTERN_LABELS.get(lang, EXTRA_PATTERN_LABELS["ja"])
    return ids + list(EXTRA_PATTERN_IDS), labels + list(extra)


def perlin2d(h, w, cell, octaves=1, seed=0):
    """勾配 (Perlin) ノイズの fBm。float32 (h, w) を 0..1 で返す。"""
    h, w = int(h), int(w)
    total = np.zeros((h, w), np.float32)
    amp, amp_sum = 1.0, 0.0
    for o in range(max(1, int(octaves))):
        c = max(2.0, float(cell) / (2 ** o))
        gy = int(np.ceil(h / c)) + 2
        gx = int(np.ceil(w / c)) + 2
        rng = np.random.default_rng(int(seed) + o * 1013)
        ang = rng.uniform(0, 2 * np.pi, (gy, gx)).astype(np.float32)
        grad = np.stack([np.cos(ang), np.sin(ang)], -1)   # (gy, gx, 2)
        ys = np.arange(h, dtype=np.float32) / c
        xs = np.arange(w, dtype=np.float32) / c
        yi = np.floor(ys).astype(int)
        xi = np.floor(xs).astype(int)
        yf = (ys - yi)[:, None]
        xf = (xs - xi)[None, :]
        g00 = grad[yi][:, xi]
        g01 = grad[yi][:, xi + 1]
        g10 = grad[yi + 1][:, xi]
        g11 = grad[yi + 1][:, xi + 1]
        d00 = g00[..., 0] * xf + g00[..., 1] * yf
        d01 = g01[..., 0] * (xf - 1) + g01[..., 1] * yf
        d10 = g10[..., 0] * xf + g10[..., 1] * (yf - 1)
        d11 = g11[..., 0] * (xf - 1) + g11[..., 1] * (yf - 1)
        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)      # smoothstep^5
        v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
        n0 = d00 + u * (d01 - d00)
        n1 = d10 + u * (d11 - d10)
        total += amp * (n0 + v * (n1 - n0))
        amp_sum += amp
        amp *= 0.5
    total /= max(amp_sum, 1e-6)
    return np.clip(total * 0.7071 + 0.5, 0.0, 1.0).astype(np.float32)


def render_layer(h, w, p, scale=1.0):
    """1 レイヤーを float32 (h, w) 0..1 で描画する。

    p: LayerWidget.params() が返す dict。
    scale: プレビュー縮小率 (px 単位のパラメータ — 周期/ドット/ブラー/セル —
           に乗算して見た目を実サイズと一致させる)。
    """
    h, w = int(h), int(w)
    pid = p.get("pattern", "solid_gray")
    if pid == "wave":
        img16 = render_pattern(
            h, w, "wave",
            amplitude=p.get("amp", 1.0),
            period=max(1.0, p.get("period", h) * scale),
            phase_deg=p.get("phase", 0.0),
            angle_deg=p.get("angle", 0.0),
        )
        return img16.astype(np.float32) / 65535.0
    if pid == "random":
        dot = max(1, int(round(p.get("dot", 1) * scale)))
        rng = np.random.default_rng(int(p.get("nseed", 0)))
        gh = max(1, int(np.ceil(h / dot)))
        gw = max(1, int(np.ceil(w / dot)))
        base = rng.random((gh, gw), dtype=np.float32)
        img = np.repeat(np.repeat(base, dot, 0), dot, 1)[:h, :w]
        sigma = float(p.get("blur", 0.0)) * scale
        if sigma > 0.1:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        return np.clip(np.ascontiguousarray(img), 0.0, 1.0)
    if pid == "perlin":
        return perlin2d(h, w, max(2.0, p.get("cell", 64) * scale),
                        octaves=p.get("octaves", 3), seed=p.get("pseed", 0))
    if pid == "image":
        path = p.get("image_path")
        if path and os.path.exists(path):
            m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if m is not None:
                if m.ndim == 3:
                    m = m[..., 0]
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
                mx = 65535.0 if m.dtype == np.uint16 else 255.0
                return np.clip(m.astype(np.float32) / mx, 0.0, 1.0)
        return np.full((h, w), 0.5, np.float32)   # 未選択/読込失敗 → 50% グレー
    # 基本グラデーション / グレー
    img16 = render_pattern(h, w, pid)
    return img16.astype(np.float32) / 65535.0


def apply_blend(base, img, mode):
    """float 0..1 同士のブレンド。"""
    if mode == "add":
        return np.clip(base + img, 0.0, 1.0)
    if mode == "subtract":
        return np.clip(base - img, 0.0, 1.0)
    if mode == "multiply":
        return base * img
    if mode == "screen":
        return 1.0 - (1.0 - base) * (1.0 - img)
    if mode == "difference":
        return np.abs(base - img)
    return img   # normal


def composite_layers(h, w, layer_params, scale=1.0):
    """レイヤースタックを上から順に合成し uint16 (h, w) を返す。

    layer_params[0] がベース。以降の各レイヤーは
        result = base × (1 - opacity) + blend(base, layer) × opacity
    で積み重なる (opacity は 0..100 の %)。
    """
    if not layer_params:
        return np.full((h, w), 32767, np.uint16)
    acc = render_layer(h, w, layer_params[0], scale)
    for p in layer_params[1:]:
        img = render_layer(h, w, p, scale)
        op = min(100, max(0, p.get("opacity", 100))) / 100.0
        blended = apply_blend(acc, img, p.get("blend", "normal"))
        acc = np.clip(acc * (1.0 - op) + blended * op, 0.0, 1.0)
    return (acc * 65535.0 + 0.5).astype(np.uint16)


def sample_filename(image_type, space_range=None, time_vmin=None,
                    time_vmax=None, rate_maxdev=None, scan_size=None):
    """img_to_maneuver の extract_params_from_filename 規約のファイル名。"""
    if image_type == "space":
        if space_range is None:
            space_range = scan_size
        return f"sample_space_{int(space_range)}.png"
    if image_type == "time":
        return f"sample_time_{int(time_vmin or 0)}-{int(time_vmax or 100)}.png"
    if image_type == "rate":
        return f"sample_rate_{rate_maxdev if rate_maxdev is not None else 0.5}.png"
    raise ValueError(f"image_type must be space/time/rate, got {image_type!r}")


def generate_sample_image(out_dir, image_type, pattern_id,
                          scan_size, time_size,
                          scan_direction,
                          space_range=None, time_vmin=None, time_vmax=None,
                          rate_maxdev=None,
                          wave_angle_deg=0.0, wave_amplitude=1.0,
                          wave_period=None, wave_phase_deg=0.0):
    """サンプル画像を生成してパスを返す。

    image_type: "space" / "time" / "rate"
    scan_direction: 1=vertical slit, 0=horizontal slit
        - vertical:   file shape (H, W) = (time_size, scan_size)
        - horizontal: file shape (H, W) = (scan_size, time_size)  ※img_to_maneuver が .T するため

    pattern_id == "wave" の場合の追加パラメータ:
        wave_angle_deg : 波の角度 (0°=上下, 90°=左右, 任意の斜めも可)
        wave_amplitude : 0.0 - 1.0
        wave_period    : ピクセル数 (None なら高さ方向サイズと同じ → 1周期)
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

    # Wave のデフォルト period (高さ方向サイズ)
    if pattern_id == "wave" and wave_period is None:
        wave_period = h_pix

    img16 = render_pattern(
        h_pix, w_pix, pattern_id,
        amplitude=wave_amplitude,
        period=wave_period,
        phase_deg=wave_phase_deg,
        angle_deg=wave_angle_deg,
    )

    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, img16)
    return out_path


# ======== Layer editor widget ========
class LayerWidget(QFrame):
    """セクションジェネレータの 1 レイヤー分の編集 UI。

    パターン (基本 + パーリン + 画像ファイル) と、そのパターン固有の
    パラメータ、レイヤー 2 枚目以降は合成モード + 不透明度を持つ。
    値が変わるたび changed を emit し、親がプレビューを再合成する。
    """
    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, section, index, sd=1):
        super().__init__()
        self.section = section
        self.index = index
        self.sd = int(sd)          # スリット方向 (通常再生パターンの判定に使用)
        self._image_path = None
        self.pattern_ids = []
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("LayerWidget { background: rgba(128,128,128,20); }")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 4, 8, 4)
        v.setSpacing(3)

        # ヘッダ (レイヤー番号 + 削除)
        head = QHBoxLayout()
        self.head_label = QLabel()
        self.head_label.setStyleSheet("font-weight: bold; color: #557;")
        head.addWidget(self.head_label)
        head.addStretch()
        # 削除ボタン: はっきり見える ✕ (赤系・ホバーで強調)
        self.remove_btn = QPushButton("✕")
        self.remove_btn.setFixedSize(30, 26)
        self.remove_btn.setCursor(Qt.PointingHandCursor)
        self.remove_btn.setToolTip("このレイヤーを削除 / Delete this layer")
        self.remove_btn.setStyleSheet(
            "QPushButton { background: #fbe9e9; color: #c0392b; border: 1px solid #d98880;"
            " border-radius: 5px; font-size: 14px; font-weight: bold; padding: 0; }"
            "QPushButton:hover { background: #e74c3c; color: white; border-color: #c0392b; }"
            "QPushButton:disabled { background: transparent; color: #bbb; border-color: #ddd; }")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        head.addWidget(self.remove_btn)
        v.addLayout(head)

        # 合成モード + 不透明度 (レイヤー 2 枚目以降のみ表示)
        self.blend_frame = QFrame()
        bl = QHBoxLayout(self.blend_frame)
        bl.setContentsMargins(0, 0, 0, 0)
        self.blend_label = QLabel()
        bl.addWidget(self.blend_label)
        self.blend = QComboBox()
        bl.addWidget(self.blend)
        self.opacity_label = QLabel()
        bl.addWidget(self.opacity_label)
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(100)
        self.opacity_spin.setSuffix(" %")
        bl.addWidget(self.opacity_spin)
        bl.addStretch()
        v.addWidget(self.blend_frame)

        # パターン選択
        pr = QHBoxLayout()
        self.pattern_label = QLabel()
        pr.addWidget(self.pattern_label)
        self.pattern = QComboBox()
        pr.addWidget(self.pattern, 1)
        v.addLayout(pr)

        # --- Wave パラメータ ---
        self.wave_frame = QFrame()
        wf = QHBoxLayout(self.wave_frame)
        wf.setContentsMargins(0, 0, 0, 0)
        self.wave_amp_label = QLabel()
        wf.addWidget(self.wave_amp_label)
        self.wave_amp = QDoubleSpinBox()
        self.wave_amp.setRange(0.0, 1.0); self.wave_amp.setDecimals(3)
        self.wave_amp.setSingleStep(0.05); self.wave_amp.setValue(1.0)
        wf.addWidget(self.wave_amp)
        self.wave_period_label = QLabel()
        wf.addWidget(self.wave_period_label)
        self.wave_period = QSpinBox()
        self.wave_period.setRange(1, 32768); self.wave_period.setValue(120)
        wf.addWidget(self.wave_period)
        self.wave_phase_label = QLabel()
        wf.addWidget(self.wave_phase_label)
        self.wave_phase = QDoubleSpinBox()
        self.wave_phase.setRange(-360.0, 720.0); self.wave_phase.setDecimals(1)
        self.wave_phase.setSingleStep(15.0); self.wave_phase.setValue(0.0)
        wf.addWidget(self.wave_phase)
        self.wave_angle_label = QLabel()
        wf.addWidget(self.wave_angle_label)
        self.wave_angle = QDoubleSpinBox()
        self.wave_angle.setRange(0.0, 360.0); self.wave_angle.setDecimals(1)
        self.wave_angle.setSingleStep(5.0); self.wave_angle.setValue(0.0)
        wf.addWidget(self.wave_angle)
        wf.addStretch()
        v.addWidget(self.wave_frame)

        # --- Random ノイズパラメータ (ドット / ブラー / シード) ---
        self.noise_frame = QFrame()
        nf = QHBoxLayout(self.noise_frame)
        nf.setContentsMargins(0, 0, 0, 0)
        self.dot_label = QLabel()
        nf.addWidget(self.dot_label)
        self.dot_spin = QSpinBox()
        self.dot_spin.setRange(1, 512); self.dot_spin.setValue(1)
        self.dot_spin.setSuffix(" px")
        nf.addWidget(self.dot_spin)
        self.blur_label = QLabel()
        nf.addWidget(self.blur_label)
        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0.0, 128.0); self.blur_spin.setDecimals(1)
        self.blur_spin.setSingleStep(0.5); self.blur_spin.setValue(0.0)
        self.blur_spin.setSuffix(" px")
        nf.addWidget(self.blur_spin)
        self.nseed_label = QLabel()
        nf.addWidget(self.nseed_label)
        self.nseed_spin = QSpinBox()
        self.nseed_spin.setRange(0, 99999)
        self.nseed_spin.setValue(int(np.random.default_rng().integers(0, 10000)))
        nf.addWidget(self.nseed_spin)
        nf.addStretch()
        v.addWidget(self.noise_frame)

        # --- Perlin ノイズパラメータ (スケール / オクターブ / シード) ---
        self.perlin_frame = QFrame()
        pf = QHBoxLayout(self.perlin_frame)
        pf.setContentsMargins(0, 0, 0, 0)
        self.cell_label = QLabel()
        pf.addWidget(self.cell_label)
        self.cell_spin = QSpinBox()
        self.cell_spin.setRange(2, 4096); self.cell_spin.setValue(64)
        self.cell_spin.setSuffix(" px")
        pf.addWidget(self.cell_spin)
        self.oct_label = QLabel()
        pf.addWidget(self.oct_label)
        self.oct_spin = QSpinBox()
        self.oct_spin.setRange(1, 6); self.oct_spin.setValue(3)
        pf.addWidget(self.oct_spin)
        self.pseed_label = QLabel()
        pf.addWidget(self.pseed_label)
        self.pseed_spin = QSpinBox()
        self.pseed_spin.setRange(0, 99999)
        self.pseed_spin.setValue(int(np.random.default_rng().integers(0, 10000)))
        pf.addWidget(self.pseed_spin)
        pf.addStretch()
        v.addWidget(self.perlin_frame)

        # --- 画像ファイル ---
        self.image_frame = QFrame()
        imf = QHBoxLayout(self.image_frame)
        imf.setContentsMargins(0, 0, 0, 0)
        self.image_btn = QPushButton()
        self.image_btn.clicked.connect(self._pick_image)
        imf.addWidget(self.image_btn)
        self.image_label = QLabel()
        self.image_label.setStyleSheet("color: gray; font-size: 10px;")
        imf.addWidget(self.image_label, 1)
        v.addWidget(self.image_frame)

        # 初期テキスト/combo 構築
        self.retranslate()
        # 既定パターン: ベースレイヤーはセクションの通常再生 (sd 依存)、追加レイヤーはグレー
        default_pid = normal_pattern_for(section, self.sd) if index == 0 else "solid_gray"
        if default_pid in self.pattern_ids:
            self.pattern.setCurrentIndex(self.pattern_ids.index(default_pid))
        self.set_index(index)
        self._on_pattern()

        # 変更シグナル配線
        self.pattern.currentIndexChanged.connect(self._on_pattern)
        self.blend.currentIndexChanged.connect(lambda *_: self.changed.emit())
        for sp in (self.opacity_spin, self.wave_amp, self.wave_period,
                   self.wave_phase, self.wave_angle, self.dot_spin,
                   self.blur_spin, self.nseed_spin, self.cell_spin,
                   self.oct_spin, self.pseed_spin):
            sp.valueChanged.connect(lambda *_: self.changed.emit())

    # --- helpers ---
    def set_index(self, index):
        """レイヤー番号の更新 (削除後の再番号付けにも使う)。"""
        self.index = index
        self.head_label.setText(tr("lbl_layer", n=index + 1))
        self.blend_frame.setVisible(index > 0)
        self.remove_btn.setVisible(index > 0)

    def current_pattern_id(self):
        i = self.pattern.currentIndex()
        return self.pattern_ids[i] if 0 <= i < len(self.pattern_ids) else "solid_gray"

    def _on_pattern(self, *_):
        pid = self.current_pattern_id()
        self.wave_frame.setVisible(pid == "wave")
        self.noise_frame.setVisible(pid == "random")
        self.perlin_frame.setVisible(pid == "perlin")
        self.image_frame.setVisible(pid == "image")
        self.changed.emit()

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select layer image", "", "Images (*.png *.jpg *.bmp *.tif)")
        if not path:
            return
        self._image_path = path
        self.image_label.setText(os.path.basename(path))
        self.changed.emit()

    def params(self):
        return {
            "pattern": self.current_pattern_id(),
            "amp": self.wave_amp.value(),
            "period": self.wave_period.value(),
            "phase": self.wave_phase.value(),
            "angle": self.wave_angle.value(),
            "dot": self.dot_spin.value(),
            "blur": self.blur_spin.value(),
            "nseed": self.nseed_spin.value(),
            "cell": self.cell_spin.value(),
            "octaves": self.oct_spin.value(),
            "pseed": self.pseed_spin.value(),
            "image_path": self._image_path,
            "blend": BLEND_IDS[max(0, self.blend.currentIndex())],
            "opacity": self.opacity_spin.value(),
        }

    def retranslate(self):
        """現在言語でラベル/combo を再構築 (選択は保持)。"""
        self.head_label.setText(tr("lbl_layer", n=self.index + 1))
        self.blend_label.setText(tr("lbl_blend"))
        self.opacity_label.setText(tr("lbl_opacity"))
        self.pattern_label.setText(tr("lbl_pattern"))
        self.wave_amp_label.setText(tr("lbl_wave_amp"))
        self.wave_period_label.setText(tr("lbl_wave_period"))
        self.wave_phase_label.setText(tr("lbl_wave_phase"))
        self.wave_angle_label.setText(tr("lbl_wave_angle"))
        self.dot_label.setText(tr("lbl_dot"))
        self.blur_label.setText(tr("lbl_blur"))
        self.nseed_label.setText(tr("lbl_seed"))
        self.cell_label.setText(tr("lbl_cell"))
        self.oct_label.setText(tr("lbl_octaves"))
        self.pseed_label.setText(tr("lbl_seed"))
        self.image_btn.setText(tr("btn_layer_image"))
        if not self._image_path:
            self.image_label.setText(tr("no_layer_image"))
        # pattern combo (選択保持)
        ids, labels = layer_pattern_order(self.section, sd=self.sd)
        idx = self.pattern.currentIndex() if self.pattern.count() else 0
        self.pattern.blockSignals(True)
        self.pattern.clear()
        self.pattern.addItems(labels)
        self.pattern.setCurrentIndex(max(0, min(idx, len(labels) - 1)))
        self.pattern.blockSignals(False)
        self.pattern_ids = ids
        # blend combo (選択保持)
        bidx = self.blend.currentIndex() if self.blend.count() else 0
        blabels = BLEND_LABELS.get(LANG, BLEND_LABELS["ja"])
        self.blend.blockSignals(True)
        self.blend.clear()
        self.blend.addItems(blabels)
        self.blend.setCurrentIndex(max(0, min(bidx, len(blabels) - 1)))
        self.blend.blockSignals(False)


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
            self.play_btn = QPushButton(tr("btn_pause"))
            self.play_btn.clicked.connect(self._toggle)
            ctl.addWidget(self.play_btn)
            self.open_btn = QPushButton(tr("btn_open_external"))
            self.open_btn.clicked.connect(self._open_external)
            ctl.addWidget(self.open_btn)
            ctl.addStretch()
            v.addLayout(ctl)
        else:
            self.info_label = QLabel(tr("no_multimedia"))
            self.info_label.setWordWrap(True)
            self.info_label.setStyleSheet("color: #a66; font-size: 11px;")
            v.addWidget(self.info_label)
            self.open_btn = QPushButton(tr("btn_open_external"))
            self.open_btn.clicked.connect(self._open_external)
            v.addWidget(self.open_btn)

        self.setVisible(False)

    def set_base_title(self, base_title):
        """タイトルおよびボタン等のテキストを現在言語で更新する (言語切替時に呼ばれる)。"""
        self._base_title = base_title
        if self.loaded and self.path:
            self.title_label.setText(f"{base_title}: {os.path.basename(self.path)}")
        else:
            self.title_label.setText(base_title)
        if hasattr(self, "open_btn"):
            self.open_btn.setText(tr("btn_open_external"))
        if hasattr(self, "info_label"):
            self.info_label.setText(tr("no_multimedia"))
        if hasattr(self, "play_btn") and HAS_MULTIMEDIA:
            playing = self.player.state() == QMediaPlayer.PlayingState
            self.play_btn.setText(tr("btn_pause") if playing else tr("btn_play"))

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
            self.play_btn.setText(tr("btn_pause"))
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
            self.play_btn.setText(tr("btn_play"))
        else:
            self.player.play()
            self.play_btn.setText(tr("btn_pause"))

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
    percent_signal = pyqtSignal(int)          # ステージ基準のおおよその進捗 (0-100)
    done_signal = pyqtSignal(bool, str, str)  # success, plot2d_path, gif_path

    def __init__(self, dm, mode, space_img, time_img, rate_img,
                 space_set, time_vmin, time_vmax,
                 rate_maxdev, rate_baseline, rate_startpoint,
                 anim_frames=20, anim_fps=10, anim_dpi=80,
                 skip_2d=False):
        super().__init__()
        self.dm = dm
        self.mode = mode  # "time" or "rate"
        self.skip_2d = skip_2d   # ライブ3Dプレビュー用: 2D プロット生成を省略
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
            self.percent_signal.emit(5)
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
            self.percent_signal.emit(25)
            self.dm.zPointCheck()

            # 2D プロット生成: mtime で「呼び出し後に変更されたファイル」を検出
            # (同じファイル名で上書きされるケースに対応するため set 差分は使わない)
            plot2d = ""
            if not self.skip_2d:
                ts_2d = time.time() - 0.5  # 小さなクロックスラックを許容
                self.progress_signal.emit("maneuver_2dplot: 2D プロット生成中…")
                self.percent_signal.emit(35)
                self.dm.maneuver_2dplot()
                plot2d = self._latest_file(cwd, (".png",), ts_2d)

            # 3D アニメ生成: 同じく mtime で検出
            ts_3d = time.time() - 0.5
            self.progress_signal.emit(
                f"maneuver_3dplot: 3D アニメ生成中 ({self.anim_frames} frames @ {self.anim_dpi} dpi)…"
            )
            self.percent_signal.emit(55)
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
                self.percent_signal.emit(85)
                # 横幅 400 にスケール (高さは自動)、ループ無限
                cmd = ["ffmpeg", "-y", "-i", mp4,
                       "-vf", f"fps={self.anim_fps},scale=400:-1:flags=lanczos",
                       "-loop", "0", gif]
                proc = subprocess.run(cmd, capture_output=True)
                if proc.returncode != 0:
                    self.progress_signal.emit(f"[WARN] GIF 変換失敗: {proc.stderr.decode('utf-8', 'ignore')[:200]}")
                    gif = ""

            self.progress_signal.emit("完了")
            self.percent_signal.emit(100)
            self.done_signal.emit(True, plot2d, gif)
        except Exception as e:
            self.progress_signal.emit(f"[ERROR] {e}")
            self.done_signal.emit(False, "", "")


# ======== 適用済みマップのサムネイル (再生位置の赤ライン付き) ========
class MapThumb(QLabel):
    """適用済みの space/time/rate マップを小さく表示し、3D アニメの再生位置を
    赤いラインで重ねるサムネイル。

    - 縦スリット (sd=1): マップの時間軸は縦 → 水平の赤ラインが上下に動く
    - 横スリット (sd=0): マップの時間軸は横 → 垂直の赤ラインが左右に動く
    """

    def __init__(self, caption="", fixed_height=110):
        super().__init__()
        self._src = None            # 元画像 QPixmap
        self._base = None           # ラベルサイズに合わせた縮小キャッシュ
        self._frac = None           # 再生位置 [0,1) / None = 非表示
        self._time_vertical = True
        self.setAlignment(Qt.AlignCenter)
        if fixed_height is not None:
            self.setFixedHeight(fixed_height)
            self.setMinimumWidth(100)
        else:
            # 可変サイズ (レイアウトのストレッチに従う)。sizeHint 由来の
            # 拡大ループを避けるため Ignored ポリシーにする。
            self.setMinimumSize(160, 240)
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet(
            "QLabel { background: #222; border: 1px solid #555;"
            " color: #777; font-size: 10px; }")
        self.setText(caption)

    def set_time_vertical(self, vertical):
        self._time_vertical = bool(vertical)
        self._recompose()

    def set_map(self, path):
        pm = QPixmap()
        if path and os.path.exists(path) and pm.load(path):
            self._src = pm
        else:
            self._src = None
        self._rescale()

    def set_playhead(self, frac):
        self._frac = frac
        self._recompose()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()

    def _rescale(self):
        if self._src is None:
            self._base = None
            return
        self._base = self._src.scaled(
            max(10, self.width() - 2), max(10, self.height() - 2),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._recompose()

    def _recompose(self):
        if self._base is None:
            return
        pm = QPixmap(self._base)
        if self._frac is not None:
            p = QPainter(pm)
            pen = QPen(QColor(255, 40, 40))
            pen.setWidth(2)
            p.setPen(pen)
            if self._time_vertical:
                y = int(self._frac * (pm.height() - 1))
                p.drawLine(0, y, pm.width(), y)
            else:
                x = int(self._frac * (pm.width() - 1))
                p.drawLine(x, 0, x, pm.height())
            p.end()
        self.setPixmap(pm)


# ======== Main GUI ========
class IMGTransApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(1360, 900)   # 3カラム (Space/Time/Rate) を横並びで収める幅
        self.setMinimumSize(640, 480)

        self.videopath = None
        self.space_img_path = None
        self.time_img_path = None
        self.rate_img_path = None
        self.dm = None
        self.worker = None
        self.render_completed = False
        self._preview_stale = False

        # 3D軌道ライブプレビュー (タブ2) の状態
        self._live3d_worker = None
        self._live3d_busy = False
        self._live3d_pending = False
        self._live3d_movie = None
        self._live3d_timer = QTimer(self)
        self._live3d_timer.setSingleShot(True)
        self._live3d_timer.setInterval(800)     # 編集のデバウンス
        self._live3d_timer.timeout.connect(self._run_live3d)

        # i18n: 再翻訳用コールバックの登録簿。各エントリは呼ぶと現在の LANG で
        # 対応 widget のテキストを更新する。
        self._i18n = []

        self.init_ui()

        # 「未選択」系ラベルは画像/動画ロード時にファイル名で上書きされるため、
        # 言語切替時は「未ロードのときだけ」既定文言を訳し直す (条件付き登録)。
        self._i18n.append(lambda: (None if self.videopath else self.video_label.setText(tr("no_video"))))
        self._i18n.append(lambda: (None if self.dm else self.info_label.setText(tr("video_not_init"))))
        self._i18n.append(lambda: (None if self.space_img_path else self.space_label.setText(tr("no_space_image"))))
        self._i18n.append(lambda: (None if self.time_img_path else self.time_label.setText(tr("no_time_image"))))
        self._i18n.append(lambda: (None if self.rate_img_path else self.rate_label.setText(tr("no_rate_image"))))
        # ライブプロットのプレースホルダ (未生成時のみ訳し直す)
        self._i18n.append(lambda: (None if self._live3d_movie else self.live3d_label.setText(tr("live3d_waiting"))))
        self._i18n.append(lambda: (None if (self.live2d_thumb.pixmap() and not self.live2d_thumb.pixmap().isNull()) else self.live2d_thumb.setText(tr("live3d_waiting"))))

        self.update_ui_state("initial")

    # --- i18n helpers ---
    def _reg(self, fn):
        """再翻訳コールバック fn を登録し、初期テキスト適用のため即実行する。"""
        self._i18n.append(fn)
        fn()

    def _trlabel(self, key, **fmt):
        """tr(key) を表示し、言語切替時に自動更新される QLabel を返す。"""
        lbl = QLabel()
        self._reg(lambda l=lbl, k=key, f=fmt: l.setText(tr(k, **f)))
        return lbl

    def on_language_changed(self, *_):
        global LANG
        sel = self.lang_select.currentData()
        if sel in ("ja", "en") and sel != LANG:
            LANG = sel
            self.retranslate_ui()

    def retranslate_ui(self):
        """登録済みの全 i18n コールバックを再実行して UI を現在言語に更新する。"""
        self.setWindowTitle(tr("window_title"))
        for fn in self._i18n:
            try:
                fn()
            except Exception:
                pass
        # パターン/波形の QComboBox は項目テキストの入れ替えが必要
        for t in getattr(self, "_section_gens", {}):
            self._retranslate_section_combo(t)
            # プレビュー未生成 (pixmap 無し) のプレースホルダのみ差し替え
            lbl = self._section_gens[t].get('preview_label')
            if lbl is not None and (lbl.pixmap() is None or lbl.pixmap().isNull()):
                lbl.setText(tr("preview_after_init"))
        # マニューバプレビューのプレースホルダ (未生成時のみ)
        for lbl in (getattr(self, "preview_2dplot_label", None),
                    getattr(self, "preview_3d_label", None)):
            if lbl is not None and (lbl.pixmap() is None or lbl.pixmap().isNull()) \
                    and lbl.movie() is None:
                lbl.setText(tr("preview_after_gen"))
        # ステータス表示は idle 相当のときだけ翻訳を反映
        if hasattr(self, "preview_status_label"):
            self._update_preview_btn_state()
        # リアルタイムプレビューの言語も切替
        if getattr(self, "rt_preview", None):
            self.rt_preview.set_lang(LANG)

    def _retranslate_section_combo(self, type_name):
        """セクションの全レイヤーを現在言語で再構築する (選択は保持)。"""
        g = self._section_gens.get(type_name, {})
        for lw in g.get('layers', []):
            lw.retranslate()

    # --- UI Setup ---
    def _wrap_scroll(self, widget):
        """タブのコンテンツ widget を QScrollArea で包む (縦に長くてもスクロール可能)"""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def init_ui(self):
        # --- Language selector (Setup タブ上部に配置) ---
        self.lang_label = self._trlabel("lang_label")
        self.lang_select = QComboBox()
        self.lang_select.addItem("日本語", "ja")
        self.lang_select.addItem("English", "en")
        self.lang_select.setCurrentIndex(0 if LANG == "ja" else 1)
        self.lang_select.currentIndexChanged.connect(self.on_language_changed)
        lang_row = QHBoxLayout()
        lang_row.addWidget(self.lang_label)
        lang_row.addWidget(self.lang_select)
        lang_row.addStretch()
        self.lang_row = lang_row

        # --- Video file ---
        # パス表示は小さめ (左カラムを圧迫しないように)
        self.video_label = QLabel(tr("no_video"))
        self.video_label.setWordWrap(True)
        self.video_label.setStyleSheet("color: gray; font-size: 10px;")
        self.video_btn = QPushButton()
        self._reg(lambda: self.video_btn.setText(tr("btn_select_video")))
        self.video_btn.clicked.connect(self.select_video)

        # --- Slit toggle ---
        self.slit_toggle = QCheckBox()
        self._reg(lambda: self.slit_toggle.setText(tr("chk_vertical")))
        self.slit_label = QLabel(tr("slit_h"))
        self.slit_toggle.stateChanged.connect(self.update_slit_label)
        self._reg(self.update_slit_label)  # 言語切替時にスリット表示も更新

        # --- Initialize ---
        self.init_btn = QPushButton()
        self._reg(lambda: self.init_btn.setText(tr("btn_initialize")))
        self.init_btn.clicked.connect(self.initialize_drawmaneuver)
        self.info_label = QLabel(tr("video_not_init"))
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: gray; font-size: 10px;")

        # ===== 共通サイズ設定 (Space/Time/Rate で共有) =====
        # img_to_maneuver は space と time/rate 画像の形状一致を要求するため、サイズは共有。
        # パターン/波形/プレビューは各セクション (Space/Time/Rate) に個別配置。
        self.gen_group = QGroupBox()
        self._reg(lambda: self.gen_group.setTitle(tr("grp_shared_size")))
        gen_v = QVBoxLayout(self.gen_group)

        s_layout = QHBoxLayout()
        s_layout.addWidget(self._trlabel("lbl_scan_size"))
        self.gen_scan_size = QSpinBox()
        self.gen_scan_size.setRange(16, 32768)
        self.gen_scan_size.setValue(1920)
        s_layout.addWidget(self.gen_scan_size)
        s_layout.addWidget(self._trlabel("hint_scan_auto"))
        gen_v.addLayout(s_layout)

        t2_layout = QHBoxLayout()
        t2_layout.addWidget(self._trlabel("lbl_time_size"))
        self.gen_time_size = QSpinBox()
        self.gen_time_size.setRange(2, 32768)
        self.gen_time_size.setValue(900)
        t2_layout.addWidget(self.gen_time_size)
        t2_layout.addWidget(self._trlabel("hint_time_any"))
        gen_v.addLayout(t2_layout)

        # 出力FPS (選択制) — 最終映像の尺は「時間方向サイズ ÷ 出力FPS」で決まる
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(self._trlabel("lbl_out_fps"))
        self.gen_out_fps = QComboBox()
        for f in (10, 24, 30, 60, 120):
            self.gen_out_fps.addItem(str(f), f)
        self.gen_out_fps.setCurrentIndex(2)   # 30
        fps_layout.addWidget(self.gen_out_fps)
        fps_layout.addWidget(self._trlabel("hint_out_fps"))
        gen_v.addLayout(fps_layout)

        self.gen_hint = QLabel("")
        self.gen_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.gen_hint.setWordWrap(True)
        gen_v.addWidget(self.gen_hint)

        self.gen_group.setVisible(False)

        # サイズ変更時は全セクションのプレビューを更新
        self.gen_scan_size.valueChanged.connect(self._update_gen_hint)
        self.gen_time_size.valueChanged.connect(self._update_gen_hint)
        self.gen_out_fps.currentIndexChanged.connect(self._update_gen_hint)
        self.gen_scan_size.valueChanged.connect(self._update_all_section_previews)
        self.gen_time_size.valueChanged.connect(self._update_all_section_previews)
        # 時間方向サイズ / 出力FPS はリアルタイムプレビューのタイムラインにも反映
        self.gen_time_size.valueChanged.connect(self._sync_rt_timeline)
        self.gen_out_fps.currentIndexChanged.connect(self._sync_rt_timeline)
        # Time 画像の vmin/vmax 既定値 (0 .. 出力FPS×時間方向サイズ) を追従更新
        self.gen_time_size.valueChanged.connect(self._maybe_update_time_defaults)
        self.gen_out_fps.currentIndexChanged.connect(self._maybe_update_time_defaults)

        # 各セクション (Space/Time/Rate) の独立ジェネレータ widget bundle を保持
        self._section_gens = {}

        # --- Space image ---
        # 画像の指定はジェネレータパネル (パターン: 画像ファイル…) に統合済み。
        # 単独の Select ボタンは廃止。
        self.space_label = QLabel(tr("no_space_image"))
        self.space_label.setWordWrap(True)

        sp_layout = QHBoxLayout()
        sp_label = self._trlabel("lbl_space_range")
        self.space_set_value = QSpinBox()
        self.space_set_value.setRange(0, 999999)
        sp_layout.addWidget(sp_label)
        sp_layout.addWidget(self.space_set_value)

        self.space_info_label = QLabel("")
        self.space_info_label.setStyleSheet("color: gray; font-size: 10px;")

        self.space_param_frame = QFrame()
        sp_vbox = QVBoxLayout(self.space_param_frame)
        sp_vbox.addLayout(sp_layout)
        sp_vbox.addWidget(self.space_info_label)
        self.space_param_frame.setVisible(False)

        # Space 用のジェネレータパネル (パターン / 波形エディタ / プレビュー)
        self.space_gen_frame = self._build_section_gen('space')

        # --- Time image ---
        self.time_label = QLabel(tr("no_time_image"))
        self.time_label.setWordWrap(True)

        time_layout = QHBoxLayout()
        self.time_vmin_spin = QSpinBox()
        self.time_vmax_spin = QSpinBox()
        self.time_vmin_spin.setRange(-999999, 999999)
        self.time_vmax_spin.setRange(-999999, 999999)
        time_layout.addWidget(self._trlabel("lbl_vmin"))
        time_layout.addWidget(self.time_vmin_spin)
        time_layout.addWidget(self._trlabel("lbl_vmax"))
        time_layout.addWidget(self.time_vmax_spin)

        self.time_info_label = QLabel("")
        self.time_info_label.setStyleSheet("color: gray; font-size: 10px;")

        self.time_param_frame = QFrame()
        time_vbox = QVBoxLayout(self.time_param_frame)
        time_vbox.addLayout(time_layout)
        time_vbox.addWidget(self.time_info_label)
        self.time_param_frame.setVisible(False)

        # Time 用のジェネレータパネル
        self.time_gen_frame = self._build_section_gen('time')

        # --- Rate image ---
        self.rate_label = QLabel(tr("no_rate_image"))
        self.rate_label.setWordWrap(True)

        rate_layout = QHBoxLayout()
        rate_layout.addWidget(self._trlabel("lbl_baseline"))
        self.rate_baseline_spin = QDoubleSpinBox()
        self.rate_baseline_spin.setRange(0.0, 999999.0)
        self.rate_baseline_spin.setDecimals(3)
        rate_layout.addWidget(self.rate_baseline_spin)
        rate_layout.addWidget(self._trlabel("lbl_max_range"))
        self.rate_maxdev_spin = QDoubleSpinBox()
        self.rate_maxdev_spin.setRange(0.0, 999999.0)
        self.rate_maxdev_spin.setDecimals(3)
        rate_layout.addWidget(self.rate_maxdev_spin)
        rate_layout.addWidget(self._trlabel("lbl_start_frame"))
        self.rate_startpoint_spin = QDoubleSpinBox()
        self.rate_startpoint_spin.setRange(-999999, 999999)
        rate_layout.addWidget(self.rate_startpoint_spin)

        self.rate_info_label = QLabel("")
        self.rate_info_label.setStyleSheet("color: gray; font-size: 10px;")
        self.rate_param_frame = QFrame()
        rate_vbox = QVBoxLayout(self.rate_param_frame)
        rate_vbox.addLayout(rate_layout)
        rate_vbox.addWidget(self.rate_info_label)
        self.rate_param_frame.setVisible(False)

        # Rate 用のジェネレータパネル
        self.rate_gen_frame = self._build_section_gen('rate')

        # ===== 適用方法の選択 (タブ2 下部・必須) =====
        # ここで選択しない限り「3. プレビュー」「4. 出力」タブは開かない。
        # combo の項目テキストはロジックの識別子も兼ねるため翻訳しない。
        self.apply_mode_group = QGroupBox()
        self._reg(lambda: self.apply_mode_group.setTitle(tr("grp_apply_mode")))
        am_v = QVBoxLayout(self.apply_mode_group)
        am_hint = self._trlabel("apply_mode_hint")
        am_hint.setStyleSheet("color: gray; font-size: 11px;")
        am_hint.setWordWrap(True)
        am_v.addWidget(am_hint)
        am_row = QHBoxLayout()
        am_row.addWidget(self._trlabel("lbl_gen_method"))
        self.preview_mode_select = QComboBox()
        self.preview_mode_select.addItems(
            ["― 選択 / Select ―", "time to data", "rate to data"])
        self.preview_mode_select.currentIndexChanged.connect(self.on_apply_mode_changed)
        am_row.addWidget(self.preview_mode_select)
        am_row.addStretch()
        am_v.addLayout(am_row)
        self.apply_mode_group.setVisible(False)  # Initialize 後に表示

        # ===== 軌道プロット ライブプレビュー (3D | 2D の2カラム・自動更新) =====
        # 画像/パラメータ/適用方法を編集するたびにデバウンス後、軽量設定で
        # maneuver_3dplot (GIF) + maneuver_2dplot (PNG) を再生成して表示する。
        self.live3d_group = QGroupBox()
        self._reg(lambda: self.live3d_group.setTitle(tr("grp_live3d")))
        # レイアウト: [2D プロット (左・幅2/5, 再生赤ライン付き)]
        #             [右 3/5: 上=3D GIF / 下=Space・Time・Rate サムネイル]
        # 全体は縦方向センタリング (下側の空白を防ぐ)
        l3_outer = QVBoxLayout(self.live3d_group)
        l3_outer.addStretch(1)
        l3_cols = QHBoxLayout()

        # 左: 2D プロット (MapThumb — 赤ラインが常に左→右へスライド)
        self.live2d_thumb = MapThumb("2D Plot", fixed_height=None)
        self.live2d_thumb.set_time_vertical(False)   # 2D の時間軸は常に横
        self.live2d_thumb.setStyleSheet(
            "QLabel { background: #ffffff; border: 1px solid #555;"
            " color: #888; font-size: 10px; }")
        self.live2d_thumb.setText(tr("live3d_waiting"))
        l3_cols.addWidget(self.live2d_thumb, 2)

        # 右カラム: 3D GIF (上) + マップサムネイル3枚 (下)
        right_col = QVBoxLayout()
        self.live3d_label = QLabel(tr("live3d_waiting"))
        self.live3d_label.setAlignment(Qt.AlignCenter)
        self.live3d_label.setMinimumSize(320, 240)
        self.live3d_label.setStyleSheet(
            "QLabel { background: #222; color: #888; border: 1px solid #555; }")
        right_col.addWidget(self.live3d_label, 1)

        # 適用済みマップ 3 枚のサムネイル (3D アニメの再生位置を赤ラインで表示)
        self._map_thumbs = {}
        thumb_row = QHBoxLayout()
        thumb_row.setSpacing(6)
        for t, cap in (("space", "Space"), ("time", "Time"), ("rate", "Rate")):
            col = QVBoxLayout()
            col.setSpacing(1)
            cap_lbl = QLabel(cap)
            cap_lbl.setStyleSheet("color: gray; font-size: 10px;")
            cap_lbl.setAlignment(Qt.AlignCenter)
            col.addWidget(cap_lbl)
            th = MapThumb(cap)
            self._map_thumbs[t] = th
            col.addWidget(th)
            thumb_row.addLayout(col, 1)
        right_col.addLayout(thumb_row)
        l3_cols.addLayout(right_col, 3)

        l3_outer.addLayout(l3_cols)
        self.live3d_status = QLabel("")
        self.live3d_status.setStyleSheet("color: gray; font-size: 11px;")
        l3_outer.addWidget(self.live3d_status)
        l3_outer.addStretch(1)
        self.live3d_group.setVisible(False)      # Initialize 後に表示

        # ===== マニューバ プレビュー (Time+Space or Rate+Space 揃った時点で確認) =====
        self.preview_group = QGroupBox()
        self._reg(lambda: self.preview_group.setTitle(tr("grp_maneuver_preview")))
        prev_v = QVBoxLayout(self.preview_group)
        prev_hint = self._trlabel("preview_hint")
        prev_hint.setStyleSheet("color: gray; font-size: 11px;")
        prev_hint.setWordWrap(True)
        prev_v.addWidget(prev_hint)

        # ※ 適用方法 (time to data / rate to data) の選択はタブ2「画像」下部に
        #    移動した (self.apply_mode_group)。ここには置かない。

        # Settings row: anim frame count + dpi for quick preview
        pset_layout = QHBoxLayout()
        pset_layout.addWidget(self._trlabel("lbl_3d_frames"))
        self.preview_frames_spin = QSpinBox()
        self.preview_frames_spin.setRange(5, 200)
        self.preview_frames_spin.setValue(20)
        pset_layout.addWidget(self.preview_frames_spin)
        pset_layout.addWidget(self._trlabel("lbl_dpi"))
        self.preview_dpi_spin = QSpinBox()
        self.preview_dpi_spin.setRange(40, 300)
        self.preview_dpi_spin.setValue(80)
        pset_layout.addWidget(self.preview_dpi_spin)
        pset_layout.addStretch()
        prev_v.addLayout(pset_layout)

        self.preview_btn = QPushButton()
        self._reg(lambda: self.preview_btn.setText(tr("btn_gen_preview")))
        self.preview_btn.clicked.connect(self.start_maneuver_preview)
        prev_v.addWidget(self.preview_btn)

        self.preview_status_label = QLabel(tr("status_idle"))
        self.preview_status_label.setStyleSheet("color: gray; font-size: 11px;")
        prev_v.addWidget(self.preview_status_label)

        # 生成中の進捗バー (% 表示付き、実行中のみ表示)
        self.preview_progress = QProgressBar()
        self.preview_progress.setRange(0, 100)
        self.preview_progress.setValue(0)
        self.preview_progress.setTextVisible(True)
        self.preview_progress.setVisible(False)
        prev_v.addWidget(self.preview_progress)

        prev_v.addWidget(self._trlabel("lbl_2d_plot"))
        self.preview_2dplot_label = QLabel(tr("preview_after_gen"))
        self.preview_2dplot_label.setAlignment(Qt.AlignCenter)
        self.preview_2dplot_label.setMinimumSize(400, 250)
        # 2D プロットは透過 PNG (黒文字/黒線) なので背景を白にして視認性を確保
        self.preview_2dplot_label.setStyleSheet(
            "QLabel { background: #ffffff; color: #888; border: 1px solid #555; }"
        )
        prev_v.addWidget(self.preview_2dplot_label)

        prev_v.addWidget(self._trlabel("lbl_3d_anim"))
        self.preview_3d_label = QLabel(tr("preview_after_gen"))
        self.preview_3d_label.setAlignment(Qt.AlignCenter)
        self.preview_3d_label.setMinimumSize(400, 300)
        self.preview_3d_label.setStyleSheet(
            "QLabel { background: #222; color: #888; border: 1px solid #555; }"
        )
        prev_v.addWidget(self.preview_3d_label)

        self.preview_group.setVisible(False)
        self._preview_movie = None  # QMovie の生存維持用

        # --- Mode info (選択そのものはタブ2の apply_mode_group で行う) ---
        self.apply_mode_info = QLabel("")
        self.apply_mode_info.setStyleSheet("color: #555; font-size: 12px;")
        self.apply_mode_info.setWordWrap(True)
        self._reg(self._update_apply_mode_info)

        # --- Animation toggle ---
        anim_label = self._trlabel("lbl_anim_settings")
        self.anim_toggle = QCheckBox()
        self._reg(lambda: self.anim_toggle.setText(tr("chk_enable_anim")))
        self.anim_toggle.stateChanged.connect(self.on_anim_toggle_changed)

        self.anim_settings_container = QFrame()
        anim_settings_layout = QVBoxLayout(self.anim_settings_container)
        duration_layout = QHBoxLayout()
        duration_label = self._trlabel("lbl_anim_duration")
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 120)
        self.duration_spin.setValue(10)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_spin)
        anim_settings_layout.addLayout(duration_layout)
        self.anim_settings_container.setVisible(False)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton()
        self._reg(lambda: self.start_btn.setText(tr("btn_start_render")))
        self.start_btn.clicked.connect(self.start_rendering)
        self.animonly_btn = QPushButton()
        self._reg(lambda: self.animonly_btn.setText(tr("btn_anim_only")))
        self.animonly_btn.clicked.connect(self.start_animation_only)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.animonly_btn)

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)

        # ===== タブ構造でレイアウト組み立て =====
        self.tabs = QTabWidget()
        tabs = self.tabs

        # --- Tab 1: 入力 + 画像 (Setup & Images 統合) ---
        # 上段 2 カラム: 左 = 入力(Setup) + 適用方法 + 共通サイズ設定 /
        #               右 = 軌道プロット ライブプレビュー (2D|3D)
        # 下段: Space / Time / Rate の 3 カラム。
        # 1 画面で入力から画像編集まで全状況を見ながら操作できる。
        setup_group = QGroupBox()
        self._reg(lambda b=setup_group: b.setTitle(tr("grp_setup")))
        sg_l = QVBoxLayout(setup_group)
        sg_l.addLayout(self.lang_row)
        for w in [self.video_btn, self.video_label,
                  self.slit_toggle, self.slit_label,
                  self.init_btn, self.info_label]:
            sg_l.addWidget(w)

        t2 = QWidget(); t2_l = QVBoxLayout(t2)
        top_row = QHBoxLayout()
        top_left = QVBoxLayout()
        top_left.addWidget(setup_group)
        top_left.addWidget(self.apply_mode_group)
        top_left.addWidget(self.gen_group)
        top_left.addStretch()
        top_row.addLayout(top_left, 1)
        top_row.addWidget(self.live3d_group, 3)   # シミュレーション側を幅 3/4 に
        t2_l.addLayout(top_row)

        cols = QHBoxLayout()
        cols.setSpacing(8)
        # パス表示 (Selected: …) は冗長なため列に含めない (サムネイルで確認できる)。
        # パラメータ枠はコンパクト化 (小さめフォント + 詰めたマージン)。
        for type_name, title_key, param_frame, gen_frame in [
            ('space', "grp_space_image",
             self.space_param_frame, self.space_gen_frame),
            ('time', "grp_time_image",
             self.time_param_frame, self.time_gen_frame),
            ('rate', "grp_rate_image",
             self.rate_param_frame, self.rate_gen_frame),
        ]:
            param_frame.setStyleSheet(
                "QLabel { font-size: 11px; }"
                " QSpinBox, QDoubleSpinBox { font-size: 11px; }")
            if param_frame.layout() is not None:
                param_frame.layout().setContentsMargins(2, 0, 2, 0)
                param_frame.layout().setSpacing(2)
            box = QGroupBox()
            self._reg(lambda b=box, k=title_key: b.setTitle(tr(k)))
            bv = QVBoxLayout(box)
            bv.addWidget(param_frame)
            bv.addWidget(gen_frame)
            bv.addStretch()
            cols.addWidget(box, 1)
        t2_l.addLayout(cols)


        t2_l.addStretch()
        tabs.addTab(self._wrap_scroll(t2), tr("tab_main"))

        # --- Tab 2: リアルタイムプレビュー (Preview) ---
        # 映像ビューをタブ領域いっぱいに拡大させるため、ストレッチ係数 1 で
        # 追加し余白 stretch は置かない (スクロールにも包まない)。
        t3 = QWidget(); t3_l = QVBoxLayout(t3)
        if _HAS_RT_PREVIEW:
            self.rt_group = QGroupBox()
            self._reg(lambda: self.rt_group.setTitle(tr("grp_realtime")))
            rt_v = QVBoxLayout(self.rt_group)
            self.rt_preview = RealtimePreviewWidget(lang=LANG)
            rt_v.addWidget(self.rt_preview)
            t3_l.addWidget(self.rt_group, 1)
        else:
            self.rt_preview = None
            t3_l.addStretch()
        # 2D/3D 軌道プロットは「1. 入力・画像」タブのライブプレビューへ完全移行。
        # このタブは動画のリアルタイムプレビュー専用 (preview_group は非表示のまま
        # 保持し、内部ロジック互換のためウィジェットだけ残す)。
        tabs.addTab(t3, tr("tab_preview"))
        tabs.currentChanged.connect(self._on_tab_changed)

        # --- Tab 3: 出力 (Render) ---
        t4 = QWidget(); t4_l = QVBoxLayout(t4)
        for w in [self.apply_mode_info,
                  anim_label, self.anim_toggle,
                  self.anim_settings_container]:
            t4_l.addWidget(w)
        t4_l.addLayout(btn_layout)

        # レンダリング結果プレビュー (完了後に動画を上下に並べて再生)
        self.result_group = QGroupBox()
        self._reg(lambda: self.result_group.setTitle(tr("grp_rendered_preview")))
        result_v = QVBoxLayout(self.result_group)
        self.rendered_preview = VideoPreview(tr("rendered_video_title"))
        self.anim_preview = VideoPreview(tr("anim_title"))
        self._reg(lambda: self.rendered_preview.set_base_title(tr("rendered_video_title")))
        self._reg(lambda: self.anim_preview.set_base_title(tr("anim_title")))
        result_v.addWidget(self.rendered_preview)
        result_v.addWidget(self.anim_preview)
        self.result_group.setVisible(False)
        t4_l.addWidget(self.result_group)

        t4_l.addStretch()
        tabs.addTab(self._wrap_scroll(t4), tr("tab_render"))

        # タブ見出しの再翻訳を登録
        self._reg(lambda: (
            self.tabs.setTabText(0, tr("tab_main")),
            self.tabs.setTabText(1, tr("tab_preview")),
            self.tabs.setTabText(2, tr("tab_render")),
        ))

        # ===== ログ (メインの入力・画像ページでは非表示、他タブで表示) =====
        log_label = self._trlabel("lbl_log")
        log_label.setStyleSheet("color: gray; font-size: 11px; margin-top: 4px;")
        self.log_window.setMinimumHeight(80)
        self.log_window.setMaximumHeight(160)

        # Splitter で「タブ」と「ログ」のサイズを可変に
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(tabs)
        self.log_box = QWidget()
        log_box_l = QVBoxLayout(self.log_box)
        log_box_l.setContentsMargins(0, 0, 0, 0)
        log_box_l.addWidget(log_label)
        log_box_l.addWidget(self.log_window)
        splitter.addWidget(self.log_box)
        splitter.setStretchFactor(0, 5)  # tabs 側を広く
        splitter.setStretchFactor(1, 1)
        self.log_box.setVisible(False)   # 起動時はメインページ (index 0)

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
            for b in [self.init_btn,
                      self.anim_toggle, self.start_btn,
                      self.animonly_btn, *gen_btns]:
                b.setEnabled(False)
        elif stage == "video_selected":
            self.init_btn.setEnabled(True)
        elif stage == "initialized":
            for b in gen_btns:
                b.setEnabled(True)
            self.gen_group.setVisible(True)
            self.apply_mode_group.setVisible(True)
            self.live3d_group.setVisible(True)
            self.preview_btn.setEnabled(False)  # 適用方法+画像が揃うまで無効
            self._apply_video_defaults()
            self._auto_apply_normal_maps()
            self._update_preview_btn_state()
        elif stage == "rendered":
            self.animonly_btn.setEnabled(True)
            self.anim_settings_container.setVisible(True)
            self.log("Animation-only rendering is now available.")
        self._update_tab_gating()

    # --- Tab gating ---
    # 段階制ワークフロー:
    #   タブ1 (入力)     : 常に有効
    #   タブ2 (画像)     : Initialize 完了で解放
    #   タブ3/4 (プレビュー/出力):
    #       適用方法 (time to data / rate to data) が選択され、かつ
    #       Space + (Time または Rate) 画像が揃った時点で解放
    def _selected_apply_mode(self):
        """タブ2の適用方法 combo の選択。未選択なら None。"""
        txt = self.preview_mode_select.currentText() if hasattr(self, "preview_mode_select") else ""
        return txt if txt in ("time to data", "rate to data") else None

    def _pipeline_ready(self):
        """プレビュー/出力に進める状態か (初期化 + 適用方法 + 必要画像)。"""
        if not self.dm:
            return False
        mode = self._selected_apply_mode()
        if mode is None or not self.space_img_path:
            return False
        if mode == "time to data":
            return bool(self.time_img_path)
        return bool(self.rate_img_path)

    def _update_tab_gating(self):
        if not hasattr(self, "tabs"):
            return
        ready = self._pipeline_ready()
        self.tabs.setTabEnabled(1, ready)   # プレビュー
        self.tabs.setTabEnabled(2, ready)   # 出力
        # 出力操作も同じ条件でゲート (レンダリング可能条件と一致)
        self.start_btn.setEnabled(ready)
        self.anim_toggle.setEnabled(ready)
        # 現在表示中のタブが無効化されたら、有効な直近のタブへ戻す
        cur = self.tabs.currentIndex()
        if not self.tabs.isTabEnabled(cur):
            for i in range(cur, -1, -1):
                if self.tabs.isTabEnabled(i):
                    self.tabs.setCurrentIndex(i)
                    break

    def _update_apply_mode_info(self):
        """タブ4上部の「適用方法」表示を更新 (選択はタブ2で行う)。"""
        if not hasattr(self, "apply_mode_info"):
            return
        m = self._selected_apply_mode() or "—"
        self.apply_mode_info.setText(tr("lbl_apply_mode_info", m=m))

    def on_apply_mode_changed(self, *_):
        mode = self._selected_apply_mode()
        if mode:
            self.log(f"Apply mode selected: {mode}")
            # 選択された基準画像から対になるマップを即導出
            self._sync_derived_maps()
        self._update_apply_mode_info()
        self._update_preview_btn_state()
        self._mark_preview_stale()
        self._update_tab_gating()

    def _apply_video_defaults(self):
        """drawManeuver 初期化直後に、スピンボックスの既定値を映像情報から賢く設定する"""
        if not self.dm:
            return
        # マップサムネイルの時間軸向き (縦スリット=縦 / 横スリット=横)
        sd = int(getattr(self.dm, "scan_direction", 1))
        for th in getattr(self, "_map_thumbs", {}).values():
            th.set_time_vertical(sd == 1)
        # レイヤーの「通常再生」パターンをスリット方向に合わせて更新
        # (ラベルの（通常再生）表記も sd 依存なので combo を再構築し、
        #  ベースレイヤーは sd に応じた通常再生パターンへリセットする)
        for sec, g in self._section_gens.items():
            for li, lw in enumerate(g.get('layers', [])):
                lw.sd = sd
                lw.retranslate()
                if li == 0:
                    pid = normal_pattern_for(sec, sd)
                    if pid in lw.pattern_ids:
                        lw.pattern.blockSignals(True)
                        lw.pattern.setCurrentIndex(lw.pattern_ids.index(pid))
                        lw.pattern.blockSignals(False)
        # 共通サイズ
        self.gen_scan_size.setValue(int(self.dm.scan_nums))
        self.gen_time_size.setValue(900)
        # 出力FPS の既定は 30 固定 (900 frames ÷ 30 fps = 30 秒)
        self.gen_out_fps.setCurrentIndex(2)   # 30
        # 各 type の既定パラメータ
        self.space_set_value.setValue(int(self.dm.scan_nums))
        # Time 画像の既定レンジ: vmin=0, vmax=出力FPS×時間方向サイズ
        self.time_vmin_spin.setValue(0)
        self.time_vmax_spin.setValue(self._default_time_vmax())
        self._last_time_default = (0, self._default_time_vmax())
        self.rate_baseline_spin.setValue(1.0)
        self.rate_maxdev_spin.setValue(0.5)
        # 各セクション全レイヤーの波形周期既定値 = 時間方向サイズ (= 全体で 1 周期)
        for t in self._section_gens:
            for lw in self._section_gens[t].get('layers', []):
                lw.wave_period.setValue(120)
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
        """共通サイズと、生成されるファイル形状/出力尺を表示"""
        if not self.dm:
            self.gen_hint.setText("")
            return
        sd = int(getattr(self.dm, "scan_direction", 1))
        scan_size = self.gen_scan_size.value()
        time_size = self.gen_time_size.value()
        out_fps = max(1, self._out_fps())
        if sd == 1:
            file_dim = f"{scan_size}(W) × {time_size}(H)  → Width=scan, Height=time"
        else:
            file_dim = f"{time_size}(W) × {scan_size}(H)  → Width=time, Height=scan"
        dur = time_size / out_fps
        self.gen_hint.setText(
            tr("gen_hint", dim=file_dim) + "\n" +
            tr("gen_hint_dur", dur=f"{dur:.2f}", ts=time_size, fps=out_fps))

    def _out_fps(self):
        """出力FPS combo の現在値 (int)。"""
        v = self.gen_out_fps.currentData()
        return int(v) if v else 30

    def _auto_apply_normal_maps(self):
        """Initialize 直後、通常再生グラデーションを Space/Time/Rate に自動適用する。

        以後レイヤーを編集してもプレビューが変わるだけで、Generate & Apply を
        押すまで適用画像は上書きされない (適用されれば 3D/2D ライブプロットも
        自動更新される)。
        """
        for t in ("space", "time", "rate"):
            try:
                self.generate_sample_image_action(t)
            except Exception as e:
                self.log(f"[WARN] auto-apply {t}: {e}")

    def _default_time_vmax(self):
        """Time 画像 vmax の既定値 = 出力フレーム数 (時間方向サイズ)。

        通常再生では出力1フレーム = 入力1フレームを参照するため、時間マップの
        レンジは出力フレーム数そのもの (= 出力FPS × 出力秒数)。
        入力映像の総フレーム数を超える場合は総フレーム数に制限する。
        (旧実装は出力FPS×フレーム数で一桁大きかった)
        """
        v = self.gen_time_size.value()
        if self.dm is not None:
            v = min(v, int(self.dm.count))
        return v

    def _maybe_update_time_defaults(self, *_):
        """時間方向サイズ / 出力FPS の変更を Time の vmin/vmax に即座に反映する。"""
        if not self.dm:
            return
        self.time_vmin_spin.setValue(0)
        self.time_vmax_spin.setValue(self._default_time_vmax())

    def _sync_rt_timeline(self, *_):
        """時間方向サイズ / 出力FPS をリアルタイムプレビューのタイムラインへ反映。
        rate の累積積分が fps/時間方向サイズに依存するためマップも再構築する。"""
        if getattr(self, "rt_preview", None):
            self.rt_preview.set_params(time_size=self.gen_time_size.value(),
                                       out_fps=self._out_fps())
            self.rt_preview.refresh_maps()

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
            if getattr(self, "rt_preview", None):
                self.rt_preview.set_video(self.videopath)
                # スリット方向と入力実FPS をプレビューに同期
                self.rt_preview.set_params(sd=int(self.dm.scan_direction),
                                           rec_fps=float(self.dm.recfps))
                self._sync_rt_timeline()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.update_ui_state("video_selected")

    def _build_section_gen(self, type_name):
        """セクション ({type_name}=space/time/rate) 専用のジェネレータパネルを生成。

        レイヤースタック方式: LayerWidget を何枚でも追加でき、上から順に
        合成モード + 不透明度で合成した結果がプレビュー/生成される。

        widget は self._section_gens[type_name] に dict として保存
        (layers / layers_box / add_btn / preview_label / generate_btn)。
        Returns: 構築済の QFrame (Tab2 のセクション内に addWidget する用)
        """
        g = {}
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        v = QVBoxLayout(frame)

        head = self._trlabel("gen_header", t=type_name)
        head.setStyleSheet("font-weight: bold; color: #555; margin-top: 4px;")
        v.addWidget(head)

        # レイヤースタック
        g['layers'] = []
        g['layers_box'] = QVBoxLayout()
        g['layers_box'].setSpacing(4)
        v.addLayout(g['layers_box'])

        g['add_btn'] = QPushButton()
        self._reg(lambda b=g['add_btn']: b.setText(tr("btn_add_layer")))
        g['add_btn'].clicked.connect(lambda *_, t=type_name: self._add_layer(t))
        v.addWidget(g['add_btn'])

        # Preview (合成結果。画像読み込み後はその画像を表示)
        g['preview_label'] = QLabel(tr("preview_after_init"))
        g['preview_label'].setAlignment(Qt.AlignCenter)
        g['preview_label'].setMinimumSize(320, 180)
        g['preview_label'].setStyleSheet(
            "QLabel { background: #222; color: #888; border: 1px solid #555; }"
        )
        v.addWidget(g['preview_label'])

        # Generate ボタン (Auto Generate を統合)
        g['generate_btn'] = QPushButton()
        self._reg(lambda b=g['generate_btn'], t=type_name:
                  b.setText(tr("btn_generate_apply", t=t.capitalize())))
        g['generate_btn'].clicked.connect(lambda *_, t=type_name: self.generate_sample_image_action(t))
        g['generate_btn'].setEnabled(False)  # Initialize 前は無効

        self._section_gens[type_name] = g
        self._add_layer(type_name)   # ベースレイヤー
        v.addWidget(g['generate_btn'])
        return frame

    def _current_sd(self):
        """現在のスリット方向 (dm 初期化前はチェックボックスから)。"""
        if self.dm is not None:
            return int(getattr(self.dm, "scan_direction", 1))
        return 1 if self.slit_toggle.isChecked() else 0

    def _add_layer(self, type_name):
        """セクションにレイヤーを 1 枚追加する。"""
        g = self._section_gens[type_name]
        lw = LayerWidget(type_name, len(g['layers']), sd=self._current_sd())
        lw.changed.connect(lambda t=type_name: self._update_section_preview(t))
        lw.remove_requested.connect(lambda w, t=type_name: self._remove_layer(t, w))
        g['layers'].append(lw)
        g['layers_box'].addWidget(lw)
        self._update_section_preview(type_name)

    def _remove_layer(self, type_name, widget):
        g = self._section_gens[type_name]
        if widget not in g['layers'] or len(g['layers']) <= 1:
            return
        g['layers'].remove(widget)
        g['layers_box'].removeWidget(widget)
        widget.deleteLater()
        for i, lw in enumerate(g['layers']):
            lw.set_index(i)
        self._update_section_preview(type_name)

    def _make_preview_pixmap_for(self, type_name, max_w=240, max_h=160):
        """セクション {type_name} の現在のレイヤースタックを合成してプレビュー QPixmap を生成。"""
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

        layers = [lw.params() for lw in g['layers']]
        img16 = composite_layers(ph, pw, layers, scale=scale)
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
        """適用方法 (タブ2) を "time" / "rate" で返す。未選択なら None。"""
        m = self._selected_apply_mode()
        if m == "rate to data":
            return "rate"
        if m == "time to data":
            return "time"
        return None

    def _can_preview_mode(self):
        """選択された適用方法でプレビュー可能なら "time"/"rate"、不足があれば None。"""
        if not self.dm or not self.space_img_path:
            return None
        mode = self._selected_preview_mode()
        if mode == "time":
            return "time" if self.time_img_path else None
        if mode == "rate":
            return "rate" if self.rate_img_path else None
        return None

    def _update_preview_btn_state(self):
        """選択された適用方法と画像セット状態に応じてボタンの有効/無効を切り替え"""
        if not hasattr(self, "preview_btn"):
            return
        mode = self._can_preview_mode()
        self.preview_btn.setEnabled(mode is not None)
        if self._selected_preview_mode() is None:
            self.preview_status_label.setText(tr("status_need_mode"))
        elif not self.dm or not self.space_img_path:
            self.preview_status_label.setText(tr("status_need_space"))
        elif mode is not None:
            self.preview_status_label.setText(tr("status_ready", m=mode))
        else:
            need = "Time" if self._selected_preview_mode() == "time" else "Rate"
            self.preview_status_label.setText(tr("status_need_img", need=need))
        # リアルタイムプレビューのモードも同期 (選択済みのときのみ)
        if getattr(self, "rt_preview", None) and self._selected_preview_mode():
            self.rt_preview.set_params(mode=self._selected_preview_mode())

    def start_maneuver_preview(self):
        mode = self._can_preview_mode()
        if mode is None:
            QMessageBox.warning(self, "Error",
                                "Space + (Time または Rate) 画像が必要です")
            return
        # ライブ3D生成が走っていたら完了を待つ (dm 共有のため並走させない)
        if self._live3d_busy and self._live3d_worker is not None:
            self._live3d_worker.wait(8000)
        self.preview_btn.setEnabled(False)
        self.preview_status_label.setText("Status: running…")
        # 生成中であることをプロット領域自体にも表示 (古い表示は消す)
        if self._preview_movie is not None:
            try:
                self._preview_movie.stop()
            except Exception:
                pass
            self.preview_3d_label.setMovie(None)
            self._preview_movie = None
        self.preview_2dplot_label.setPixmap(QPixmap())
        self.preview_2dplot_label.setText(tr("processing_wait"))
        self.preview_3d_label.setText(tr("processing_wait"))
        self.preview_progress.setValue(0)
        self.preview_progress.setVisible(True)
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
        self._preview_worker.percent_signal.connect(self._on_preview_percent)
        self._preview_worker.done_signal.connect(self._on_preview_done)
        self._preview_worker.start()

    def _on_preview_progress(self, msg):
        self.preview_status_label.setText(f"Status: {msg}")
        self.log(f"[preview] {msg}")

    def _on_preview_percent(self, pct):
        self.preview_progress.setValue(int(pct))

    def _on_preview_done(self, success, plot2d, gif):
        self.preview_btn.setEnabled(True)
        self.preview_progress.setVisible(False)
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
        # 3D軌道ライブプレビューは編集のたびにデバウンス再生成
        self._schedule_live3d()
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

    # --- 3D軌道 ライブプレビュー (タブ2・自動更新) ---
    def _schedule_live3d(self):
        """編集イベントを800msデバウンスして _run_live3d を起動する。"""
        if not self.dm or not getattr(self, "live3d_group", None):
            return
        self._live3d_timer.start()      # 連続編集中はタイマーが巻き戻る

    def _live3d_prereq_mode(self):
        """ライブ3D生成が可能なら "time"/"rate"、不可なら None。"""
        return self._can_preview_mode()

    def _run_live3d(self):
        if self._live3d_busy:
            self._live3d_pending = True
            return
        # 重い処理 (レンダリング / 手動プレビュー) の実行中は後回し
        if (self.worker is not None and self.worker.isRunning()) or \
           (getattr(self, "_preview_worker", None) is not None
                and self._preview_worker.isRunning()):
            self._live3d_timer.start(1500)
            return
        mode = self._live3d_prereq_mode()
        if mode is None:
            self.live3d_status.setText("")
            if self._live3d_movie is None:
                self.live3d_label.setText(tr("live3d_waiting"))
            return
        self._live3d_busy = True
        self.live3d_status.setText(tr("live3d_updating"))
        self._live3d_worker = ManeuverPreviewWorker(
            self.dm, mode,
            self.space_img_path, self.time_img_path, self.rate_img_path,
            self.space_set_value.value(),
            self.time_vmin_spin.value(), self.time_vmax_spin.value(),
            self.rate_maxdev_spin.value(),
            self.rate_baseline_spin.value(),
            self.rate_startpoint_spin.value(),
            anim_frames=10, anim_fps=8, anim_dpi=55,
            skip_2d=False,   # 2D プロットもライブ表示する (タブ2へ完全移行)
        )
        self._live3d_worker.done_signal.connect(self._on_live3d_done)
        self._live3d_worker.start()

    def _on_live3d_done(self, success, plot2d, gif):
        self._live3d_busy = False
        # 2D プロット (左カラム・赤ライン付きサムネイル)
        if success and plot2d and os.path.exists(plot2d):
            self.live2d_thumb.set_map(plot2d)
        if success and gif and os.path.exists(gif):
            if self._live3d_movie is not None:
                try:
                    self._live3d_movie.stop()
                except Exception:
                    pass
                self.live3d_label.setMovie(None)
            movie = QMovie(gif)
            movie.setCacheMode(QMovie.CacheNone)
            if movie.isValid():
                native = QImageReader(gif).size()
                box = self.live3d_label.size()
                if native.width() > 0 and native.height() > 0:
                    scale = min(box.width() / native.width(),
                                box.height() / native.height())
                    movie.setScaledSize(QSize(
                        max(1, int(native.width() * scale)),
                        max(1, int(native.height() * scale))))
                self.live3d_label.setMovie(movie)
                movie.frameChanged.connect(self._on_live3d_frame)
                movie.start()
                self._live3d_movie = movie
            self.live3d_status.setText("")
        else:
            self.live3d_status.setText("")
        # 実行中に編集が入っていたら追いかけ再生成
        if self._live3d_pending:
            self._live3d_pending = False
            self._schedule_live3d()

    def _on_live3d_frame(self, frame_idx):
        """3D アニメの再生位置をマップサムネイルの赤ラインに同期させる。"""
        movie = self._live3d_movie
        if movie is None:
            return
        n = max(1, movie.frameCount())
        frac = (frame_idx + 0.5) / n
        for th in getattr(self, "_map_thumbs", {}).values():
            th.set_playhead(frac)
        # 2D プロットにも赤ラインを左→右へスライド表示
        self.live2d_thumb.set_playhead(frac)

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
        scan_size = self.gen_scan_size.value()
        time_size = self.gen_time_size.value()
        sd = int(getattr(self.dm, "scan_direction", 1))
        out_dir = os.path.dirname(self.videopath) or "."

        # Slit 方向に応じてファイル形状を決定
        if sd == 1:
            h_pix, w_pix = int(time_size), int(scan_size)   # (time, scan)
        else:
            h_pix, w_pix = int(scan_size), int(time_size)   # (scan, time) — .T される

        layers = [lw.params() for lw in g['layers']]
        try:
            img16 = composite_layers(h_pix, w_pix, layers, scale=1.0)
            fname = sample_filename(
                type_name,
                space_range=self.space_set_value.value(),
                time_vmin=self.time_vmin_spin.value(),
                time_vmax=self.time_vmax_spin.value(),
                rate_maxdev=self.rate_maxdev_spin.value(),
                scan_size=scan_size,
            )
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, img16)
        except Exception as e:
            QMessageBox.critical(self, "Generate Error", str(e))
            self.log(f"[ERROR] generate composite: {e}")
            return

        pats = "+".join(p["pattern"] for p in layers)
        self.log(f"Sample {type_name} ({len(layers)} layer(s): {pats}): {out_path}")
        setattr(self, f"{type_name}_img_path", out_path)
        self._wire_loaded_image(type_name, out_path)

        # 適用方法に応じて相互に導出:
        #   time to data で time を適用 → rate を time から自動生成
        #   rate to data で rate を適用 → time を rate から自動生成
        mode = self._selected_apply_mode()
        if (type_name == "time" and mode == "time to data") or \
           (type_name == "rate" and mode == "rate to data"):
            self._sync_derived_maps()

    # --- time ⇄ rate の相互導出 ---
    def _load_map_datacoords(self, path):
        """マップ PNG をデータ座標系 (time行 × scan列) の float 0..1 で返す。"""
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            return None
        if m.ndim == 3:
            m = m[..., 0]
        mx = 65535.0 if m.dtype == np.uint16 else 255.0
        m = m.astype(np.float64) / mx
        if self._current_sd() == 0:
            m = m.T
        return m

    def _save_map_datacoords(self, arr01, fname):
        """データ座標系 0..1 配列を 16bit PNG としてファイル規約の向きで保存。"""
        img16 = (np.clip(arr01, 0.0, 1.0) * 65535.0).astype(np.uint16)
        if self._current_sd() == 0:
            img16 = img16.T
        out_path = os.path.join(os.path.dirname(self.videopath) or ".", fname)
        cv2.imwrite(out_path, img16)
        return out_path

    def _sync_derived_maps(self):
        """適用方法の基準画像から、対になるマップを書き出しと同じ式で導出する。

        time to data: rate = ΔTime / (recfps/outfps)  (時間マップの微分)
        rate to data: time = Σ rate × (recfps/outfps) (レートマップの累積積分)
        導出結果は sample_*.png として保存し、通常の適用フローに乗せる
        (サムネイル/ライブプロット/RTプレビューも自動更新される)。
        """
        if getattr(self, "_syncing_maps", False) or not self.dm:
            return
        mode = self._selected_apply_mode()
        if mode is None:
            return
        self._syncing_maps = True
        try:
            frame_step = float(self.dm.recfps) / max(1, self._out_fps())
            if mode == "time to data" and self.time_img_path:
                t01 = self._load_map_datacoords(self.time_img_path)
                if t01 is None or t01.shape[0] < 2:
                    return
                vmin = self.time_vmin_spin.value()
                vmax = self.time_vmax_spin.value()
                T = vmin + t01 * (vmax - vmin)
                r = np.diff(T, axis=0) / frame_step
                r = np.vstack([r, r[-1:]])
                rmin, rmax = float(r.min()), float(r.max())
                baseline = round((rmax + rmin) / 2.0, 3)
                maxdev = max(round((rmax - rmin) / 2.0, 3), 0.001)
                r01 = (r - (baseline - maxdev)) / (2.0 * maxdev)
                path = self._save_map_datacoords(r01, f"sample_rate_{maxdev}.png")
                self.rate_baseline_spin.setValue(baseline)
                self.rate_img_path = path
                self._wire_loaded_image("rate", path)
                self.log(f"[sync] rate を time から自動生成 "
                         f"(baseline={baseline}, max_dev={maxdev})")
            elif mode == "rate to data" and self.rate_img_path:
                r01 = self._load_map_datacoords(self.rate_img_path)
                if r01 is None or r01.shape[0] < 1:
                    return
                baseline = self.rate_baseline_spin.value()
                maxdev = self.rate_maxdev_spin.value()
                rates = baseline + (r01 - 0.5) * 2.0 * maxdev
                cum = np.cumsum(rates, axis=0) * frame_step
                cum = np.vstack([np.zeros((1, cum.shape[1])), cum[:-1]])
                vmin_i = int(np.floor(cum.min()))
                vmax_i = int(np.ceil(cum.max()))
                span = max(1, vmax_i - vmin_i)
                t01 = (cum - vmin_i) / span
                path = self._save_map_datacoords(
                    t01, f"sample_time_{vmin_i}-{vmax_i}.png")
                self.time_img_path = path
                self._wire_loaded_image("time", path)
                self.log(f"[sync] time を rate から自動生成 (vrange {vmin_i}-{vmax_i})")
        except Exception as e:
            self.log(f"[WARN] map sync failed: {e}")
        finally:
            self._syncing_maps = False

    def _wire_loaded_image(self, img_type, path):
        """select_image() の "画像情報表示 + パラメータ抽出 + プレビュー" 共通処理"""
        getattr(self, f"{img_type}_label").setText(f"Selected: {path}")
        # ライブプロット下の適用済みマップサムネイルを更新
        if img_type in getattr(self, "_map_thumbs", {}):
            self._map_thumbs[img_type].set_map(path)
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

        # マニューバプレビューボタン / タブゲートの更新 + stale マーク (画像が変わったため)
        self._update_preview_btn_state()
        self._mark_preview_stale()
        self._update_tab_gating()

        # リアルタイムプレビューにも最新のマップ/パラメータを反映 (常駐済みなら即更新)
        if getattr(self, "rt_preview", None):
            self.rt_preview.set_maps(self.space_img_path, self.time_img_path,
                                     self.rate_img_path)
            self.rt_preview.set_params(
                mode=self._selected_preview_mode(),
                space_set=self.space_set_value.value(),
                vmin=self.time_vmin_spin.value(), vmax=self.time_vmax_spin.value(),
                baseline=self.rate_baseline_spin.value(),
                maxdev=self.rate_maxdev_spin.value())
            self.rt_preview.refresh_maps()

    def on_anim_toggle_changed(self, state):
        if state == Qt.Checked:
            self.anim_settings_container.setVisible(True)
        else:
            self.anim_settings_container.setVisible(False)
        self._update_tab_gating()

    def update_slit_label(self):
        if self.slit_toggle.isChecked():
            self.slit_label.setText(tr("slit_v"))
        else:
            self.slit_label.setText(tr("slit_h"))

    def _on_tab_changed(self, idx):
        """プレビュータブ (index 1) を表示中だけリアルタイムプレビューを再生。
        ログはレンダリング進捗を見る出力タブ (index 2) のみ表示
        (入力・画像/プレビューでは映像領域を最大化するため非表示)。"""
        if getattr(self, "log_box", None):
            self.log_box.setVisible(idx == 2)
        rt = getattr(self, "rt_preview", None)
        if not rt:
            return
        if idx == 1:
            rt.start()
        else:
            rt.stop()

    def start_rendering(self):
        mode = self._selected_apply_mode()
        if mode is None:
            QMessageBox.warning(self, "Error",
                                "適用方法を「1. 入力・画像」タブで選択してください。")
            return
        # ライブ3D生成が走っていたら完了を待つ (dm 共有のため並走させない)
        if self._live3d_busy and self._live3d_worker is not None:
            self._live3d_worker.wait(8000)
        animout = self.anim_toggle.isChecked()
        duration = self.duration_spin.value()

        # 出力FPS を drawManeuver に反映 (最終尺 = 時間方向サイズ ÷ 出力FPS)
        try:
            self.dm.outfps = self._out_fps()
            self.log(f"Output FPS: {self.dm.outfps}")
        except Exception as e:
            self.log(f"[WARN] could not set outfps: {e}")

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
