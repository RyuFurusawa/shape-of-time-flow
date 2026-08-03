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
import shutil
import subprocess
from pathlib import Path

# Continue normal imports
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,
    QComboBox, QTextEdit, QCheckBox, QMessageBox, QSpinBox, QHBoxLayout,
    QFrame, QDoubleSpinBox, QGroupBox, QTabWidget, QScrollArea, QSplitter,
    QProgressBar, QSizePolicy, QSlider
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
    "tab_preview": {"ja": "2. プレビュー・出力 / Preview & Render",
                     "en": "2. Preview & Render"},
    "chk_video_only": {"ja": "映像ビューのみ表示",
                        "en": "Video view only"},
    "chk_audio": {"ja": "音声を適用", "en": "Apply audio"},
    "audio_mode_play": {"ja": "play (可変速再生)", "en": "play (varispeed)"},
    "audio_mode_grain": {"ja": "grain (グラニュラー)", "en": "grain (granular)"},
    "audio_out_on": {"ja": "音声出力: {m} × {v}声 (プレビューの設定を使用)",
                      "en": "Audio out: {m} × {v} voices (from preview settings)"},
    "audio_out_off": {"ja": "音声出力: なし (プレビューの「音声」で有効化)",
                       "en": "Audio out: none (enable via the preview's Audio)"},
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
    "tip_time_gen_disabled": {
        "ja": "rate to data 選択中は Time は Rate から自動導出されるため無効です",
        "en": "Disabled while 'rate to data' is selected — Time is derived from Rate"},
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
    # --- 入力映像の回転 (Initialize 前に ffmpeg で回転コピーを作る) ---
    "lbl_video_rotate": {"ja": "入力映像の回転:", "en": "Input video rotation:"},
    "vrot_none":  {"ja": "なし (0°)",            "en": "None (0°)"},
    "vrot_cw90":  {"ja": "右90° (時計回り)",      "en": "90° clockwise"},
    "vrot_180":   {"ja": "180°",                 "en": "180°"},
    "vrot_ccw90": {"ja": "左90° (反時計回り)",    "en": "90° counter-clockwise"},
    "vrot_hflip": {"ja": "左右反転",              "en": "Flip horizontal"},
    "vrot_vflip": {"ja": "上下反転",              "en": "Flip vertical"},
    "hint_video_rotate": {
        "ja": "(90°系はメタデータ書き換えのみ = 瞬時・無劣化。反転のみ再エンコード)",
        "en": "(90° variants remux metadata only — instant, lossless; flips re-encode)"},
    "vrot_no_ffmpeg": {"ja": "ffmpeg が見つかりません。映像の回転には ffmpeg が必要です。",
                        "en": "ffmpeg not found — video rotation requires ffmpeg."},
    "vrot_working":  {"ja": "映像を回転中… (ffmpeg)", "en": "Rotating video… (ffmpeg)"},
    "vrot_reuse":    {"ja": "回転済みの映像を再利用: {p}",
                       "en": "Reusing existing rotated video: {p}"},
    "vrot_failed":   {"ja": "映像の回転に失敗しました。ログを確認してください。",
                       "en": "Video rotation failed — see the log."},
    "vrot_reinit":   {"ja": "入力設定を変更しました → 「初期化」をもう一度押してください",
                       "en": "Input settings changed → press Initialize again"},
    # --- 入力映像プレビュー / 使用範囲 ---
    "vid_info": {"ja": "{w}×{h}  |  {n} frames  |  {fps:.2f} fps  |  {dur:.2f} 秒",
                  "en": "{w}×{h}  |  {n} frames  |  {fps:.2f} fps  |  {dur:.2f} s"},
    "lbl_use_range": {"ja": "使用範囲 / 再生位置:", "en": "Use range / playhead:"},
    "btn_range_full": {"ja": "全尺", "en": "Full"},
    "hint_use_range": {
        "ja": "(青=開始/終了・赤=再生位置。既定は全尺・頭合わせ。\n"
              "範囲は軌道データの時間軸調整で適用 — コピー不要・初期化不要)",
        "en": "(blue = start/end, red = playhead. Default: full length, head-aligned.\n"
              "Applied by adjusting the trajectory time axis — no copy, no re-init)"},
    "range_readout": {
        "ja": "{s:.2f} – {e:.2f} 秒  (使用尺 {d:.2f}s)   |   再生位置 {p:.2f}s",
        "en": "{s:.2f} – {e:.2f} s  (selected {d:.2f}s)   |   playhead {p:.2f}s"},
    # --- rate to data の同期点 ---
    "lbl_sync_anchor": {"ja": "同期点:", "en": "Sync point:"},
    "sync_head": {"ja": "頭", "en": "Head"},
    "sync_mid":  {"ja": "中央", "en": "Mid"},
    "sync_tail": {"ja": "尾", "en": "Tail"},
    "hint_sync_anchor": {
        "ja": "(rate to data: 全スリットの時刻が一致する出力位置。"
              "0%=先頭 / 50%=中央 / 100%=最終フレームで同期)",
        "en": "(rate to data: output position where all slit times coincide — "
              "0% head / 50% middle / 100% last frame)"},
    # --- 階調表示モード ---
    "chk_colormap": {"ja": "黄(255)–青(0) 表示", "en": "Yellow(255)–Blue(0) view"},
    "hint_colormap": {"ja": "(表示のみ。書き出す PNG はグレースケールのまま)",
                       "en": "(display only — exported PNGs stay grayscale)"},
    # --- 適用画像の後処理 ---
    "grp_postproc": {"ja": "適用画像の後処理 (破壊的)",
                      "en": "Post-process applied image (destructive)"},
    "btn_pp_invert": {"ja": "⚡ 階調反転 (元に戻せません)",
                       "en": "⚡ Invert tones (no undo)"},
    "lbl_pp_midgray": {"ja": "基準グレー:", "en": "Mid-gray:"},
    "hint_pp_midgray": {"ja": "(ヒストグラム中間値をずらす / 0.50 = 変更なし)",
                         "en": "(shifts the histogram midpoint / 0.50 = no change)"},
    "lbl_pp_rotate": {"ja": "回転:", "en": "Rotate:"},
    "hint_pp_rotate": {"ja": "(正=反時計回り。縦横サイズは維持したまま再マッピング)",
                        "en": "(+ = counter-clockwise; remapped, original pixel size kept)"},
    "btn_pp_apply": {"ja": "適用 (画像に書き込み)", "en": "Apply (write to image)"},
    "btn_pp_reset": {"ja": "リセット", "en": "Reset"},
    "pp_pending": {"ja": "▲ プレビュー中 — 「適用」で画像ファイルへ書き込みます",
                    "en": "▲ Preview only — press Apply to write to the image file"},
    "pp_no_image": {"ja": "(適用画像がまだありません)", "en": "(no applied image yet)"},
    "pp_confirm_title": {"ja": "破壊的な編集の確認", "en": "Confirm destructive edit"},
    "pp_confirm_invert": {
        "ja": "{t} の適用画像 ({f}) の階調を反転して上書きします。\n\n"
              "この操作は元に戻せません。\n"
              "(必要なら「生成して適用」でレイヤーから作り直せます)\n\n続行しますか?",
        "en": "This inverts the tones of the applied {t} image ({f}) and overwrites it.\n\n"
              "This cannot be undone.\n"
              "(You can rebuild it from the layers with Generate & Apply.)\n\nContinue?"},
    "pp_confirm_apply": {
        "ja": "{t} の適用画像 ({f}) に次の後処理を書き込みます:\n{ops}\n\n"
              "この操作は元に戻せません。\n\n続行しますか?",
        "en": "The following post-process will be written into the applied {t} image ({f}):\n{ops}\n\n"
              "This cannot be undone.\n\nContinue?"},
    "pp_op_midgray": {"ja": "・基準グレー: 0.50 → {v:.2f}", "en": "・Mid-gray: 0.50 → {v:.2f}"},
    "pp_op_rotate": {"ja": "・回転: {v:.1f}°", "en": "・Rotate: {v:.1f}°"},
    "pp_nothing": {"ja": "後処理の変更がありません (基準グレー 0.50 / 回転 0°)。",
                    "en": "Nothing to apply (mid-gray 0.50 / rotation 0°)."},
}


def tr(key, **fmt):
    """現在の言語 LANG に応じた訳文を返す。未知キーはキー名をそのまま返す。"""
    d = TR.get(key)
    s = (d.get(LANG) or d.get("ja")) if d else key
    return s.format(**fmt) if fmt else s


# ======== 階調表示モード (グレースケール / 黄(255)–青(0)) ========
# GUI 上の "表示" にのみ効くモード。書き出される 16bit PNG は常にグレースケール
# なので、レンダリング結果には一切影響しない。
COLOR_MODE = "gray"          # "gray" | "yellowblue"

_YB_LUT = None


def _yb_lut():
    """0..255 → RGB の 黄(255)–青(0) ランプ LUT (uint8, shape (256, 3))。

    純粋な 青 (0,0,255) → グレー → 黄 (255,255,0) の RGB 補間をベースに、
    線形輝度を中間グレーの輝度へ部分補正して明度差をできるだけ抑える
    (完全equalize すると青緑〜オレンジに見えてしまうため色相は動かさない)。
    中間値 0.5 は無彩色グレーのままなので「基準グレー」も目視しやすい。
    """
    global _YB_LUT
    if _YB_LUT is None:
        W = np.array([0.2126, 0.7152, 0.0722], np.float32)   # 線形輝度係数
        v = np.arange(256, dtype=np.float32) / 255.0
        rgb = np.stack([v, v, 1.0 - v], axis=-1)             # 前版と同じ補間
        # sRGB → 線形
        lin = np.where(rgb <= 0.04045, rgb / 12.92,
                       ((rgb + 0.055) / 1.055) ** 2.4)
        Y = lin @ W
        Yt = float(Y[128])                                   # 目標 = 中間グレー
        # 部分補正 (65%): 黄側を減光 / 青側を増光 (クリップまで)
        gain = (Yt / np.maximum(Y, 1e-6)) ** 0.65
        lin = np.clip(lin * gain[:, None], 0.0, 1.0)
        # 青側はクリップで持ち上げきれないため、白を少量混ぜて補う (上限付き)
        add = np.clip((Yt - lin @ W) * 0.5, 0.0, 0.25)
        lin = lin + add[:, None] * (1.0 - lin)
        srgb = np.where(lin <= 0.0031308, lin * 12.92,
                        1.055 * np.power(lin, 1 / 2.4) - 0.055)
        _YB_LUT = (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return _YB_LUT


def colorize_gray8(gray8):
    """uint8 グレースケール (h, w) → 現在の階調表示モードの RGB uint8 (h, w, 3)。"""
    return np.ascontiguousarray(_yb_lut()[gray8])


def gray8_to_qpixmap(gray8):
    """uint8 グレースケール配列を、現在の階調表示モードで QPixmap 化する。"""
    gray8 = np.ascontiguousarray(gray8)
    h, w = gray8.shape[:2]
    if COLOR_MODE == "yellowblue":
        rgb = colorize_gray8(gray8)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    else:
        qimg = QImage(gray8.data, w, h, w, QImage.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)


def colorize_pixmap(pm):
    """既存 QPixmap をグレースケールとみなして着色する (gray モードでは素通し)。"""
    if COLOR_MODE != "yellowblue" or pm is None or pm.isNull():
        return pm
    img = pm.toImage().convertToFormat(QImage.Format_Grayscale8)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    gray = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())[:, :w]
    return gray8_to_qpixmap(gray)


# ======== 適用画像の後処理 (破壊的 / uint16 ファイル座標系で動作) ========

def read_map16(path):
    """マップ PNG を uint16 (h, w) 単チャンネルで読み込む (失敗時 None)。"""
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != np.uint16:
        m = (m.astype(np.float32) / 255.0 * 65535.0 + 0.5).astype(np.uint16)
    return m


def pp_invert(img16):
    """階調反転 (v → 65535 - v)。"""
    return (65535 - img16.astype(np.int32)).astype(np.uint16)


def pp_midgray(img16, target):
    """基準グレーの移動: 入力 0.5 が target に来るガンマ補正 (Levels の中間調)。

    端点 0 / 1 は固定されるため、レンジを保ったままヒストグラムの重心だけが
    動く。target=0.5 は恒等変換。
    """
    t = float(np.clip(target, 0.01, 0.99))
    if abs(t - 0.5) < 1e-6:
        return img16
    gamma = float(np.log(t) / np.log(0.5))
    x = img16.astype(np.float32) / 65535.0
    return (np.clip(np.power(x, gamma), 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def pp_rotate(img16, deg):
    """画像を deg 回転し、回転後の外接矩形を元の W×H へ引き伸ばして戻す。

    「縦横の画像サイズは維持したまま再マッピングする」方針。90° 回転や斜め
    回転では縦横比が変わるので非等方スケールがかかり、多少のブロックノイズ /
    ぼけが出るが、img_to_maneuver が要求する space/time の形状一致は保たれる。
    """
    deg = float(deg) % 360.0
    if abs(deg) < 1e-6:
        return img16
    h, w = img16.shape[:2]
    a = np.deg2rad(deg)
    c, s = abs(np.cos(a)), abs(np.sin(a))
    bw = w * c + h * s          # 回転後の外接矩形
    bh = w * s + h * c
    sx, sy = w / max(bw, 1e-6), h / max(bh, 1e-6)
    cx, cy = w / 2.0, h / 2.0
    # p' = C + diag(sx,sy) · R · (p - C)
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    M[0, :] *= sx
    M[1, :] *= sy
    M[0, 2] += cx * (1.0 - sx)
    M[1, 2] += cy * (1.0 - sy)
    return cv2.warpAffine(img16, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def pp_apply_pending(img16, midgray=0.5, rotate=0.0):
    """未適用の後処理 (基準グレー → 回転) をまとめて適用する。"""
    out = pp_midgray(img16, midgray)
    return pp_rotate(out, rotate)


# ======== rate to data: 同期点 (スリット時刻が一致する出力位置) ========
def apply_sync_anchor(dm, anchor01):
    """各スリットの時間軌道に列ごとの定数オフセットを与え、出力タイムライン上の
    anchor01 (0=頭 / 0.5=中央 / 1=最終フレーム) で全スリットの時刻を一致させる。

    rate to data の累積積分は先頭フレームで全スリット同時刻 (=startpoint) から
    始まり徐々にズレていく。この関数はその「同期位置」を任意の出力位置へ移す
    (anchor01=0 は現状どおりで no-op)。同期時刻は各スリットの anchor 時刻の
    平均にするため、全体の時間的な置き場所は大きく動かない。
    """
    a = float(anchor01)
    if a <= 1e-9:
        return
    z = dm.data[:, :, 1]
    row = int(round(min(1.0, a) * (z.shape[0] - 1)))
    ref = z[row, :].copy()
    dm.data[:, :, 1] = z - ref[None, :] + float(ref.mean())
    try:
        dm.maneuver_log(f"SyncAnchor{a:.2f}")
    except Exception:
        pass


# ======== 使用範囲: 軌道データの時間軸調整 ========
def fit_trajectory_to_range(dm, s_frame, e_frame):
    """軌道データ (dm.data[:,:,1] = 参照する入力フレーム番号) を
    使用範囲 [s_frame, e_frame] に収める。ファイルコピーは作らない。

    1. applyTimeSlide: 冒頭フレーム (中央スリット) の参照時刻を範囲開始へ
       スライドする (頭合わせ)
    2. 範囲開始より前を参照する軌道があれば全体を押し上げる
    3. 範囲終了を超える軌道があれば開始点基準でスケーリングして収める
       (zPointCheck の [0, count] 版と同じ方針を [s, e] に適用)
    """
    dm.applyTimeSlide(int(round(s_frame)), baseframe=0)
    z = dm.data[:, :, 1]
    zmin = float(np.amin(z))
    if zmin < s_frame:
        dm.data[:, :, 1] += (s_frame - zmin)
    zmax = float(np.amax(dm.data[:, :, 1]))
    if zmax > e_frame and (zmax - s_frame) > 1e-9:
        scale = (e_frame - s_frame) / (zmax - s_frame)
        dm.data[:, :, 1] = s_frame + (dm.data[:, :, 1] - s_frame) * scale
        print(f"range fit: scaled x{scale:.4f} into [{s_frame}, {e_frame}]")
    try:
        dm.maneuver_log(f"RangeFit{int(s_frame)}-{int(e_frame)}")
    except Exception:
        pass


# ======== 入力映像の回転 ========
# 90°系: Display Matrix メタデータの書き換えリマックス (-display_rotation +
#        ストリームコピー)。再エンコードなしで瞬時・画質劣化ゼロ。
#        cv2 は自動回転、imgtrans は既存の input_rotation 機構が解釈する。
# 反転:  メタデータで表現できないため従来どおり ffmpeg 再エンコード。
VIDEO_ROTATIONS = [
    ("none",  "vrot_none",  None),
    ("cw90",  "vrot_cw90",  "transpose=1"),
    ("180",   "vrot_180",   "transpose=1,transpose=1"),
    ("ccw90", "vrot_ccw90", "transpose=2"),
    ("hflip", "vrot_hflip", "hflip"),
    ("vflip", "vrot_vflip", "vflip"),
]
VIDEO_ROTATION_VF = {rid: vf for rid, _key, vf in VIDEO_ROTATIONS}
# Display Matrix の角度 (probe_video_rotation / frame_to_ndarray の規約:
# 正 = 反時計回り)。ここに無い id (hflip/vflip) は再エンコードで対応。
VIDEO_ROTATION_ANGLE = {"cw90": -90, "180": 180, "ccw90": 90}

# プレビュー用: 回転 id → cv2 での即時変換 (ffmpeg を待たずに見た目を確認)
def apply_rotation_cv2(frame, rot_id):
    if rot_id == "cw90":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot_id == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rot_id == "ccw90":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot_id == "hflip":
        return cv2.flip(frame, 1)
    if rot_id == "vflip":
        return cv2.flip(frame, 0)
    return frame


def probe_video(path):
    """ffprobe で pix_fmt / 色メタ / 総フレーム数を拾う (取れないキーは None)。"""
    info = {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=pix_fmt,color_primaries,color_transfer,color_space,nb_frames",
             "-of", "default=noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = None if v.strip() in ("", "unknown", "N/A") else v.strip()
    except Exception:
        pass
    return info


def rotated_video_path(src, rot_id):
    """回転済みコピーの出力パス (元映像と同じフォルダ)。"""
    p = Path(src)
    return str(p.with_name(f"{p.stem}_rot-{rot_id}{p.suffix}"))


class VideoRotateWorker(QThread):
    """ffmpeg で入力映像の回転コピーを作るワーカー。

    rot_angle が指定されると Display Matrix メタデータの書き換えリマックス
    (ストリームコピー・瞬時) を行い、それ以外は vf で再エンコードする。
    rot_angle は「追加の表示回転角」— 既存メタデータの角度と合算して書き込む。
    """
    log_signal = pyqtSignal(str)
    progress = pyqtSignal(int)
    done_signal = pyqtSignal(bool, str)

    def __init__(self, src, out, vf, info, rot_angle=None):
        super().__init__()
        self.src, self.out, self.vf = src, out, vf
        self.info = info or {}
        self.rot_angle = rot_angle

    def _encode_args(self):
        """元素材の pix_fmt / 色メタをできるだけ引き継ぐ x264 設定。"""
        args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12"]
        pix = self.info.get("pix_fmt")
        if pix:
            args += ["-pix_fmt", pix]
        for key, flag in (("color_primaries", "-color_primaries"),
                          ("color_transfer", "-color_trc"),
                          ("color_space", "-colorspace")):
            v = self.info.get(key)
            if v:
                args += [flag, v]
        return args

    def run(self):
        if self.rot_angle is not None:
            self._run_remux()
            return
        total = 0
        try:
            total = int(self.info.get("nb_frames") or 0)
        except Exception:
            total = 0
        cmd = (["ffmpeg", "-y", "-nostdin", "-i", self.src, "-vf", self.vf]
               + self._encode_args()
               + ["-c:a", "copy", "-progress", "pipe:1", "-nostats",
                  "-loglevel", "error", self.out])
        self.log_signal.emit("[ffmpeg] " + " ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            self.log_signal.emit("[ERROR] ffmpeg not found")
            self.done_signal.emit(False, "")
            return
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("frame=") and total > 0:
                try:
                    n = int(line.split("=", 1)[1])
                    self.progress.emit(min(99, int(n * 100 / total)))
                except Exception:
                    pass
        err = proc.stderr.read()
        rc = proc.wait()
        if err.strip():
            self.log_signal.emit(err.strip())
        if rc != 0 or not os.path.exists(self.out):
            self.log_signal.emit(f"[ERROR] ffmpeg exited with {rc}")
            self.done_signal.emit(False, "")
            return
        self.progress.emit(100)
        self.done_signal.emit(True, self.out)

    def _run_remux(self):
        """Display Matrix メタデータ書き換えのみのリマックス (再エンコードなし)。

        既存メタデータの回転角と指定角を合算した値を書き込む
        (例: iPhone 縦位置 (-90) + 右90° (-90) = -180)。
        """
        try:
            from imgtrans_lib._utils import probe_video_rotation
            existing = int(probe_video_rotation(self.src))
        except Exception:
            existing = 0
        total_angle = (existing + int(self.rot_angle)) % 360
        if total_angle > 180:
            total_angle -= 360
        cmd = ["ffmpeg", "-y", "-nostdin",
               "-display_rotation", str(total_angle),
               "-i", self.src, "-c", "copy",
               "-loglevel", "error", self.out]
        self.log_signal.emit("[ffmpeg remux] " + " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=600)
        except FileNotFoundError:
            self.log_signal.emit("[ERROR] ffmpeg not found")
            self.done_signal.emit(False, "")
            return
        if proc.stderr.strip():
            self.log_signal.emit(proc.stderr.strip())
        if proc.returncode != 0 or not os.path.exists(self.out):
            self.log_signal.emit(f"[ERROR] ffmpeg exited with {proc.returncode}")
            self.done_signal.emit(False, "")
            return
        self.progress.emit(100)
        self.done_signal.emit(True, self.out)


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
                 anim_only=False, rate_baseline=None, rate_startpoint=None,
                 audio_out=False, audio_mode="play", use_range=None,
                 sync_anchor=0.0, audio_voices=7, audio_grain_ms=90):
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
        self.audio_out = audio_out      # 音声を適用して最終出力を作る
        self.audio_mode = audio_mode    # "play"=可変速再生 / "grain"=グラニュラー
        self.audio_voices = max(1, int(audio_voices))    # ボイス分割数
        self.audio_grain_ms = max(20, int(audio_grain_ms))  # グレイン長 (ms)
        # 使用範囲 (start_frame, end_frame)。None = 全尺。
        # コピーは作らず、zPointCheck 後に軌道の時間軸をこの範囲へ調整する。
        self.use_range = use_range
        # 同期点 (0..1)。rate to data で全スリット時刻が一致する出力位置。
        self.sync_anchor = float(sync_anchor or 0.0)

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

            # rate to data: 同期点 (全スリット時刻が一致する出力位置) を適用
            if self.mode == "rate to data" and self.sync_anchor > 0:
                self.emit(f"sync anchor: {self.sync_anchor:.2f}")
                apply_sync_anchor(bm, self.sync_anchor)

            bm.zPointCheck()
            # 使用範囲: 軌道の時間軸を範囲へスライド/フィット (コピー不要)
            if self.use_range is not None:
                self.emit(f"applyTimeSlide + range fit: frames {self.use_range[0]}"
                          f"–{self.use_range[1]}")
                fit_trajectory_to_range(bm, *self.use_range)
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

            # 音声適用: new_transprocess の出力 (out_videopath) に対して
            # audio_render (mode=play/grain) → mux した動画を最終出力にする。
            # del_data=False で self.data が残っているため音声レンダリング可能。
            if self.audio_out and video_path:
                try:
                    self.emit(f"=== audio_video_out (mode={self.audio_mode}, "
                              f"voices={self.audio_voices}, "
                              f"grain={self.audio_grain_ms}ms) ===")
                    audio_final = bm.audio_video_out(
                        mode=self.audio_mode,
                        thread_num=self.audio_voices,
                        grain_dur=self.audio_grain_ms / 1000.0)
                    if audio_final and os.path.exists(audio_final):
                        video_path = audio_final
                        self.emit(f"audio applied: {os.path.basename(audio_final)}")
                    else:
                        self.emit("[WARN] audio_video_out returned no file; "
                                  "using video-only output.")
                except Exception as e:
                    self.emit(f"[WARN] audio_video_out failed: {e} — "
                              "音声なしの出力を使用します")

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
                 skip_2d=False, plot_w_inc=None, plot_h_inc=None,
                 plot3d_fig=None, gif_width=400, use_range=None,
                 sync_anchor=0.0):
        super().__init__()
        self.dm = dm
        self.mode = mode  # "time" or "rate"
        self.skip_2d = skip_2d   # ライブ3Dプレビュー用: 2D プロット生成を省略
        # 2D プロットの図サイズ (インチ)。None なら dm の既定値のまま。
        self.plot_w_inc = plot_w_inc
        self.plot_h_inc = plot_h_inc
        # 3D プロットの表示領域フィット: (fig_w_inc, fig_h_inc, box_aspect)。
        # None なら maneuver_3dplot の既定動作。
        self.plot3d_fig = plot3d_fig
        self.gif_width = max(160, int(gif_width))
        self.use_range = use_range   # (start_f, end_f) or None — 軌道の時間軸調整
        self.sync_anchor = float(sync_anchor or 0.0)   # rate モードの同期点
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

            # rate モード: 同期点を適用してから範囲チェック/フィット
            if self.mode == "rate" and self.sync_anchor > 0:
                apply_sync_anchor(self.dm, self.sync_anchor)

            self.progress_signal.emit("zPointCheck…")
            self.percent_signal.emit(25)
            self.dm.zPointCheck()
            if self.use_range is not None:
                fit_trajectory_to_range(self.dm, *self.use_range)

            # 2D プロット生成: mtime で「呼び出し後に変更されたファイル」を検出
            # (同じファイル名で上書きされるケースに対応するため set 差分は使わない)
            plot2d = ""
            if not self.skip_2d:
                ts_2d = time.time() - 0.5  # 小さなクロックスラックを許容
                self.progress_signal.emit("maneuver_2dplot: 2D プロット生成中…")
                self.percent_signal.emit(35)
                # 表示領域のアスペクト比に合わせた図サイズを dm の
                # クラス変数 (plot_w_inc/plot_h_inc) に反映してから生成する
                if self.plot_w_inc and self.plot_h_inc:
                    self.dm.plot_w_inc = self.plot_w_inc
                    self.dm.plot_h_inc = self.plot_h_inc
                self.dm.maneuver_2dplot()
                plot2d = self._latest_file(cwd, (".png",), ts_2d)

            # 3D アニメ生成: 同じく mtime で検出
            ts_3d = time.time() - 0.5
            self.progress_signal.emit(
                f"maneuver_3dplot: 3D アニメ生成中 ({self.anim_frames} frames @ {self.anim_dpi} dpi)…"
            )
            self.percent_signal.emit(55)
            kw3d = {}
            if self.plot3d_fig:
                fw, fh, box = self.plot3d_fig
                kw3d = dict(fig_w_inc=fw, fig_h_inc=fh, box_aspect=box)
            self.dm.maneuver_3dplot(
                out_framenums=self.anim_frames,
                out_fps=self.anim_fps,
                dpi=self.anim_dpi,
                **kw3d,
            )
            mp4 = self._latest_file(cwd, (".mp4", ".mov"), ts_3d)

            gif = ""
            if mp4 and os.path.exists(mp4):
                gif = os.path.splitext(mp4)[0] + "_preview.gif"
                self.progress_signal.emit("ffmpeg で GIF 変換…")
                self.percent_signal.emit(85)
                # 表示領域幅に合わせてスケール (高さは自動)、ループ無限
                cmd = ["ffmpeg", "-y", "-i", mp4,
                       "-vf", f"fps={self.anim_fps},"
                              f"scale={self.gif_width}:-1:flags=lanczos",
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


# ======== stdout の進捗% 検出 (レンダリング進捗バー用) ========
class StdoutPercentTee:
    """sys.stdout を透過しつつ「NN%」を検出してコールバックする。

    imgtrans のレンダリングは進捗バーを stdout に print するため、
    レンダリング中だけ差し込んで GUI のプログレスバーへ転送する。
    "57.9%" のような小数 (メモリ表示等) は除外する。
    """
    _PCT_RE = re.compile(r"(?<![\d.])(\d{1,3})%")

    def __init__(self, orig, cb):
        self._orig = orig
        self._cb = cb

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        m = None
        for m in self._PCT_RE.finditer(s):
            pass
        if m is not None:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    self._cb(v)
            except Exception:
                pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass


# ======== 使用範囲 + 再生位置の 3 ライン タイムライン ========
class RangeTimelineSlider(QWidget):
    """開始 / 終了 / 再生位置の 3 本のラインを 1 本のバーで操作する UI。

    - 開始・終了ライン (青) : 入力映像の使用範囲。間の領域が着色される
    - 再生ライン (赤)       : プレビュー表示フレーム
    - 実使用バンド (緑)     : 軌道データ生成後、実際に参照している時間範囲。
      バンド内をドラッグすると前後スライド、端をドラッグすると伸縮でき、
      usedRangeChanged で通知される (Time 画像の vmin/vmax へ反映される)。
    ラインの近くを押すとそのラインを掴んでドラッグ。どのラインからも
    離れた位置を押すと再生ラインがそこへジャンプする。
    値はすべて 0..1 の割合 (秒への換算は呼び出し側)。
    """
    rangeChanged = pyqtSignal(float, float)      # (start_frac, end_frac)
    playheadChanged = pyqtSignal(float)          # frac
    usedRangeChanged = pyqtSignal(float, float)  # 実使用バンドの操作

    GRAB_PX = 8          # ラインのつかみ判定 (px)
    MIN_GAP = 0.005      # 開始と終了の最小間隔 (割合)

    def __init__(self):
        super().__init__()
        self._start = 0.0
        self._end = 1.0
        self._pos = 0.0
        self._used = None            # (s, e) 実使用範囲 / None = 未表示
        # "start"|"end"|"pos"|"used_start"|"used_end"|"used_body"|None
        self._drag = None
        self._drag_dx = 0.0          # used_body ドラッグの掴んだ位置オフセット
        self.setFixedHeight(34)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    # --- 値 (プログラム側から。シグナルは発火しない) ---
    def set_range(self, s, e):
        self._start = min(max(0.0, float(s)), 1.0)
        self._end = min(max(self._start + self.MIN_GAP, float(e)), 1.0)
        self.update()

    def set_playhead(self, f):
        self._pos = min(max(0.0, float(f)), 1.0)
        self.update()

    def set_used_range(self, s, e):
        """軌道データが実際に参照している範囲 (緑バンド) を表示する。"""
        if s is None or e is None:
            self._used = None
        else:
            s = min(max(0.0, float(s)), 1.0)
            e = min(max(s + self.MIN_GAP, float(e)), 1.0)
            self._used = (s, e)
        self.update()

    def clear_used_range(self):
        self._used = None
        self.update()

    def values(self):
        return self._start, self._end, self._pos

    def used_range(self):
        return self._used

    # --- 座標変換 ---
    def _frac_to_x(self, f):
        return 2 + f * max(1, self.width() - 5)

    def _x_to_frac(self, x):
        return min(1.0, max(0.0, (x - 2) / max(1, self.width() - 5)))

    # --- マウス操作 ---
    def _hit_test(self, x):
        """つかむ対象を返す。優先度: 3ライン > 緑バンド端 > 緑バンド内。"""
        cands = [(abs(x - self._frac_to_x(self._pos)), "pos"),
                 (abs(x - self._frac_to_x(self._start)), "start"),
                 (abs(x - self._frac_to_x(self._end)), "end")]
        cands.sort()
        if cands[0][0] <= self.GRAB_PX:
            return cands[0][1]
        if self._used is not None:
            u0, u1 = self._used
            x0, x1 = self._frac_to_x(u0), self._frac_to_x(u1)
            if abs(x - x0) <= self.GRAB_PX:
                return "used_start"
            if abs(x - x1) <= self.GRAB_PX:
                return "used_end"
            if x0 < x < x1:
                return "used_body"
        return None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        hit = self._hit_test(ev.pos().x())
        if hit is None:
            hit = "pos"                       # 何もない所 → 再生ラインをジャンプ
            self._apply_drag("pos", self._x_to_frac(ev.pos().x()))
        elif hit == "used_body":
            self._drag_dx = self._x_to_frac(ev.pos().x()) - self._used[0]
        self._drag = hit

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._apply_drag(self._drag, self._x_to_frac(ev.pos().x()))
        else:
            hit = self._hit_test(ev.pos().x())
            if hit == "used_body":
                self.setCursor(Qt.OpenHandCursor)
            elif hit:
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def _apply_drag(self, which, f):
        if which == "start":
            self._start = min(f, self._end - self.MIN_GAP)
            self._start = max(0.0, self._start)
            self.rangeChanged.emit(self._start, self._end)
        elif which == "end":
            self._end = max(f, self._start + self.MIN_GAP)
            self._end = min(1.0, self._end)
            self.rangeChanged.emit(self._start, self._end)
        elif which in ("used_start", "used_end", "used_body") \
                and self._used is not None:
            u0, u1 = self._used
            if which == "used_start":
                u0 = min(max(0.0, f), u1 - self.MIN_GAP)
            elif which == "used_end":
                u1 = max(min(1.0, f), u0 + self.MIN_GAP)
            else:                                # 前後スライド (尺は維持)
                span = u1 - u0
                u0 = min(max(0.0, f - self._drag_dx), 1.0 - span)
                u1 = u0 + span
            self._used = (u0, u1)
            self.usedRangeChanged.emit(u0, u1)
        else:
            self._pos = f
            self.playheadChanged.emit(self._pos)
        self.update()

    # --- 描画 ---
    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        bar_y, bar_h = 6, h - 12
        # ベース
        p.fillRect(2, bar_y, w - 4, bar_h, QColor(40, 40, 40))
        # 選択範囲
        x0 = int(self._frac_to_x(self._start))
        x1 = int(self._frac_to_x(self._end))
        p.fillRect(x0, bar_y, max(1, x1 - x0), bar_h, QColor(42, 111, 214, 90))
        # 実使用バンド (緑・内側に細く表示。掴んでスライド/伸縮できる)
        if self._used is not None:
            u0 = int(self._frac_to_x(self._used[0]))
            u1 = int(self._frac_to_x(self._used[1]))
            band_y = bar_y + bar_h // 4
            band_h = max(3, bar_h // 2)
            p.fillRect(u0, band_y, max(1, u1 - u0), band_h,
                       QColor(60, 200, 120, 160))
            pen = QPen(QColor(60, 220, 130))
            pen.setWidth(2)
            p.setPen(pen)
            for x in (u0, u1):
                p.drawLine(x, band_y - 2, x, band_y + band_h + 2)
        # 開始/終了ライン (青) — 上下に短いつまみ
        pen = QPen(QColor(90, 160, 255))
        pen.setWidth(3)
        p.setPen(pen)
        for x in (x0, x1):
            p.drawLine(x, 0, x, h)
        # 再生ライン (赤)
        xp = int(self._frac_to_x(self._pos))
        pen = QPen(QColor(255, 40, 40))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(xp, 0, xp, h)
        p.end()


# ======== 適用済みマップのサムネイル (再生位置の赤ライン付き) ========
class MapThumb(QLabel):
    """適用済みの space/time/rate マップを小さく表示し、3D アニメの再生位置を
    赤いラインで重ねるサムネイル。

    - 縦スリット (sd=1): マップの時間軸は縦 → 水平の赤ラインが上下に動く
    - 横スリット (sd=0): マップの時間軸は横 → 垂直の赤ラインが左右に動く
    """

    def __init__(self, caption="", fixed_height=110, colorizable=True,
                 stretch=False):
        super().__init__()
        self._src = None            # 元画像 QPixmap
        self._base = None           # ラベルサイズに合わせた縮小キャッシュ
        self._frac = None           # 再生位置 [0,1) / None = 非表示
        self._time_vertical = True
        # グレースケールのマップだけ「黄–青」表示モードの対象にする
        # (2D プロットのような既に色付きの画像は colorizable=False)
        self._colorizable = bool(colorizable)
        # stretch=True: 縦横比を無視して表示領域いっぱいに引き伸ばす
        # (適用マップのサムネイル用。プロット画像は False のまま)
        self._stretch = bool(stretch)
        # 赤ラインの可動範囲 (表示画像内の割合)。2D プロットでは軸の
        # データ領域だけを動くように外から設定される。
        self._range = (0.0, 1.0)
        self.setAlignment(Qt.AlignCenter)
        if fixed_height is not None:
            self.setFixedHeight(fixed_height)
            self.setMinimumWidth(100)
        else:
            # 可変サイズ (レイアウトのストレッチに従う)。sizeHint 由来の
            # 拡大ループを避けるため Ignored ポリシーにする。
            self.setMinimumSize(160, 180)
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

    def set_playhead_range(self, x0, x1):
        """赤ラインの可動範囲を画像内の割合 [x0, x1] に制限する。

        2D プロットの PNG には軸ラベル等の余白が含まれるため、データ領域
        (時間軸の始点〜終点) だけを赤ラインが動くように調整する。
        """
        try:
            x0 = float(x0); x1 = float(x1)
        except (TypeError, ValueError):
            return
        if 0.0 <= x0 < x1 <= 1.0:
            self._range = (x0, x1)
            self._recompose()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()

    def refresh_color_mode(self):
        """階調表示モードが切り替わったときに再着色する。"""
        self._rescale()

    def _rescale(self):
        if self._src is None:
            self._base = None
            return
        base = self._src.scaled(
            max(10, self.width() - 2), max(10, self.height() - 2),
            Qt.IgnoreAspectRatio if self._stretch else Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        if self._colorizable:
            base = colorize_pixmap(base)
        self._base = base
        self._recompose()

    def _recompose(self):
        if self._base is None:
            return
        pm = QPixmap(self._base)
        if self._frac is not None:
            r0, r1 = self._range
            f = r0 + self._frac * (r1 - r0)   # 可動範囲内へ写像
            p = QPainter(pm)
            pen = QPen(QColor(255, 40, 40))
            pen.setWidth(2)
            p.setPen(pen)
            if self._time_vertical:
                y = int(f * (pm.height() - 1))
                p.drawLine(0, y, pm.width(), y)
            else:
                x = int(f * (pm.width() - 1))
                p.drawLine(x, 0, x, pm.height())
            p.end()
        self.setPixmap(pm)


# ======== Main GUI ========
class IMGTransApp(QWidget):
    _render_pct = pyqtSignal(int)   # レンダリング進捗% (stdout 検出 → バー)

    def __init__(self):
        super().__init__()
        self._orig_stdout = None
        self._render_pct.connect(lambda v: self.render_progress.setValue(v))
        self.setWindowTitle(tr("window_title"))
        self.resize(1360, 900)   # 3カラム (Space/Time/Rate) を横並びで収める幅
        self.setMinimumSize(640, 480)

        self.videopath = None        # 実際に処理する映像 (回転指定時は回転済みコピー)
        self.videopath_src = None    # ユーザーが選択した元映像
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
        self._live2d_last_aspect = None   # 直近の 2D プロット生成時のアスペクト比
        # ウィンドウリサイズ後、表示領域のアスペクト比が大きく変わっていたら
        # 2D プロットを作り直す (デバウンス)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(900)
        self._resize_timer.timeout.connect(self._maybe_replot_on_resize)
        self._live3d_movie = None
        self._live3d_frames = None   # 同期表示用の先読み GIF フレーム
        self._live3d_timer = QTimer(self)
        self._live3d_timer.setSingleShot(True)
        self._live3d_timer.setInterval(800)     # 編集のデバウンス
        self._live3d_timer.timeout.connect(self._run_live3d)
        # プロット同期タイマ: GPU 映像の再生位置 → 赤ライン/3D GIF フレーム
        self._plot_sync_timer = QTimer(self)
        self._plot_sync_timer.setInterval(100)   # 10fps で十分滑らか
        self._plot_sync_timer.timeout.connect(self._sync_plots_tick)
        self._plot_sync_timer.start()

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
            # 後処理パネルの「未適用」表示も現在言語で貼り直す
            self._update_postproc_state(t)
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

        # --- 入力映像プレビュー (選択直後に表示。回転操作を即時反映) ---
        self._vid_cap = None            # プレビュー用 VideoCapture
        self._vid_info = None           # (w, h, fps, frames, dur_sec)
        self._vid_pos = -1              # cap が次に read するフレーム番号
        # 常時ループ再生: 使用範囲 (緑バンド優先) を実時間で再生し続ける。
        # スクラブ中は一時停止し、しばらく操作が無ければ自動再開する。
        self._vplay_timer = QTimer(self)
        self._vplay_timer.setInterval(66)          # ≈15fps 表示 (再生は実時間)
        self._vplay_timer.timeout.connect(self._vplay_tick)
        self._vplay_t0 = None                      # 再生アンカーの wallclock
        self._vplay_anchor = 0.0                   # t0 時点のフレーム位置
        self._vplay_paused = False                 # スクラブによる一時停止
        self._vplay_resume = QTimer(self)
        self._vplay_resume.setSingleShot(True)
        self._vplay_resume.setInterval(1500)       # 操作後 1.5s で再生再開
        self._vplay_resume.timeout.connect(self._vplay_resume_now)
        self.video_preview = QLabel()
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.video_preview.setFixedHeight(150)
        self.video_preview.setStyleSheet(
            "QLabel { background: #111; border: 1px solid #555; }")
        self.video_preview.setVisible(False)
        self.video_dim_label = QLabel("")
        self.video_dim_label.setStyleSheet("color: gray; font-size: 10px;")
        self.video_dim_label.setVisible(False)

        # 使用範囲 + 再生位置: 3 ラインの統合タイムライン。既定 = 全尺・頭合わせ。
        self.range_frame = QFrame()
        rf_v = QVBoxLayout(self.range_frame)
        rf_v.setContentsMargins(0, 0, 0, 0)
        rf_v.setSpacing(2)
        rr = QHBoxLayout()
        rr.addWidget(self._trlabel("lbl_use_range"))
        rr.addStretch()
        self.range_full_btn = QPushButton()
        self._reg(lambda: self.range_full_btn.setText(tr("btn_range_full")))
        self.range_full_btn.setMaximumWidth(56)
        self.range_full_btn.clicked.connect(self._set_range_full)
        rr.addWidget(self.range_full_btn)
        rf_v.addLayout(rr)
        self.range_slider = RangeTimelineSlider()
        self.range_slider.rangeChanged.connect(self._on_range_changed)
        self.range_slider.playheadChanged.connect(self._on_playhead_dragged)
        self.range_slider.usedRangeChanged.connect(self._on_used_range_dragged)
        self._range_override = None   # 緑バンド操作による実使用範囲 (frames)
        rf_v.addWidget(self.range_slider)
        self.range_span_label = QLabel("")
        self.range_span_label.setStyleSheet("color: gray; font-size: 10px;")
        rf_v.addWidget(self.range_span_label)
        rng_hint = self._trlabel("hint_use_range")
        rng_hint.setStyleSheet("color: gray; font-size: 10px;")
        rng_hint.setWordWrap(True)
        rf_v.addWidget(rng_hint)
        self.range_frame.setVisible(False)

        # --- Slit toggle ---
        self.slit_toggle = QCheckBox()
        self._reg(lambda: self.slit_toggle.setText(tr("chk_vertical")))
        self.slit_label = QLabel(tr("slit_h"))
        self.slit_toggle.stateChanged.connect(self.update_slit_label)
        self._reg(self.update_slit_label)  # 言語切替時にスリット表示も更新

        # --- 入力映像の回転 (Initialize 時に ffmpeg で回転コピーを作る) ---
        self.vrot_combo = QComboBox()
        for rid, key, _vf in VIDEO_ROTATIONS:
            self.vrot_combo.addItem(tr(key), rid)
        self._reg(lambda: [self.vrot_combo.setItemText(i, tr(k))
                           for i, (_r, k, _v) in enumerate(VIDEO_ROTATIONS)])
        self.vrot_combo.currentIndexChanged.connect(self._on_video_rotation_changed)
        vrot_row = QHBoxLayout()
        vrot_row.addWidget(self._trlabel("lbl_video_rotate"))
        vrot_row.addWidget(self.vrot_combo, 1)
        self.vrot_row = vrot_row
        self.vrot_hint = self._trlabel("hint_video_rotate")
        self.vrot_hint.setStyleSheet("color: gray; font-size: 10px;")
        self.vrot_hint.setWordWrap(True)
        self.vrot_progress = QProgressBar()
        self.vrot_progress.setRange(0, 100)
        self.vrot_progress.setTextVisible(True)
        self.vrot_progress.setVisible(False)
        self._vrot_worker = None

        # --- 階調表示モード (グレースケール / 黄–青) ---
        self.colormap_chk = QCheckBox()
        self._reg(lambda: self.colormap_chk.setText(tr("chk_colormap")))
        self.colormap_chk.toggled.connect(self._on_colormap_toggled)
        self.colormap_hint = self._trlabel("hint_colormap")
        self.colormap_hint.setStyleSheet("color: gray; font-size: 10px;")
        self.colormap_hint.setWordWrap(True)

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

        # 同期点: 全スリットの時刻が一致する出力位置 (0=頭 / 0.5=中央 / 1=尾)。
        # rate to data のときのみ軌道生成へ効く (時間マップは変形しない)。
        sync_row = QHBoxLayout()
        sync_row.addWidget(self._trlabel("lbl_sync_anchor"))
        self.sync_anchor_slider = QSlider(Qt.Horizontal)
        self.sync_anchor_slider.setRange(0, 100)
        self.sync_anchor_slider.setValue(0)          # 既定 = 頭 (従来動作)
        self.sync_anchor_slider.valueChanged.connect(self._on_sync_anchor_changed)
        sync_row.addWidget(self.sync_anchor_slider, 1)
        self.sync_anchor_val = QLabel("0%")
        self.sync_anchor_val.setMinimumWidth(34)
        sync_row.addWidget(self.sync_anchor_val)
        for key, v in (("sync_head", 0), ("sync_mid", 50), ("sync_tail", 100)):
            b = QPushButton()
            self._reg(lambda b_=b, k=key: b_.setText(tr(k)))
            b.setMaximumWidth(44)
            b.clicked.connect(lambda *_, vv=v:
                              self.sync_anchor_slider.setValue(vv))
            sync_row.addWidget(b)
        sync_hint = self._trlabel("hint_sync_anchor")
        sync_hint.setStyleSheet("color: gray; font-size: 10px;")
        sync_hint.setWordWrap(True)

        self.rate_param_frame = QFrame()
        rate_vbox = QVBoxLayout(self.rate_param_frame)
        rate_vbox.addLayout(rate_layout)
        rate_vbox.addLayout(sync_row)
        rate_vbox.addWidget(sync_hint)
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
        self.live2d_thumb = MapThumb("2D Plot", fixed_height=None, colorizable=False)
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
        self.live3d_label.setMinimumSize(320, 200)
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
            # 適用マップは表示領域いっぱいに引き伸ばして表示 (縦横比可変)
            th = MapThumb(cap, stretch=True)
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

        # --- Animation 系 UI (アニメーション書き出しは一旦 UI から撤去。
        #     内部ロジック互換のためウィジェットは生成のみ・レイアウト非配置) ---
        self.anim_toggle = QCheckBox()
        self.anim_toggle.stateChanged.connect(self.on_anim_toggle_changed)
        self.anim_settings_container = QFrame()
        anim_settings_layout = QVBoxLayout(self.anim_settings_container)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 120)
        self.duration_spin.setValue(10)
        anim_settings_layout.addWidget(self.duration_spin)
        self.anim_settings_container.setVisible(False)
        self.animonly_btn = QPushButton()
        self.animonly_btn.clicked.connect(self.start_animation_only)

        # --- Start Rendering ボタン (統合タブの出力行に配置) ---
        self.start_btn = QPushButton()
        self._reg(lambda: self.start_btn.setText(tr("btn_start_render")))
        self.start_btn.clicked.connect(self.start_rendering)

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
                  self.video_preview, self.video_dim_label,
                  self.range_frame,
                  self.slit_toggle, self.slit_label]:
            sg_l.addWidget(w)
        sg_l.addLayout(self.vrot_row)
        sg_l.addWidget(self.vrot_hint)
        sg_l.addWidget(self.vrot_progress)
        for w in [self.init_btn, self.info_label,
                  self.colormap_chk, self.colormap_hint]:
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
        self._t2_top_row = top_row   # live3d_group をタブ間で移動するための帰り先
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

        # --- Tab 2: プレビュー・出力 (Preview & Render 統合) ---
        # スクロール無しで「映像ビュー + 軌道プロットライブビュー + 出力操作」を
        # 1 画面に収める。アニメーション書き出し UI はこのタブには置かない。
        t3 = QWidget(); t3_l = QVBoxLayout(t3)

        # 上段バー: 適用方法表示 + 「映像ビューのみ表示」チェック
        pv_top = QHBoxLayout()
        pv_top.addWidget(self.apply_mode_info, 1)
        self.video_only_chk = QCheckBox()
        self._reg(lambda: self.video_only_chk.setText(tr("chk_video_only")))
        self.video_only_chk.toggled.connect(self._on_video_only_toggled)
        pv_top.addWidget(self.video_only_chk)
        t3_l.addLayout(pv_top)

        # 映像エリア: GPU リアルタイムプレビュー ⇄ レンダリング結果を差し替え
        if _HAS_RT_PREVIEW:
            self.rt_group = QGroupBox()
            self._reg(lambda: self.rt_group.setTitle(tr("grp_realtime")))
            rt_v = QVBoxLayout(self.rt_group)
            self.rt_preview = RealtimePreviewWidget(lang=LANG)
            rt_v.addWidget(self.rt_preview)
            t3_l.addWidget(self.rt_group, 3)
            # Rebuild / 構築ボタンで GPU ビューへ戻す
            self.rt_preview.rebuild_btn.clicked.connect(self._show_gpu_view)
            self.rt_preview.center_btn.clicked.connect(self._show_gpu_view)
        else:
            self.rt_preview = None
        self.rendered_preview = VideoPreview(tr("rendered_video_title"))
        self._reg(lambda: self.rendered_preview.set_base_title(tr("rendered_video_title")))
        self.rendered_preview.setVisible(False)
        t3_l.addWidget(self.rendered_preview, 3)

        # 軌道プロットライブビューの受け皿 (このタブ表示中は live3d_group を
        # タブ1からここへ移動して併置する)
        self._live3d_slot = QVBoxLayout()
        t3_l.addLayout(self._live3d_slot, 2)

        # 出力行: Start Rendering + 進捗バー
        # (音声の適用/モード/分割数/グレイン長は GPU プレビューの音声設定が
        #  そのまま書き出しにも使われる — 二重の選択 UI は置かない)
        render_row = QHBoxLayout()
        self.audio_out_info = QLabel("")
        self.audio_out_info.setStyleSheet("color: gray; font-size: 11px;")
        render_row.addWidget(self.audio_out_info)
        render_row.addWidget(self.start_btn, 1)
        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 100)
        self.render_progress.setValue(0)
        self.render_progress.setTextVisible(True)
        render_row.addWidget(self.render_progress, 2)
        t3_l.addLayout(render_row)

        # プレビューの音声設定変更 → 書き出し予定の表示を更新
        if self.rt_preview is not None:
            self.rt_preview.audio_chk.toggled.connect(self._update_audio_out_info)
            self.rt_preview.audio_method.currentIndexChanged.connect(
                self._update_audio_out_info)
            self.rt_preview.audio_voices_spin.valueChanged.connect(
                self._update_audio_out_info)
        self._reg(self._update_audio_out_info)

        tabs.addTab(t3, tr("tab_preview"))
        tabs.currentChanged.connect(self._on_tab_changed)

        # タブ見出しの再翻訳を登録
        self._reg(lambda: (
            self.tabs.setTabText(0, tr("tab_main")),
            self.tabs.setTabText(1, tr("tab_preview")),
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
        self._update_generate_gates()
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
        self.tabs.setTabEnabled(1, ready)   # プレビュー・出力 (統合)
        # 出力操作も同じ条件でゲート (レンダリング可能条件と一致)
        self.start_btn.setEnabled(ready)
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

    def _update_generate_gates(self):
        """適用方法に応じて Generate & Apply の可否を制御する。

        rate to data 選択中は Time が Rate から自動導出されるため、
        Time セクションの生成適用ボタンを無効化する。
        """
        g = self._section_gens.get("time", {})
        btn = g.get("generate_btn")
        if btn is None:
            return
        rate_mode = (self._selected_apply_mode() == "rate to data")
        btn.setEnabled(self.dm is not None and not rate_mode)
        btn.setToolTip(tr("tip_time_gen_disabled") if rate_mode else "")

    def on_apply_mode_changed(self, *_):
        mode = self._selected_apply_mode()
        if mode:
            self.log(f"Apply mode selected: {mode}")
            # 選択された基準画像から対になるマップを即導出
            self._sync_derived_maps()
        self._update_generate_gates()
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
        rng = self._range_frames()
        if rng is not None:
            v = min(v, rng[1] - rng[0])   # 使用範囲を超えないレンジに制限
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
            self.videopath_src = path          # ユーザーが選んだ元映像
            self.videopath = path              # 実際に処理する映像 (回転/切り出し後は差し替わる)
            self.video_label.setText(f"Selected: {path}")
            self.log(f"Video selected: {path}")
            self._open_video_preview(path)
            self.update_ui_state("video_selected")

    # --- 入力映像プレビュー / 使用範囲 ---
    def _open_video_preview(self, path):
        """選択直後: プレビュー用キャプチャを開き、情報表示と範囲 UI を初期化。"""
        self._vplay_timer.stop()
        self._vplay_resume.stop()
        if self._vid_cap is not None:
            try:
                self._vid_cap.release()
            except Exception:
                pass
            self._vid_cap = None
        self._vid_info = None
        self._vid_pos = -1
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.video_preview.setVisible(False)
            self.video_dim_label.setVisible(False)
            self.range_frame.setVisible(False)
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / max(1e-6, fps)
        self._vid_cap = cap
        self._vid_info = (w, h, fps, n, dur)
        self.video_dim_label.setText(
            tr("vid_info", w=w, h=h, n=n, fps=fps, dur=dur))
        # 使用範囲: 既定は全尺・頭合わせ / 再生位置は先頭 / 実使用バンドはクリア
        self.range_slider.set_range(0.0, 1.0)
        self.range_slider.set_playhead(0.0)
        self.range_slider.clear_used_range()
        self._range_override = None
        self._update_range_span()
        for wgt in (self.video_preview, self.video_dim_label, self.range_frame):
            wgt.setVisible(True)
        self._update_video_preview()
        # 使用範囲のループ再生を開始 (以後は常時再生しっぱなし)
        self._vplay_start()

    def _render_video_frame(self, idx):
        """フレーム idx を読み、回転を即時適用して表示する。

        連続再生を軽くするため、前回位置からの前進なら seek せず grab で
        読み飛ばす (cv2 のフレームシークはランダムアクセスだと重い)。
        """
        if self._vid_cap is None or self._vid_info is None:
            return
        w, h, fps, n, dur = self._vid_info
        idx = min(max(0, int(idx)), n - 1)
        gap = idx - self._vid_pos
        if gap < 0 or gap > 12:
            self._vid_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        else:
            for _ in range(gap):
                self._vid_cap.grab()
        ret, frame = self._vid_cap.read()
        self._vid_pos = idx + 1
        if not ret or frame is None:
            return
        frame = apply_rotation_cv2(frame, self.vrot_combo.currentData())
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        pm = QPixmap.fromImage(qimg).scaled(
            max(1, self.video_preview.width() - 2),
            max(1, self.video_preview.height() - 2),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_preview.setPixmap(pm)
        t = idx / max(1e-6, fps)
        self.video_preview.setToolTip(f"frame {idx} / {n}  ({t:.2f}s)")

    def _update_video_preview(self, *_):
        """再生ライン位置のフレームを表示する (回転変更などの再描画用)。"""
        if self._vid_info is None:
            return
        n = self._vid_info[3]
        self._render_video_frame(self.range_slider.values()[2] * max(0, n - 1))

    # --- 常時ループ再生 ---
    def _vplay_bounds(self):
        """再生ループの範囲 (start_f, end_f)。緑バンド優先、無ければ青選択。"""
        if self._vid_info is None:
            return (0, 1)
        n = self._vid_info[3]
        used = self.range_slider.used_range()
        if used is not None:
            s, e = used
        else:
            s, e, _ = self.range_slider.values()
        s_f = max(0, int(round(s * (n - 1))))
        e_f = min(n - 1, int(round(e * (n - 1))))
        return (s_f, max(s_f + 1, e_f))

    def _vplay_start(self, from_frame=None):
        """アンカーを合わせてループ再生を開始/再開する。"""
        if self._vid_cap is None or self._vid_info is None:
            return
        s_f, e_f = self._vplay_bounds()
        if from_frame is None:
            from_frame = s_f
        self._vplay_anchor = min(max(s_f, float(from_frame)), e_f)
        self._vplay_t0 = time.time()
        self._vplay_paused = False
        if not self._vplay_timer.isActive():
            self._vplay_timer.start()

    def _vplay_tick(self):
        """実時間で使用範囲をループ再生し、赤ラインと readout を追従させる。"""
        if (self._vplay_paused or self._vid_cap is None
                or self._vid_info is None or self._vplay_t0 is None):
            return
        if not self.video_preview.isVisible():
            return                       # 別タブ表示中は描画を省く
        fps = self._vid_info[2]
        n = self._vid_info[3]
        s_f, e_f = self._vplay_bounds()
        span = max(1, e_f - s_f)
        elapsed = time.time() - self._vplay_t0
        idx = s_f + (self._vplay_anchor - s_f + elapsed * fps) % span
        self._render_video_frame(idx)
        self.range_slider.set_playhead(idx / max(1, n - 1))   # 赤ライン追従
        self._update_range_span()

    def _vplay_resume_now(self):
        """スクラブ後の自動再開 (現在の再生ラインの位置から続きを再生)。"""
        if self._vid_info is None:
            return
        n = self._vid_info[3]
        self._vplay_start(from_frame=self.range_slider.values()[2] * (n - 1))

    def _on_playhead_dragged(self, frac):
        """赤ラインのドラッグ: 再生を一時停止してスクラブ表示 →
        しばらく操作が無ければその位置から自動で再生再開する。"""
        self._vplay_paused = True
        self._vplay_resume.start()       # 連続ドラッグ中はタイマーが巻き戻る
        if self._vid_info is not None:
            n = self._vid_info[3]
            self._render_video_frame(frac * max(0, n - 1))
        self._update_range_span()

    def _selected_trim(self):
        """使用範囲 (start_sec, end_sec)。全尺なら None。"""
        if self._vid_info is None:
            return None
        dur = self._vid_info[4]
        s_frac, e_frac, _ = self.range_slider.values()
        s, e = s_frac * dur, e_frac * dur
        if s <= 0.05 and e >= dur - 0.05:
            return None
        return (s, e)

    def _range_frames(self):
        """使用範囲を入力フレーム番号 (start_f, end_f) で返す。全尺なら None。

        軌道データの時間軸調整 (applyTimeSlide + レンジフィット) に使う。
        ファイルコピーは作らないため、初期化のやり直しは不要。
        """
        trim = self._selected_trim()
        if trim is None or self._vid_info is None:
            return None
        fps = self._vid_info[2]
        n = self._vid_info[3]
        s_f = max(0, int(round(trim[0] * fps)))
        e_f = min(n - 1, int(round(trim[1] * fps)))
        return (s_f, max(s_f + 1, e_f))

    def _effective_range(self):
        """ワーカーが軌道フィットに使う範囲。緑バンド操作 (override) 優先。"""
        return self._range_override or self._range_frames()

    def _sync_anchor01(self):
        """同期点 (0..1)。rate to data 以外では 0 (無効) を返す。"""
        if self._selected_apply_mode() != "rate to data":
            return 0.0
        return self.sync_anchor_slider.value() / 100.0

    def _on_sync_anchor_changed(self, v):
        """同期点スライダー: 表示更新 + プレビュー/RT へ反映。"""
        self.sync_anchor_val.setText(f"{int(v)}%")
        if self.dm is None:
            return
        if getattr(self, "rt_preview", None):
            self.rt_preview.set_params(sync_anchor=self._sync_anchor01())
            self.rt_preview.refresh_maps()
        self._mark_preview_stale()

    def _update_range_span(self):
        if self._vid_info is None:
            self.range_span_label.setText("")
            return
        dur = self._vid_info[4]
        s_frac, e_frac, p_frac = self.range_slider.values()
        self.range_span_label.setText(
            tr("range_readout", s=s_frac * dur, e=e_frac * dur,
               d=(e_frac - s_frac) * dur, p=p_frac * dur))

    def _set_range_full(self):
        self.range_slider.set_range(0.0, 1.0)
        self._on_range_changed()

    def _sync_range_downstream(self):
        """使用範囲を GPU プレビューとライブプロットへ反映する共通処理。"""
        if self.dm is None:
            return
        if getattr(self, "rt_preview", None):
            self.rt_preview.set_params(use_range=self._effective_range())
            self.rt_preview.refresh_maps()
        self._mark_preview_stale()

    def _on_range_changed(self, *_):
        """青ライン (選択範囲) の変更: 頭合わせの既定動作へ戻して反映する。"""
        self._update_range_span()
        self._range_override = None      # 緑バンドの手動操作を解除
        if self.dm is None:
            return
        # Time 画像の既定 vmin/vmax を範囲フレーム数に追従
        self._maybe_update_time_defaults()
        # rate to data: startpoint も使用範囲の開始フレームへ追従
        if self._selected_apply_mode() == "rate to data":
            rng = self._range_frames()
            self.rate_startpoint_spin.setValue(rng[0] if rng else 0)
        self._sync_range_downstream()

    def _on_used_range_dragged(self, s_frac, e_frac):
        """緑バンド (実使用範囲) のスライド/伸縮をパラメータへ適用する。

        time to data: vmin/vmax が直接この範囲になる。
        rate to data: スライド → startpoint を移動 /
                      伸縮   → baseline と max_range を尺の比率でスケール
                      (レートを一様に上げ下げすると軌道の時間スパンが
                       正確にその倍率で伸縮するため)。
        """
        if self._vid_info is None:
            return
        n = self._vid_info[3]
        s_f = max(0, int(round(s_frac * n)))
        e_f = min(n - 1, int(round(e_frac * n)))
        e_f = max(s_f + 1, e_f)
        prev = getattr(self, "_last_used_band_frames", None)
        self._range_override = (s_f, e_f)
        self._last_used_band_frames = (s_f, e_f)
        mode = self._selected_apply_mode()
        if mode == "time to data":
            # vmin/vmax へ直接適用 (valueChanged 経由で stale マークも入る)
            self.time_vmin_spin.setValue(s_f)
            self.time_vmax_spin.setValue(e_f)
        elif mode == "rate to data":
            # スライド分 → startpoint
            self.rate_startpoint_spin.setValue(s_f)
            # 伸縮分 → レート全体のスケール (baseline / max_range)
            if prev is not None:
                old_span = max(1, prev[1] - prev[0])
                new_span = max(1, e_f - s_f)
                factor = new_span / old_span
                if abs(factor - 1.0) > 0.01:
                    self.rate_baseline_spin.setValue(
                        round(self.rate_baseline_spin.value() * factor, 3))
                    self.rate_maxdev_spin.setValue(
                        round(max(0.001,
                                  self.rate_maxdev_spin.value() * factor), 3))
        self._update_range_span()
        self._sync_range_downstream()

    def _show_used_range_from_data(self):
        """軌道データの実使用範囲 (z min/max) を緑バンドとして表示する。"""
        if self.dm is None or getattr(self.dm, "data", None) is None \
                or self._vid_info is None:
            return
        try:
            z = self.dm.data[:, :, 1]
            n = max(1, self._vid_info[3])
            self.range_slider.set_used_range(float(z.min()) / n,
                                             float(z.max()) / n)
            self._last_used_band_frames = (int(z.min()), int(z.max()))
        except Exception:
            pass

    def initialize_drawmaneuver(self):
        """回転指定があれば先に回転済みコピーを作り、その後 drawManeuver を初期化。
        使用範囲はコピーを作らず、レンダリング/プレビュー時に軌道データの
        時間軸調整 (applyTimeSlide + レンジフィット) で適用する。"""
        src = self.videopath_src or self.videopath
        if not src:
            QMessageBox.warning(self, "Error", "Select a video first.")
            return
        rot = self.vrot_combo.currentData()
        vf = VIDEO_ROTATION_VF.get(rot)
        if not vf:
            self.videopath = src
            self._init_drawmaneuver_now()
            return

        out = rotated_video_path(src, rot)
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            self.log(tr("vrot_reuse", p=out))
            self.videopath = out
            self._init_drawmaneuver_now()
            return
        if shutil.which("ffmpeg") is None:
            QMessageBox.critical(self, "Error", tr("vrot_no_ffmpeg"))
            return

        # 90°系はメタデータ書き換えリマックス (瞬時)。反転のみ再エンコード。
        rot_angle = VIDEO_ROTATION_ANGLE.get(rot)
        self.info_label.setText(tr("vrot_working"))
        self.init_btn.setEnabled(False)
        self.vrot_progress.setValue(0)
        self.vrot_progress.setVisible(rot_angle is None)   # リマックスは一瞬
        self._vrot_worker = VideoRotateWorker(
            src, out, vf, probe_video(src) if rot_angle is None else {},
            rot_angle=rot_angle)
        self._vrot_worker.log_signal.connect(self.log)
        self._vrot_worker.progress.connect(self.vrot_progress.setValue)
        self._vrot_worker.done_signal.connect(self._on_video_rotated)
        self._vrot_worker.start()

    def _on_video_rotated(self, success, out_path):
        self.vrot_progress.setVisible(False)
        self.init_btn.setEnabled(True)
        if not success:
            self.info_label.setText(tr("vrot_failed"))
            QMessageBox.critical(self, "Error", tr("vrot_failed"))
            return
        self.log(f"Rotated video: {out_path}")
        self.videopath = out_path
        self._init_drawmaneuver_now()

    def _init_drawmaneuver_now(self):
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
                # スリット方向・入力実FPS・使用範囲をプレビューに同期
                self.rt_preview.set_params(sd=int(self.dm.scan_direction),
                                           rec_fps=float(self.dm.recfps),
                                           use_range=self._effective_range())
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
        v.addWidget(self._build_section_postproc(type_name))
        return frame

    def _build_section_postproc(self, type_name):
        """適用済み画像に対する破壊的な後処理パネル (反転 / 基準グレー / 回転)。

        レイヤー合成が「これから作る画像」の設定なのに対し、こちらは
        「すでに適用されている 16bit PNG そのもの」を書き換える。
        - 階調反転: 確認ダイアログ → 即書き込み (元に戻すボタンは無い)
        - 基準グレー / 回転: スライダーで即プレビュー → 「適用」で書き込み
        """
        g = self._section_gens[type_name]
        box = QGroupBox()
        self._reg(lambda b=box: b.setTitle(tr("grp_postproc")))
        box.setStyleSheet("QGroupBox { font-size: 11px; color: #a33; }"
                          " QLabel { font-size: 11px; color: #444; }")
        bv = QVBoxLayout(box)
        bv.setContentsMargins(6, 4, 6, 4)
        bv.setSpacing(3)

        # --- 階調反転 (破壊的・即時) ---
        g['pp_invert_btn'] = QPushButton()
        self._reg(lambda b=g['pp_invert_btn']: b.setText(tr("btn_pp_invert")))
        g['pp_invert_btn'].clicked.connect(
            lambda *_, t=type_name: self.postproc_invert(t))
        bv.addWidget(g['pp_invert_btn'])

        # --- 基準グレー (ヒストグラム中間値) ---
        mg_row = QHBoxLayout()
        mg_row.addWidget(self._trlabel("lbl_pp_midgray"))
        g['pp_midgray'] = QSlider(Qt.Horizontal)
        g['pp_midgray'].setRange(5, 95)      # 0.05 .. 0.95
        g['pp_midgray'].setValue(50)
        g['pp_midgray'].valueChanged.connect(
            lambda *_, t=type_name: self._on_postproc_param(t))
        mg_row.addWidget(g['pp_midgray'], 1)
        g['pp_midgray_val'] = QLabel("0.50")
        g['pp_midgray_val'].setMinimumWidth(32)
        mg_row.addWidget(g['pp_midgray_val'])
        bv.addLayout(mg_row)
        mg_hint = self._trlabel("hint_pp_midgray")
        mg_hint.setStyleSheet("color: gray; font-size: 10px;")
        mg_hint.setWordWrap(True)
        bv.addWidget(mg_hint)

        # --- 回転 (サイズ維持で再マッピング) ---
        rot_row = QHBoxLayout()
        rot_row.addWidget(self._trlabel("lbl_pp_rotate"))
        g['pp_rotate'] = QDoubleSpinBox()
        g['pp_rotate'].setRange(-180.0, 180.0)
        g['pp_rotate'].setDecimals(1)
        g['pp_rotate'].setSingleStep(1.0)
        g['pp_rotate'].setSuffix(" °")
        g['pp_rotate'].valueChanged.connect(
            lambda *_, t=type_name: self._on_postproc_param(t))
        rot_row.addWidget(g['pp_rotate'], 1)
        for deg, cap in ((-90.0, "↻90"), (90.0, "↺90"), (180.0, "180°")):
            b = QPushButton(cap)
            b.setMaximumWidth(46)
            b.clicked.connect(lambda *_, t=type_name, d=deg:
                              self._section_gens[t]['pp_rotate'].setValue(d))
            rot_row.addWidget(b)
        bv.addLayout(rot_row)
        rot_hint = self._trlabel("hint_pp_rotate")
        rot_hint.setStyleSheet("color: gray; font-size: 10px;")
        rot_hint.setWordWrap(True)
        bv.addWidget(rot_hint)

        # --- 適用 / リセット ---
        act_row = QHBoxLayout()
        g['pp_apply_btn'] = QPushButton()
        self._reg(lambda b=g['pp_apply_btn']: b.setText(tr("btn_pp_apply")))
        g['pp_apply_btn'].clicked.connect(
            lambda *_, t=type_name: self.postproc_apply(t))
        act_row.addWidget(g['pp_apply_btn'], 1)
        g['pp_reset_btn'] = QPushButton()
        self._reg(lambda b=g['pp_reset_btn']: b.setText(tr("btn_pp_reset")))
        g['pp_reset_btn'].clicked.connect(
            lambda *_, t=type_name: self.postproc_reset(t))
        act_row.addWidget(g['pp_reset_btn'])
        bv.addLayout(act_row)

        g['pp_status'] = QLabel("")
        g['pp_status'].setStyleSheet("color: #c60; font-size: 10px;")
        g['pp_status'].setWordWrap(True)
        g['pp_status'].setVisible(False)
        bv.addWidget(g['pp_status'])

        g['pp_box'] = box
        box.setEnabled(False)        # 適用画像ができるまで無効
        return box

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
        return gray8_to_qpixmap(img8)

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

        未適用の後処理 (基準グレー / 回転) があればそれを乗せた結果を表示する
        (ファイルはまだ書き換えていない)。階調表示モードにも従う。
        """
        if type_name not in self._section_gens:
            return
        label = self._section_gens[type_name]['preview_label']
        if not (path and os.path.exists(path)):
            label.setText(tr("pp_no_image"))
            return
        img16 = read_map16(path)
        if img16 is None:
            label.setText(f"(画像 load 失敗: {os.path.basename(path)})")
            return
        h, w = img16.shape[:2]
        # 大きい画像はプレビュー段階で縮小してから後処理をかける (軽量化)
        target_w = max(label.width(), label.minimumWidth())
        target_h = max(label.height(), label.minimumHeight())
        scale = min(target_w / max(w, 1), target_h / max(h, 1), 1.0)
        if scale < 1.0:
            small = cv2.resize(img16, (max(2, int(round(w * scale))),
                                       max(2, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = img16
        mid, rot = self._pp_pending(type_name)
        small = pp_apply_pending(small, mid, rot)
        pix = gray8_to_qpixmap((small >> 8).astype(np.uint8))
        label.setPixmap(pix)
        note = "" if (abs(mid - 0.5) < 1e-6 and abs(rot) < 1e-6) else \
            f"\n[未適用] 基準グレー={mid:.2f} / 回転={rot:.1f}°"
        label.setToolTip(
            f"ロード画像: {os.path.basename(path)}\n({w}×{h}){note}")

    # --- 適用画像の後処理 (破壊的) ---
    def _pp_pending(self, type_name):
        """セクションの未適用後処理 (midgray, rotate_deg) を返す。"""
        g = self._section_gens.get(type_name, {})
        if 'pp_midgray' not in g:
            return 0.5, 0.0
        return g['pp_midgray'].value() / 100.0, float(g['pp_rotate'].value())

    def _pp_is_neutral(self, type_name):
        mid, rot = self._pp_pending(type_name)
        return abs(mid - 0.5) < 1e-6 and abs(rot) < 1e-6

    def _refresh_section_view(self, type_name):
        """セクションのプレビューを現在の状態で再描画 (適用画像 > レイヤー合成)。"""
        path = getattr(self, f"{type_name}_img_path", None)
        if path and os.path.exists(path):
            self._show_loaded_image_in_preview(type_name, path)
        else:
            self._update_section_preview(type_name)

    def _update_postproc_state(self, type_name):
        """後処理パネルの有効/無効と「未適用」表示を更新する。"""
        g = self._section_gens.get(type_name, {})
        if 'pp_box' not in g:
            return
        path = getattr(self, f"{type_name}_img_path", None)
        has_img = bool(path and os.path.exists(path))
        g['pp_box'].setEnabled(has_img)
        mid, rot = self._pp_pending(type_name)
        g['pp_midgray_val'].setText(f"{mid:.2f}")
        pending = has_img and not self._pp_is_neutral(type_name)
        g['pp_status'].setText(tr("pp_pending") if pending else "")
        g['pp_status'].setVisible(pending)
        g['preview_label'].setStyleSheet(
            "QLabel { background: #222; color: #888; border: 2px solid #c60; }"
            if pending else
            "QLabel { background: #222; color: #888; border: 1px solid #555; }")

    def _on_postproc_param(self, type_name):
        """基準グレー / 回転スライダーの変更 → プレビューのみ更新。"""
        self._update_postproc_state(type_name)
        self._refresh_section_view(type_name)

    def postproc_reset(self, type_name):
        """未適用の後処理パラメータを既定値に戻す (ファイルは触らない)。"""
        g = self._section_gens.get(type_name, {})
        if 'pp_midgray' not in g:
            return
        for w, v in ((g['pp_midgray'], 50), (g['pp_rotate'], 0.0)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self._on_postproc_param(type_name)

    def _pp_write(self, type_name, img16, what):
        """後処理結果を適用画像に上書きし、依存する表示をすべて更新する。"""
        path = getattr(self, f"{type_name}_img_path", None)
        try:
            if not cv2.imwrite(path, img16):
                raise IOError(f"cv2.imwrite failed: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Post-process Error", str(e))
            self.log(f"[ERROR] postproc {type_name}: {e}")
            return False
        self.log(f"[postproc] {type_name}: {what} → {os.path.basename(path)}")
        self.postproc_reset(type_name)          # 適用済みなので保留値をクリア
        self._wire_loaded_image(type_name, path)
        # 適用方法の基準画像を書き換えたら、対になるマップも作り直す
        mode = self._selected_apply_mode()
        if (type_name == "time" and mode == "time to data") or \
           (type_name == "rate" and mode == "rate to data"):
            self._sync_derived_maps()
        self._mark_preview_stale()
        return True

    def postproc_invert(self, type_name):
        """適用画像の階調を反転して上書きする (破壊的・確認あり)。"""
        path = getattr(self, f"{type_name}_img_path", None)
        if not (path and os.path.exists(path)):
            QMessageBox.warning(self, "Error", tr("pp_no_image"))
            return
        if QMessageBox.question(
                self, tr("pp_confirm_title"),
                tr("pp_confirm_invert", t=type_name.capitalize(),
                   f=os.path.basename(path)),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        img16 = read_map16(path)
        if img16 is None:
            QMessageBox.critical(self, "Error", f"Could not read {path}")
            return
        self._pp_write(type_name, pp_invert(img16), "invert")

    def postproc_apply(self, type_name):
        """未適用の基準グレー / 回転を適用画像へ書き込む (破壊的・確認あり)。"""
        path = getattr(self, f"{type_name}_img_path", None)
        if not (path and os.path.exists(path)):
            QMessageBox.warning(self, "Error", tr("pp_no_image"))
            return
        mid, rot = self._pp_pending(type_name)
        if self._pp_is_neutral(type_name):
            QMessageBox.information(self, tr("grp_postproc"), tr("pp_nothing"))
            return
        ops = []
        if abs(mid - 0.5) >= 1e-6:
            ops.append(tr("pp_op_midgray", v=mid))
        if abs(rot) >= 1e-6:
            ops.append(tr("pp_op_rotate", v=rot))
        if QMessageBox.question(
                self, tr("pp_confirm_title"),
                tr("pp_confirm_apply", t=type_name.capitalize(),
                   f=os.path.basename(path), ops="\n".join(ops)),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        img16 = read_map16(path)
        if img16 is None:
            QMessageBox.critical(self, "Error", f"Could not read {path}")
            return
        self._pp_write(type_name, pp_apply_pending(img16, mid, rot),
                       f"midgray={mid:.2f} rotate={rot:.1f}deg")

    # --- 階調表示モード ---
    def _on_colormap_toggled(self, checked):
        """黄(255)–青(0) 表示の ON/OFF。表示のみでファイルには影響しない。"""
        global COLOR_MODE
        COLOR_MODE = "yellowblue" if checked else "gray"
        for t in self._section_gens:
            self._refresh_section_view(t)
        for th in getattr(self, "_map_thumbs", {}).values():
            th.refresh_color_mode()
        self.log(f"[view] tone display: {COLOR_MODE}")

    # --- 入力映像の回転 ---
    def _on_video_rotation_changed(self, *_):
        """回転の選択が変わったらプレビューへ即時反映 + 再初期化を促す。"""
        self._update_video_preview()
        if self.dm is not None:
            self.info_label.setText(tr("vrot_reinit"))
            self.init_btn.setEnabled(True)

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

        pw, ph = self._plot_inches_for(self.preview_2dplot_label)
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
            plot_w_inc=pw, plot_h_inc=ph,
            gif_width=min(720, max(320, self.preview_3d_label.width())),
            use_range=self._effective_range(),
            sync_anchor=self._sync_anchor01(),
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

    def _plot_inches_for(self, widget):
        """widget の現在の表示領域に合わせた 2D プロットの図サイズ (インチ)。

        図のアスペクト比を表示領域と一致させることで、KeepAspectRatio 縮小時の
        余白 (レターボックス) を最小化する。面積は既定 (5×9=45in²) に固定し、
        文字サイズと線の太さの見た目を既定プロットと揃える。
        """
        try:
            w = max(1, widget.width())
            h = max(1, widget.height())
        except Exception:
            return None, None
        if w < 40 or h < 40:            # レイアウト確定前は既定に任せる
            return None, None
        area = 45.0                      # 既定の plot_w_inc(5) × plot_h_inc(9)
        aspect = w / h
        w_inc = (area * aspect) ** 0.5
        h_inc = (area / aspect) ** 0.5
        # 極端な縦横比では文字が潰れるためクランプ (アスペクトは多少崩れても
        # 余白は現状より大きくならない)
        w_inc = min(16.0, max(3.0, w_inc))
        h_inc = min(16.0, max(3.0, h_inc))
        return round(w_inc, 2), round(h_inc, 2)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, "_resize_timer", None):
            self._resize_timer.start()   # 連続リサイズ中は巻き戻る

    def _maybe_replot_on_resize(self):
        """リサイズ確定後、2D プロット領域のアスペクト比が 10% 以上変わって
        いたらライブプロットを再生成する (図サイズを表示領域に追従させる)。"""
        if not self._pipeline_ready() or self._live2d_last_aspect is None:
            return
        pw, ph = self._plot_inches_for(self.live2d_thumb)
        if pw and ph and abs(pw / ph - self._live2d_last_aspect) \
                / self._live2d_last_aspect > 0.10:
            self._schedule_live3d()

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
        # 2D プロットの図サイズをライブ表示領域のアスペクト比に合わせる
        # (3D はネイティブ比率のまま — mplot3d の描画枠は歪ませない)
        pw, ph = self._plot_inches_for(self.live2d_thumb)
        self._live2d_last_aspect = (pw / ph) if (pw and ph) else None
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
            plot_w_inc=pw, plot_h_inc=ph,
            gif_width=min(720, max(320, self.live3d_label.width())),
            use_range=self._effective_range(),
            sync_anchor=self._sync_anchor01(),
        )
        self._live3d_worker.done_signal.connect(self._on_live3d_done)
        self._live3d_worker.start()

    def _on_live3d_done(self, success, plot2d, gif):
        self._live3d_busy = False
        # 軌道データが実際に参照している入力時間範囲を緑バンドで表示
        if success:
            self._show_used_range_from_data()
        # 2D プロット (左カラム・赤ライン付きサムネイル)
        if success and plot2d and os.path.exists(plot2d):
            self.live2d_thumb.set_map(plot2d)
            # 赤ラインの可動範囲を時間軸のデータ領域 (プロット内割合) に合わせる
            frac = getattr(self.dm, "plot2d_time_axis_frac", None)
            if frac:
                self.live2d_thumb.set_playhead_range(*frac)
        if success and gif and os.path.exists(gif):
            if self._live3d_movie is not None:
                try:
                    self._live3d_movie.stop()
                except Exception:
                    pass
                self.live3d_label.setMovie(None)
            # スケール寸法の決定 (ラベル枠内・アスペクト比保持)
            native = QImageReader(gif).size()
            box = self.live3d_label.size()
            if native.width() > 0 and native.height() > 0:
                scale = min(box.width() / native.width(),
                            box.height() / native.height())
                scaled = QSize(max(1, int(native.width() * scale)),
                               max(1, int(native.height() * scale)))
            else:
                scaled = box
            # 同期表示用に全フレームを先読み (QMovie の jumpToFrame は GIF で
            # 任意フレームへ飛べないため、setPixmap による直接表示で同期する)
            frames = []
            reader = QImageReader(gif)
            while True:
                img = reader.read()
                if img.isNull():
                    break
                frames.append(QPixmap.fromImage(img).scaled(
                    scaled, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                if not reader.supportsAnimation() or len(frames) > 400:
                    break
            self._live3d_frames = frames
            self._last_sync_frac = -1.0     # 新 GIF で強制再同期

            movie = QMovie(gif)
            movie.setCacheMode(QMovie.CacheNone)
            if movie.isValid():
                movie.setScaledSize(scaled)
                self._live3d_movie = movie
                rt = getattr(self, "rt_preview", None)
                if rt is not None and rt._F and frames:
                    # 同期モード: GIF は自走させず、次の sync tick が
                    # 再生位置に対応するフレームを setPixmap する
                    pass
                else:
                    # 自走モード (RT 未構築): 従来どおり QMovie 再生
                    self.live3d_label.setMovie(movie)
                    movie.frameChanged.connect(self._on_live3d_frame)
                    movie.start()
            self.live3d_status.setText("")
        else:
            self.live3d_status.setText("")
        # 実行中に編集が入っていたら追いかけ再生成
        if self._live3d_pending:
            self._live3d_pending = False
            self._schedule_live3d()

    def _on_live3d_frame(self, frame_idx):
        """[RT未構築時のみ] GIF 自走に合わせて赤ラインを動かすフォールバック。

        GPU リアルタイムプレビューのボリューム構築後は、再生位置を
        マスタークロックとする _sync_plots_tick が権限を持つ (GIF は
        jumpToFrame で従属させるため、ここでは何もしない)。
        """
        rt = getattr(self, "rt_preview", None)
        if rt is not None and rt._F:
            return
        movie = self._live3d_movie
        if movie is None:
            return
        n = max(1, movie.frameCount())
        frac = (frame_idx + 0.5) / n
        for th in getattr(self, "_map_thumbs", {}).values():
            th.set_playhead(frac)
        # 2D プロットにも赤ラインを左→右へスライド表示
        self.live2d_thumb.set_playhead(frac)

    def _sync_plots_tick(self):
        """GPU 映像の再生位置をマスタークロックとして、2D/3D プロットと
        グレー画像の赤ラインを同期させる (映像のゆっくりした時間進行に追従)。

        - 赤ライン (2D プロット + Space/Time/Rate サムネイル): 再生位置の割合で移動
        - 3D GIF: 一時停止して再生位置に対応するフレームへ jumpToFrame
          (速度変更・スクラブ・一時停止もすべて追従する)
        """
        rt = getattr(self, "rt_preview", None)
        if rt is None or not rt._F or not getattr(self, "live3d_group", None):
            return
        frac = min(1.0, rt._t_out / max(1, rt.time_size - 1))
        if abs(frac - getattr(self, "_last_sync_frac", -1.0)) <= 1e-4:
            return
        self._last_sync_frac = frac
        for th in getattr(self, "_map_thumbs", {}).values():
            th.set_playhead(frac)
        self.live2d_thumb.set_playhead(frac)
        # 3D GIF: 自走 QMovie を止め、先読みフレームを直接表示して同期
        frames = getattr(self, "_live3d_frames", None)
        if frames:
            if self._live3d_movie is not None and \
                    self._live3d_movie.state() != QMovie.NotRunning:
                self._live3d_movie.stop()
                self.live3d_label.setMovie(None)
            target = int(round(frac * (len(frames) - 1)))
            self.live3d_label.setPixmap(frames[target])

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
                # 導出 time は 3D プロット下のサムネイルにのみ反映する。
                # Time セクションの生成画面 (プレビュー/パラメータ) はユーザーの
                # 編集領域なので上書きしない。
                self.time_img_path = path
                if "time" in getattr(self, "_map_thumbs", {}):
                    self._map_thumbs["time"].set_map(path)
                self.log(f"[sync] time を rate から自動生成 (vrange {vmin_i}-{vmax_i}, "
                         f"サムネイルのみ更新)")
        except Exception as e:
            self.log(f"[WARN] map sync failed: {e}")
        finally:
            self._syncing_maps = False

    def _wire_loaded_image(self, img_type, path):
        """select_image() の "画像情報表示 + パラメータ抽出 + プレビュー" 共通処理"""
        getattr(self, f"{img_type}_label").setText(f"Selected: {path}")
        # 適用画像が差し替わったので、未適用の後処理パラメータは破棄する
        self.postproc_reset(img_type)
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
        self._update_postproc_state(img_type)

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
                maxdev=self.rate_maxdev_spin.value(),
                sync_anchor=self._sync_anchor01())
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

    def _move_live3d(self, to_preview):
        """軌道プロットライブビュー (live3d_group) をタブ間で移動する。

        タブ1 (入力・画像) では設定の右 3/4 に、統合タブ (プレビュー・出力)
        では映像ビューの下に「そのまま」併置する (ウィジェットは単一)。
        """
        g = getattr(self, "live3d_group", None)
        if g is None:
            return
        if to_preview:
            self._t2_top_row.removeWidget(g)
            self._live3d_slot.addWidget(g)
            g.setVisible(self.dm is not None and
                         not self.video_only_chk.isChecked())
        else:
            self._live3d_slot.removeWidget(g)
            self._t2_top_row.insertWidget(1, g, 3)
            g.setVisible(self.dm is not None)

    def _on_video_only_toggled(self, checked):
        """「映像ビューのみ表示」: 統合タブで軌道プロットを隠し映像を最大化。"""
        if self.tabs.currentIndex() == 1:
            self.live3d_group.setVisible(self.dm is not None and not checked)

    def _show_gpu_view(self, *_):
        """レンダリング結果表示から GPU リアルタイムプレビューへ戻す。"""
        if getattr(self, "rendered_preview", None):
            self.rendered_preview.stop()
            self.rendered_preview.setVisible(False)
        if getattr(self, "rt_group", None):
            self.rt_group.setVisible(True)

    def _on_tab_changed(self, idx):
        """統合タブ (index 1) 表示中だけ RT 再生 + ログ表示。
        軌道プロットライブビューはタブに合わせて移動する。"""
        if getattr(self, "log_box", None):
            self.log_box.setVisible(idx == 1)
        self._move_live3d(idx == 1)
        # タブ間で 2D プロット領域の形が変わるため、必要ならアスペクト再調整
        if getattr(self, "_resize_timer", None):
            self._resize_timer.start()
        rt = getattr(self, "rt_preview", None)
        if not rt:
            return
        if idx == 1 and (not getattr(self, "rt_group", None)
                         or self.rt_group.isVisible()):
            rt.start()
        else:
            rt.stop()

    def _audio_export_settings(self):
        """書き出しに使う音声設定 = GPU プレビューの音声設定そのもの。"""
        if getattr(self, "rt_preview", None):
            return self.rt_preview.audio_settings()
        return {"enabled": False, "mode": "grain", "voices": 7, "grain_ms": 90}

    def _update_audio_out_info(self, *_):
        """出力行の「音声出力: …」表示をプレビュー設定に追従させる。"""
        if not hasattr(self, "audio_out_info"):
            return
        a = self._audio_export_settings()
        if a["enabled"]:
            self.audio_out_info.setText(
                tr("audio_out_on", m=a["mode"], v=a["voices"]))
        else:
            self.audio_out_info.setText(tr("audio_out_off"))

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

        # stdout に流れる imgtrans の進捗% をプログレスバーへ転送
        self.render_progress.setValue(0)
        if self._orig_stdout is None:
            self._orig_stdout = sys.stdout
            sys.stdout = StdoutPercentTee(self._orig_stdout,
                                          self._render_pct.emit)

        # 出力FPS を drawManeuver に反映 (最終尺 = 時間方向サイズ ÷ 出力FPS)
        try:
            self.dm.outfps = self._out_fps()
            self.log(f"Output FPS: {self.dm.outfps}")
        except Exception as e:
            self.log(f"[WARN] could not set outfps: {e}")

        aset = self._audio_export_settings()   # 音声 = プレビューの設定を使用
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
            rate_maxdev=maxdev, rate_baseline=baseline, rate_startpoint=startpoint,
            audio_out=aset["enabled"], audio_mode=aset["mode"],
            audio_voices=aset["voices"], audio_grain_ms=aset["grain_ms"],
            use_range=self._effective_range(),
            sync_anchor=self._sync_anchor01(),
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
        # stdout の進捗検出を解除
        if getattr(self, "_orig_stdout", None) is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        self.render_progress.setValue(100 if success else 0)
        if success:
            self.render_completed = True
            self.log(" Rendering completed.")
            self.update_ui_state("rendered")
            self._show_rendered_preview(video_path)
        else:
            self.log("Rendering failed.")

    def _show_rendered_preview(self, video_path):
        """本レンダリング完了後、映像エリアを GPU プレビューから
        書き出し結果のプレイヤーへ差し替える (Rebuild で GPU ビューに戻る)。"""
        if not (video_path and os.path.exists(video_path)):
            self.log("[preview] no output video found to preview.")
            return
        if getattr(self, "rt_preview", None):
            self.rt_preview.stop()
        if getattr(self, "rt_group", None):
            self.rt_group.setVisible(False)
        self.rendered_preview.stop()
        self.rendered_preview.load(video_path)
        self.rendered_preview.setVisible(True)
        self.log(f"[preview] rendered video: {os.path.basename(video_path)}")

    def log(self, text):
        self.log_window.append(str(text))
        self.log_window.ensureCursorVisible()


# ======== Main entry ========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = IMGTransApp()
    win.show()
    sys.exit(app.exec_())
