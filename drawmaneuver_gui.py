"""drawmaneuver_gui — drawManeuver メソッドチェーン エディタ (2026-08)

Shape_of_time_flow.py が「グレー画像 → マニューバデータ」を作るのに対し、
本アプリは imgtrans の drawManeuver が持つ **クラスメソッドそのもの** を
モジュールとして next-to-next に積み上げてマニューバデータを組み立てる。

  [入力映像] → [ステップ1: addXxx] → [ステップ2: applyXxx] → … → [2Dプロット]
                                                              ├→ GPU プレビュー
                                                              ├→ レンダリング
                                                              └→ 実行用 .py 書き出し

設計の要点:
- メソッド一覧は **イントロスペクションで自動生成** する。imgtrans に
  メソッドが増えれば、このファイルを触らなくても選択肢に現れる。
- 引数 UI も inspect.signature から自動生成 (bool/int/float/str/リテラル)。
- ステップは (メソッド名, kwargs, 有効フラグ) の列。編集のたびに
  dm.data を空に戻して全ステップを再実行する (= 常に決定的)。
  直前のプレフィックスが変わっていなければスナップショットから再開する。
- 入力映像まわりの UI (プレビュー/スリット方向ガイド/回転) は
  Shape_of_time_flow.py の実装をトレースしている。
"""

import os
import sys
import re
import ast
import copy
import time
import json
import shutil
import glob
import html
import inspect
import subprocess
import traceback
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QComboBox, QTextEdit, QCheckBox, QMessageBox, QSpinBox, QDoubleSpinBox,
    QFrame, QGroupBox, QScrollArea, QSplitter, QProgressBar, QSizePolicy,
    QSlider, QLineEdit, QFileDialog, QToolButton, QDialog,
    QListWidget, QListWidgetItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QSize, QUrl
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QIcon,
                         QMovie, QImageReader, QFont, QTextCursor,
                         QTextCharFormat, QTextDocument, QDesktopServices)

import numpy as np
import cv2

# matplotlib は imgtrans の import 前に Agg 固定 (QtAgg だとワーカースレッドの
# 描画がメインスレッドの Qt と競合して UI が固まる)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap

from imgtrans import drawManeuver

try:
    from realtime_preview import RealtimePreviewWidget, MAP_T_RES, MAP_S_RES
    _HAS_RT_PREVIEW = True
except Exception:
    RealtimePreviewWidget = object
    MAP_T_RES = MAP_S_RES = 512
    _HAS_RT_PREVIEW = False


# ======== i18n ========
LANG = os.environ.get("STF_LANG", "ja").strip().lower()
if LANG not in ("ja", "en"):
    LANG = "ja"

TR = {
    "window_title": {"ja": "drawManeuver GUI", "en": "drawManeuver GUI"},
    "grp_setup": {"ja": "入力 (Setup)", "en": "Setup"},
    "lang_label": {"ja": "言語 / Language:", "en": "Language / 言語:"},
    "tip_drop_video": {"ja": "またはここに Finder から動画をドラッグ＆ドロップ", "en": "…or drag & drop a video here from Finder"},
    "btn_select_video": {"ja": "動画を選択 / Select Video File",
                          "en": "Select Video File"},
    "no_video": {"ja": "動画が未選択です", "en": "No video file selected"},
    "drop_placeholder": {"ja": "🎬 ここに動画をドラッグ＆ドロップ\n(または上のボタンで選択)",
                          "en": "🎬 Drop a video here\n(or use the button above)"},
    "btn_play": {"ja": "▶ 再生", "en": "▶ Play"},
    "btn_pause": {"ja": "⏸ 停止", "en": "⏸ Pause"},
    "tip_play": {"ja": "入力映像をこの場で再生 (使用範囲があればその中をループ)",
                 "en": "Play the input here (loops inside the used range if any)"},
    "btn_reveal_video": {"ja": "Finder で表示 📁", "en": "Reveal in Finder 📁"},
    "tip_reveal_video": {"ja": "入力映像のファイルを Finder で選択状態にして開く",
                          "en": "Reveal the input video file in Finder"},
    "lbl_out_size": {"ja": "出力 {w} × {h} px　{f} frames",
                      "en": "Output {w} × {h} px　{f} frames"},
    "chk_vertical": {"ja": "縦スリット (Vertical)", "en": "Vertical slit"},
    "slit_h": {"ja": "スリット方向: 横 (horizontal)", "en": "Slit: horizontal"},
    "slit_v": {"ja": "スリット方向: 縦 (vertical)", "en": "Slit: vertical"},
    "btn_initialize": {"ja": "初期化 / Initialize", "en": "Initialize"},
    "video_not_init": {"ja": "動画情報: (未初期化)", "en": "Video info: (not initialized)"},
    "vid_info": {"ja": "{w}×{h}  |  {n} frames  |  {fps:.2f} fps  |  {dur:.2f} 秒",
                  "en": "{w}×{h}  |  {n} frames  |  {fps:.2f} fps  |  {dur:.2f} s"},
    "lbl_video_rotate": {"ja": "入力映像の回転:", "en": "Input video rotation:"},
    "vrot_none": {"ja": "なし (0°)", "en": "None (0°)"},
    "vrot_cw90": {"ja": "右90° (時計回り)", "en": "90° clockwise"},
    "vrot_180": {"ja": "180°", "en": "180°"},
    "vrot_ccw90": {"ja": "左90° (反時計回り)", "en": "90° counter-clockwise"},
    "vrot_hflip": {"ja": "左右反転", "en": "Flip horizontal"},
    "vrot_vflip": {"ja": "上下反転", "en": "Flip vertical"},
    "hint_video_rotate": {
        "ja": "(90°系はメタデータ書き換えのみ = 瞬時・無劣化。反転のみ再エンコード)",
        "en": "(90° variants remux metadata only; flips re-encode)"},
    "vrot_working": {"ja": "映像を回転中… (ffmpeg)", "en": "Rotating video… (ffmpeg)"},
    "vrot_failed": {"ja": "映像の回転に失敗しました", "en": "Video rotation failed"},
    "vrot_reinit": {"ja": "入力設定を変更しました → 「初期化」をもう一度",
                     "en": "Input changed → press Initialize again"},
    "lbl_outfps": {"ja": "出力FPS:", "en": "Output FPS:"},
    # --- ステップチェーン ---
    "grp_chain": {"ja": "メソッドチェーン (Maneuver Steps)", "en": "Method chain"},
    "lbl_category": {"ja": "種別:", "en": "Category:"},
    "btn_help_methods": {"ja": "?", "en": "?"},
    "tip_help_methods": {"ja": "メソッド一覧とヘルプを別ウィンドウで開く",
                          "en": "Open the method browser / help window"},
    "help_title": {"ja": "メソッド ヘルプ", "en": "Method help"},
    "help_hint": {"ja": "上の種別をクリックすると絞り込み (もう一度押すと解除 = 全メソッド)",
                  "en": "Click a category to filter (click again to clear = all methods)"},
    "btn_add_from_help": {"ja": "＋ このメソッドをチェーンに追加",
                           "en": "+ Add this method to the chain"},
    "lbl_workdir_files": {"ja": "作業フォルダの中身:", "en": "Working folder:"},
    "btn_dir_back": {"ja": "◀ 上へ", "en": "◀ Up"},
    "tip_dir_click": {"ja": "クリック: ファイルは既定のアプリで開く / フォルダは中に入る",
                       "en": "Click: open file with default app / enter folder"},
    "dir_empty": {"ja": "(空のフォルダ)", "en": "(empty)"},
    "lbl_method": {"ja": "メソッド:", "en": "Method:"},
    "btn_add_step": {"ja": "＋ ステップを追加", "en": "+ Add step"},
    "cat_add": {"ja": "Add 系 (データを作る/継ぎ足す)", "en": "Add (create/append)"},
    "cat_apply": {"ja": "Apply 系 (全体に効果を適用)", "en": "Apply (transform all)"},
    "cat_data": {"ja": "データ操作 (整列/抽出/チェック)", "en": "Data ops"},
    "cat_expand": {"ja": "拡張系 (空間サイズを変える・置ける位置に制約あり)",
                    "en": "Expand (changes spatial size — position constrained)"},
    "cat_other": {"ja": "その他 (出力/解析)", "en": "Other (output/analysis)"},
    "chain_empty": {"ja": "(ステップがありません。下の「＋ ステップを追加」から)",
                     "en": "(no steps yet — use “+ Add step” below)"},
    "step_disabled": {"ja": "無効", "en": "off"},
    "tip_up": {"ja": "上へ", "en": "Move up"},
    "tip_down": {"ja": "下へ", "en": "Move down"},
    "tip_dup": {"ja": "複製", "en": "Duplicate"},
    "tip_del": {"ja": "削除", "en": "Delete"},
    # --- プロット / 実行 ---
    "grp_plot": {"ja": "2D プロット (自動更新)", "en": "2D plot (live)"},
    "plot_waiting": {"ja": "(初期化してステップを追加すると表示されます)",
                      "en": "(shown after Initialize + steps)"},
    "plot_building": {"ja": "再構築中…", "en": "rebuilding…"},
    "plot_error": {"ja": "エラー: {e}", "en": "Error: {e}"},
    "lbl_data_info": {"ja": "出力 data.shape = ({f}, {s}, 2)   —   {f} frames × {s} slits"
                            "   |   time {zmin:.0f}–{zmax:.0f}"
                            "   |   space {smin:.0f}–{smax:.0f}",
                       "en": "output data.shape = ({f}, {s}, 2)   —   {f} frames × {s} slits"
                             "   |   time {zmin:.0f}–{zmax:.0f}"
                             "   |   space {smin:.0f}–{smax:.0f}"},
    "grp_realtime": {"ja": "リアルタイム軸間変換プレビュー (GPU)",
                      "en": "Realtime axis-transform preview (GPU)"},
    "btn_render": {"ja": "レンダリング開始 / Start Rendering", "en": "Start Rendering"},
    "btn_export": {"ja": "実行用 .py を書き出し", "en": "Export .py"},
    "btn_zcheck": {"ja": "zPointCheck を実行", "en": "Run zPointCheck"},
    "btn_open_workdir": {"ja": "作業フォルダを開く 📁",
                          "en": "Open working folder 📁"},
    "tip_open_workdir": {
        "ja": "初期化時に作られた作業ディレクトリを Finder で開く\n(レンダリング結果や書き出した .py もここに入る)",
        "en": "Reveal the working directory created at initialization\n(renders and exported .py land here)"},
    "busy_now": {"ja": "処理中です。完了してから実行してください。",
                 "en": "Busy — wait until the current job finishes."},
    "tip_plot_dblclick": {"ja": "ダブルクリックで原寸表示",
                           "en": "Double-click to view full size"},
    "lbl_still_split": {"ja": "1枚データの出力:", "en": "Single-frame output:"},
    "still_split_one": {"ja": "1 枚にまとめて出力",
                         "en": "One image for the whole span"},
    "still_split_time": {"ja": "時間で分割して複数枚出力",
                          "en": "Split by time into multiple images"},
    "lbl_split_minutes": {"ja": "1 枚あたり", "en": "per image"},
    "unit_minutes": {"ja": "分", "en": "min"},
    "hint_still_split": {
        "ja": "1 フレームのデータは映像の全尺が 1 枚に積み重なる。"
              "長い素材では時間で区切って複数枚に分けると扱いやすい。",
        "en": "A single frame stacks the whole clip into one image. "
              "Splitting it by time is easier to handle for long sources."},
    "lbl_scan_step": {"ja": "間引き 1/", "en": "step 1/"},
    "tip_scan_step": {
        "ja": "時間軸の間引き。出力幅 = 区間のフレーム数 ÷ この値。\n"
              "1 なら 1 フレーム = 1 ピクセル (間引きなし)。",
        "en": "Time-axis decimation. Output width = frames in the chunk / this.\n"
              "1 means one frame per pixel (no decimation)."},
    "still_split_tail": {"ja": "  (最後の 1 枚だけ {last})",
                          "en": "  (last image is {last})"},
    "still_split_info": {"ja": "全 {total} → 各 {each} × {n} 枚",
                          "en": "{total} total → {each} each × {n} images"},
    "still_split_applied": {
        "ja": "1枚データを時間で分割: 全 {total} を {n} 枚に分けて出力します",
        "en": "Splitting the single frame by time: {total} into {n} images"},
    "still_chunk_run": {"ja": "  [{i}/{n}] {s} – {e}  ({w} スリット)",
                         "en": "  [{i}/{n}] {s} – {e}  ({w} slits)"},
    "still_default": {
        "ja": "1 フレームのみのため、出力形式を連番画像に切り替えました。",
        "en": "Single frame — switched the output format to image sequence."},
    "video_restored": {
        "ja": "フレームが複数になったため、出力形式を動画へ戻しました。",
        "en": "Multiple frames — switched the output format back to video."},
    "btn_full2d": {"ja": "詳細 2D プロット (PNG)", "en": "Full 2D plot (PNG)"},
    "btn_full3d": {"ja": "3D プロットアニメーション (MP4)",
                    "en": "3D plot animation (MP4)"},
    "plot3d_running": {"ja": "3D プロットを作成中… (時間がかかります)",
                        "en": "Building the 3D plot… (this takes a while)"},
    "plot3d_done": {"ja": "3D プロットを出力しました: {p}",
                     "en": "3D plot written: {p}"},
    "plot3d_fail": {"ja": "3D プロットの出力に失敗しました",
                     "en": "Failed to write the 3D plot"},
    "plot3d_still": {"ja": "3D プロット (1 フレーム) を作成中…",
                      "en": "Building the 3D plot (single frame)…"},
    "plot_single_note": {
        "ja": "1 フレームのみのデータです。2D プロットは意味を持たないため "
              "3D プロットを表示しています。",
        "en": "Single-frame data. The 2D plot is meaningless here, "
              "so the 3D plot is shown instead."},
    "lbl_log": {"ja": "Log:", "en": "Log:"},
    "need_init": {"ja": "先に動画を選択して「初期化」してください。",
                   "en": "Select a video and press Initialize first."},
    "need_steps": {"ja": "マニューバデータがありません (Add 系のステップが必要です)。",
                    "en": "No maneuver data — add an Add-category step first."},
    "export_done": {"ja": "書き出しました: {p}", "en": "Exported: {p}"},
    "export_fail": {"ja": "書き出しに失敗: {p} ({e})",
                     "en": "Export failed: {p} ({e})"},
    "rendering": {"ja": "レンダリング中…", "en": "Rendering…"},
    "render_done": {"ja": "レンダリング完了: {p}", "en": "Rendering done: {p}"},
    "render_no_output": {
        "ja": "[ERROR] 出力ファイルが作成されませんでした。"
              "time に負の値/範囲超過が残っていないか (zPointCheck) を確認してください。",
        "en": "[ERROR] No output file was produced. "
              "Check for negative/out-of-range time (zPointCheck)."},
    "btn_save_preview": {"ja": "プレビューを簡易保存 ⤓",
                          "en": "Quick-save preview ⤓"},
    "tip_save_preview": {
        "ja": "GPU プレビューの見た目そのままを、作業ディレクトリへ\n"
              "確認用の動画として書き出す (プレビュー解像度・音声なし)",
        "en": "Save the GPU preview as-is to the working directory\n"
              "as a check video (preview resolution, no audio)"},
    "save_preview_running": {"ja": "プレビューを書き出し中… {i}/{n}",
                              "en": "Saving preview… {i}/{n}"},
    "save_preview_done": {"ja": "プレビューを保存しました: {p}",
                           "en": "Preview saved: {p}"},
    "save_preview_fail": {"ja": "プレビューの保存に失敗しました",
                           "en": "Failed to save the preview"},
    "save_preview_need": {
        "ja": "先に GPU プレビューを構築してください",
        "en": "Build the GPU preview first"},
    "btn_open_output": {"ja": "書き出した動画を再生 ▶",
                         "en": "Play rendered video ▶"},
    "btn_open_output_dir": {"ja": "書き出したフォルダを開く 📁",
                             "en": "Open output folder 📁"},
    "open_output": {"ja": "既定のプレーヤーで開きます: {p}",
                     "en": "Opening with the default player: {p}"},
    "open_output_gone": {"ja": "書き出したファイルが見つかりません: {p}",
                          "en": "Rendered file not found: {p}"},
    "render_failed": {"ja": "レンダリングに失敗しました (ログ参照)",
                       "en": "Rendering failed (see log)"},
    "param_required": {"ja": "必須", "en": "required"},
    "tip_info": {"ja": "このメソッドの説明を表示", "en": "Show method description"},
    "no_doc": {"ja": "(説明がありません)", "en": "(no description)"},
    "chk_auto_update": {"ja": "自動更新", "en": "Auto update"},
    "btn_update_now": {"ja": "更新", "en": "Update"},
    "btn_update_dirty": {"ja": "更新 (変更あり)", "en": "Update (stale)"},
    "lbl_preview_scan": {"ja": "プレビュー分割:", "en": "Preview slits:"},
    "tip_preview_scan": {
        "ja": ("チェーンの試算に使うスリット本数。2D プロットは既定 19 本しか"
               "描かないため、少ない本数で計算してプレビューを高速化する。"
               "レンダリング時は本番のスリット数で計算し直す。"),
        "en": "Slit count used for preview computation; render re-runs at full width"},
    "tip_undo": {"ja": "元に戻す (Cmd+Z)", "en": "Undo (Cmd+Z)"},
    "tip_redo": {"ja": "やり直す (Shift+Cmd+Z)", "en": "Redo (Shift+Cmd+Z)"},
    "zcheck_warn": {"ja": "⚠ time に負の値 / 範囲超過 — zPointCheck を追加",
                     "en": "⚠ negative/over-range time — add zPointCheck"},
    "chk_audio": {"ja": "音声を適用", "en": "Apply audio"},
    "audio_mode_play": {"ja": "play (可変速)", "en": "play (varispeed)"},
    "audio_mode_grain": {"ja": "grain (グラニュラー)", "en": "grain (granular)"},
    "lbl_audio_voices": {"ja": "分割:", "en": "Voices:"},
    "render_full_chain": {"ja": "本番データへチェーンを適用中… (スリット {n} 本)",
                           "en": "Applying chain at full width ({n} slits)…"},
    "audio_only_render": {
        "ja": "チェーンは前回と同一 → 映像の再レンダリングを省略し、"
              "保存済み映像 {v} に音声のみ書き出します",
        "en": "Chain unchanged → skipping video render; muxing audio onto {v}"},
    "proxy_note": {"ja": "  [プレビュー {p} 本 → 本番 {r} 本]",
                    "en": "  [preview {p} → full {r} slits]"},
    "lbl_timeline": {"ja": "入力映像の時間軸 / 使用範囲:",
                      "en": "Input timeline / used range:"},
    "timeline_readout": {
        "ja": "使用範囲 {s:.2f}–{e:.2f} 秒 (frame {sf}–{ef} / 全 {n})   |   再生位置 {p:.2f}s",
        "en": "used {s:.2f}–{e:.2f}s (frame {sf}–{ef} / {n})   |   playhead {p:.2f}s"},
    "timeline_hint": {
        "ja": "(緑=軌道が参照している範囲。ドラッグでスライドすると"
              "チェーン末尾の zCenterArange が自動調整される / 赤=プレビュー位置)",
        "en": "(green = range the trajectory reads; drag to slide — a zCenterArange "
              "step at the end of the chain is auto-adjusted / red = preview playhead)"},
    "zcenter_added": {"ja": "[timeline] zCenterArange をチェーン末尾に追加 (中央 {f} frame)",
                       "en": "[timeline] appended zCenterArange (center {f})"},
    "zcenter_updated": {"ja": "[timeline] zCenterArange の中央を {f} frame に更新",
                         "en": "[timeline] zCenterArange center → {f}"},
    "expand_must_be_last": {
        "ja": "⚠ このメソッドは既存データの左右を広げるため、これ以降のステップは"
              "配列の形が合わずエラーになります。チェーンの最後へ移動してください。",
        "en": "⚠ This widens the existing data; later steps will fail on shape "
              "mismatch. Move it to the END of the chain."},
    "expand_must_be_first": {
        "ja": "⚠ このメソッドは新しいスキャン幅を定義するため、データが空の状態"
              "(=チェーンの先頭) でのみ実行できます。先頭へ移動してください。",
        "en": "⚠ This defines a new scan width and only runs on empty data. "
              "Move it to the START of the chain."},
    "expand_hint_first": {
        "ja": "拡張系(先頭専用): 新しいスキャン幅を定義します。チェーンの先頭に置いてください"
              "（以降は新しい幅で続けられます）",
        "en": "Expand (first only): defines a new scan width — place at the START"},
    # --- 出力設定 ---
    "grp_output": {"ja": "出力設定 (Output)", "en": "Output settings"},
    "lbl_out_kind": {"ja": "出力形式:", "en": "Output:"},
    "out_video": {"ja": "動画", "en": "Video"},
    "out_still": {"ja": "連番画像", "en": "Image sequence"},
    "lbl_out_type": {"ja": "コーデック:", "en": "Codec:"},
    "lbl_img_format": {"ja": "画像形式:", "en": "Image format:"},
    "hint_img_format": {"ja": "(.png/.tif は 16bit 可 / .jpg は 8bit)",
                         "en": "(.png/.tif keep 16bit; .jpg is 8bit)"},
    "lbl_separate": {"ja": "分割数:", "en": "Segments:"},
    "hint_separate": {"ja": "(0 = 空きメモリから自動)", "en": "(0 = auto from free memory)"},
    "btn_out_toggle": {"ja": "出力設定", "en": "Output settings"},
    # --- 音声設定 ---
    "grp_audio": {"ja": "音声レンダリング (Audio)", "en": "Audio rendering"},
    "lbl_audio_smooth": {"ja": "補間:", "en": "Interp:"},
    "lbl_audio_harmonics": {"ja": "成分数:", "en": "Harmonics:"},
    "lbl_audio_grain": {"ja": "粒(秒):", "en": "Grain(s):"},
    "lbl_audio_inpan": {"ja": "パン方式:", "en": "Pan mode:"},
    "lbl_audio_gain": {"ja": "音量:", "en": "Gain:"},
    "lbl_audio_jump": {"ja": "跳び閾値(秒):", "en": "Jump thresh(s):"},
    "chk_audio_normalize": {"ja": "ノーマライズ", "en": "Normalize"},
    "grp_audio_fx": {"ja": "フレーム内在時間 (now depth) 駆動の変調",
                      "en": "Now-depth driven modulation"},
    "fx_reverb": {"ja": "リバーブ", "en": "Reverb"},
    "fx_lpf": {"ja": "LPF", "en": "LPF"},
    "fx_width": {"ja": "広がり", "en": "Width"},
    "fx_detune": {"ja": "デチューン", "en": "Detune"},
    "lbl_reverb_wet": {"ja": "wet:", "en": "wet:"},
    "lbl_reverb_time": {"ja": "残響時間 RT60(秒):", "en": "Decay RT60(s):"},
    "lbl_reverb_room": {"ja": "空間の広さ:", "en": "Room size:"},
    "lbl_reverb_predelay": {"ja": "初期反射(秒):", "en": "Predelay(s):"},
    "lbl_reverb_duck": {"ja": "原音を下げる:", "en": "Dry duck:"},
    "tip_reverb_room": {
        "ja": "リバーブの空間の広さ。反射が返ってくる間隔を倍率で変える。\n"
              "0.3=小部屋 (密で金属的) / 1.0=中ホール (既定) / 2.5=大聖堂。\n"
              "「残響時間 RT60」は消えるまでの長さで、広さとは別の軸。",
        "en": "Size of the reverb space (spacing between reflections).\n"
              "0.3 = small room, 1.0 = mid hall (default), 2.5 = cathedral.\n"
              "Decay time (RT60) is a separate axis."},
    "hint_reverb": {
        "ja": "広さ=反射の間隔 / RT60=消えるまでの長さ (別の軸として調整できます)",
        "en": "Size = spacing of reflections; RT60 = time to silence (independent)"},
    "lbl_lpf_range": {"ja": "LPF範囲(Hz):", "en": "LPF range(Hz):"},
    "lbl_width_range": {"ja": "幅範囲:", "en": "Width range:"},
    "lbl_detune_cents": {"ja": "デチューン(cent):", "en": "Detune(cents):"},
    "audio_info_title": {"ja": "音声レンダリングのパラメータ",
                          "en": "Audio rendering parameters"},
    "expand_hint_last": {
        "ja": "拡張系(末尾専用): 既存データの左右を広げます。チェーンの最後に置いてください",
        "en": "Expand (last only): widens existing data — place at the END"},
}


def tr(key, **fmt):
    d = TR.get(key)
    s = (d.get(LANG) or d.get("ja")) if d else key
    return s.format(**fmt) if fmt else s


# ======== 入力映像の回転 (Shape_of_time_flow.py と同じ方式) ========
VIDEO_ROTATIONS = [
    ("none", "vrot_none", None),
    ("cw90", "vrot_cw90", "transpose=1"),
    ("180", "vrot_180", "transpose=1,transpose=1"),
    ("ccw90", "vrot_ccw90", "transpose=2"),
    ("hflip", "vrot_hflip", "hflip"),
    ("vflip", "vrot_vflip", "vflip"),
]
VIDEO_ROTATION_VF = {rid: vf for rid, _k, vf in VIDEO_ROTATIONS}
# Display Matrix の角度 (正 = 反時計回り)。90°系はリマックスのみで済む。
VIDEO_ROTATION_ANGLE = {"cw90": -90, "180": 180, "ccw90": 90}


def apply_rotation_cv2(frame, rot_id):
    """プレビュー用: 回転を cv2 で即時適用する。"""
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


def rotated_video_path(src, rot_id):
    p = Path(src)
    return str(p.with_name(f"{p.stem}_rot-{rot_id}{p.suffix}"))


def _newest_plot_output(work_dir, since, exts=(".mp4", ".png")):
    """maneuver_3dplot / 2dplot が書いた最新の出力ファイルを探す。"""
    found = []
    for ext in exts:
        for f in glob.glob(os.path.join(work_dir, "**", "*" + ext),
                           recursive=True):
            try:
                if os.path.getmtime(f) >= since and "3dPlot" in os.path.basename(f):
                    found.append(f)
            except OSError:
                pass
    return max(found, key=os.path.getmtime) if found else ""


# ドラッグ＆ドロップで受け付ける動画の拡張子
VIDEO_DROP_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
                   ".mpg", ".mpeg", ".mts", ".m2ts", ".wmv", ".flv")
VIDEO_FILE_FILTER = ("Video Files (" +
                     " ".join("*" + e for e in VIDEO_DROP_EXTS) + ")")


def dropped_video_path(mime):
    """ドロップされた MIME から最初の動画ファイルのパスを返す (無ければ "")。"""
    if mime is None or not mime.hasUrls():
        return ""
    for u in mime.urls():
        p = u.toLocalFile()
        if (p and os.path.isfile(p)
                and os.path.splitext(p)[1].lower() in VIDEO_DROP_EXTS):
            return p
    return ""


def _fmt_dur(sec):
    """秒 → 分:秒 の短い表記 (0.1 秒単位で丸めてから桁上げする)。"""
    tenths = int(round(max(0.0, float(sec)) * 10))
    m, rem = divmod(tenths, 600)
    return f"{m}:{rem / 10.0:04.1f}" if m else f"{rem / 10.0:.1f}s"


def plan_still_chunks(data, recfps, minutes):
    """1 フレームの積み重ねを「先頭から interval 分ずつ」区切る計画を返す。

    SlitScan.py と同じ区切り方: 各区間はきっかり interval 分で、
    最後だけ余った長さになる (等分ではない)。
    戻り値: [(i0, i1, 開始秒, 終了秒), ...]
    """
    if data is None or len(data) != 1:
        return []
    z = np.asarray(data[0, :, 1], dtype=np.float64)
    fps = float(recfps) or 30.0
    interval = int(round(float(minutes) * 60.0 * fps))
    if interval < 1:
        return []
    z0, z1 = float(z.min()), float(z.max())
    order = np.argsort(z) if not np.all(np.diff(z) >= 0) else None
    zs_sorted = z[order] if order is not None else z
    out = []
    start = int(np.floor(z0))
    while start <= z1:
        end = min(start + interval, int(np.floor(z1)) + 1)
        i0 = int(np.searchsorted(zs_sorted, start, "left"))
        i1 = int(np.searchsorted(zs_sorted, end, "left"))
        if i1 > i0:
            out.append((i0, i1, start / fps, end / fps))
        start += interval
    return out


def still_chunk_summary(chunks):
    """チャンク計画 → (枚数, 全体秒, 各区間の秒のリスト)。"""
    if not chunks:
        return 0, 0.0, []
    each = [c[3] - c[2] for c in chunks]
    return len(chunks), chunks[-1][3] - chunks[0][2], each


class ClickableLabel(QLabel):
    """ダブルクリックを通知する QLabel (プロットの原寸表示に使う)。"""
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, ev):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(ev)


class ImageWindow(QDialog):
    """画像を原寸で表示する別ウィンドウ (大きい場合はスクロール)。"""

    def __init__(self, parent, title, pixmap):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setSizeGripEnabled(True)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel()
        lbl.setPixmap(pixmap)
        lbl.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(lbl)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setFrameShape(QFrame.NoFrame)
        v.addWidget(scroll)
        scr = QApplication.primaryScreen()
        avail = scr.availableGeometry() if scr is not None else None
        if avail is not None:
            self.resize(min(pixmap.width() + 24, int(avail.width() * 0.9)),
                        min(pixmap.height() + 24, int(avail.height() * 0.9)))
        else:
            self.resize(min(pixmap.width() + 24, 1200),
                        min(pixmap.height() + 24, 900))


class Plot3DWorker(QThread):
    """maneuver_3dplot をバックグラウンドで走らせる (数十秒かかるため)。"""
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str)

    def __init__(self, dm, work_dir, **kwargs):
        super().__init__()
        self.dm, self.work_dir, self.kwargs = dm, work_dir, kwargs

    def run(self):
        since = time.time() - 1
        try:
            self.dm.maneuver_3dplot(**self.kwargs)
        except Exception as e:
            self.log_signal.emit("[ERROR] maneuver_3dplot: " + "".join(
                traceback.format_exception_only(type(e), e)).strip())
            self.done_signal.emit(False, "")
            return
        finally:
            try:
                plt.close("all")
            except Exception:
                pass
        out = _newest_plot_output(self.work_dir, since)
        self.done_signal.emit(bool(out), os.path.abspath(out) if out else "")


class VideoRotateWorker(QThread):
    """90°系は Display Matrix のリマックス (瞬時)、反転のみ再エンコード。"""
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str)

    def __init__(self, src, out, vf, rot_angle=None):
        super().__init__()
        self.src, self.out, self.vf, self.rot_angle = src, out, vf, rot_angle

    def run(self):
        if self.rot_angle is not None:
            try:
                from imgtrans_lib._utils import probe_video_rotation
                existing = int(probe_video_rotation(self.src))
            except Exception:
                existing = 0
            total = (existing + int(self.rot_angle)) % 360
            if total > 180:
                total -= 360
            cmd = ["ffmpeg", "-y", "-nostdin", "-display_rotation", str(total),
                   "-i", self.src, "-c", "copy", "-loglevel", "error", self.out]
        else:
            cmd = ["ffmpeg", "-y", "-nostdin", "-i", self.src, "-vf", self.vf,
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
                   "-c:a", "copy", "-loglevel", "error", self.out]
        self.log_signal.emit("[ffmpeg] " + " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] {e}")
            self.done_signal.emit(False, "")
            return
        if proc.stderr.strip():
            self.log_signal.emit(proc.stderr.strip())
        ok = proc.returncode == 0 and os.path.exists(self.out)
        self.done_signal.emit(ok, self.out if ok else "")


class FitScrollArea(QScrollArea):
    """中身の高さを推奨値として申告するスクロール領域。

    既定の QScrollArea は中身によらず小さな sizeHint を返すため、縦に
    余裕のある画面でも最小高で表示されてしまう。高さが足りるときは
    そのまま全部見せ、足りないときだけスクロールさせたい箇所で使う。
    """

    def sizeHint(self):
        s = super().sizeHint()
        w = self.widget()
        if w is not None:
            s.setHeight(w.sizeHint().height() + 2 * self.frameWidth())
        return s


class Accordion(QWidget):
    """クリックで開閉する折りたたみパネル。出力/音声設定をまとめるのに使う。"""

    def __init__(self, title, expanded=False):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self.button = QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.button.setStyleSheet(
            "QToolButton { border:none; font-weight:bold; color:#444; }")
        self.button.toggled.connect(self._on_toggle)
        v.addWidget(self.button)
        self.body = QFrame()
        self.body.setFrameShape(QFrame.StyledPanel)
        self.body.setVisible(expanded)
        self.body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 6, 8, 8)
        self.body_layout.setSpacing(6)
        v.addWidget(self.body)

    def _on_toggle(self, on):
        self.button.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)
        # 開いた分の高さを最小値として固定する。これが無いと、縦が足りない
        # ときに親レイアウトが中身を比例縮小して設定項目が見えなくなる
        # (親側はスクロールで逃がす。_sync_settings_height を参照)
        self.body.setMinimumHeight(self.body_layout.sizeHint().height()
                                   if on else 0)

    def set_title(self, t):
        self.button.setText(t)

    def addLayout(self, lay):
        lay.setContentsMargins(0, 0, 0, 0)
        self.body_layout.addLayout(lay)

    def addWidget(self, w):
        self.body_layout.addWidget(w)


# ======== 出力フォーマット / 音声パラメータ定義 ========
# (out_type, 表示名, 拡張子, 補足)  ※ imgtrans の OUT_* 定数に対応
OUT_TYPES = [
    (1, "H.264 (8bit SDR)", ".mp4", "既定。最大 4096×2160"),
    (2, "H.265 (10bit HDR10 PQ)", ".mp4", "最大 8192×4320"),
    (5, "H.265 (10bit SDR)", ".mp4", "SDR カラータグ付き"),
    (7, "H.265 HW (VideoToolbox)", ".mp4", "macOS ハードウェアエンコード・高速"),
    (3, "ProRes 422 HQ (10bit HDR)", ".mov", "解像度制限なし"),
    (6, "ProRes 422 HQ (10bit SDR)", ".mov", "SDR カラータグ付き"),
    (4, "ProRes 4444 (10bit HDR)", ".mov", "4:4:4・解像度制限なし"),
]
IMG_FORMATS = [
    (".png", "PNG (16bit 可)"),
    (".tif", "TIFF (16bit 可)"),
    (".bmp", "BMP"),
    (".jpg", "JPEG (8bit)"),
]

# audio_render の全パラメータ。Info ボタンで README の説明を引くのに使う。
AUDIO_PARAM_DOC = "audio_render"


# ======== 入力映像タイムライン (使用範囲バンド + 再生位置) ========
class ManeuverTimelineSlider(QWidget):
    """入力映像の全長を表すバーに、軌道が参照している範囲と再生位置を重ねる。

    Shape_of_time_flow.py の RangeTimelineSlider を簡略移植したもの。
      緑バンド : 軌道データが実際に参照している入力時間範囲。
                 **本体をドラッグしてスライドのみ可能** (端の伸縮は非対応)。
      赤ライン : プレビュー表示位置。ドラッグでスクラブ。
    値はすべて 0..1 の割合 (フレーム番号への換算は呼び出し側)。
    """
    usedRangeSlid = pyqtSignal(float, float)   # スライド後の (start, end)
    playheadChanged = pyqtSignal(float)

    GRAB_PX = 8

    def __init__(self):
        super().__init__()
        self._used = None            # (s, e) or None
        self._pos = 0.0
        self._drag = None            # "band" | "pos" | None
        self._drag_dx = 0.0
        self.setFixedHeight(30)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def set_used_range(self, s, e):
        """使用範囲を設定する。0..1 の外 (= 入力映像の範囲外) も保持する。"""
        if s is None or e is None:
            self._used = None
        else:
            s, e = float(s), float(e)
            self._used = (s, max(s, e))
        self.update()

    def set_playhead(self, f):
        self._pos = min(max(0.0, float(f)), 1.0)
        self.update()

    def playhead(self):
        return self._pos

    def used_range(self):
        return self._used

    def playhead(self):
        return self._pos

    # --- 座標変換 ---
    def _f2x(self, f):
        return 2 + f * max(1, self.width() - 5)

    def _x2f(self, x):
        return min(1.0, max(0.0, (x - 2) / max(1, self.width() - 5)))

    # --- マウス ---
    def _hit(self, x):
        if abs(x - self._f2x(self._pos)) <= self.GRAB_PX:
            return "pos"
        if self._used is not None:
            x0 = self._f2x(min(max(self._used[0], 0.0), 1.0))
            x1 = self._f2x(min(max(self._used[1], 0.0), 1.0))
            if x0 - 2 <= x <= x1 + 2:
                return "band"
        return None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        hit = self._hit(ev.pos().x())
        if hit == "band":
            self._drag_dx = self._x2f(ev.pos().x()) - self._used[0]
        elif hit is None:
            hit = "pos"
            self._apply("pos", self._x2f(ev.pos().x()))
        self._drag = hit

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._apply(self._drag, self._x2f(ev.pos().x()))
        else:
            hit = self._hit(ev.pos().x())
            self.setCursor(Qt.OpenHandCursor if hit == "band"
                           else (Qt.SizeHorCursor if hit else Qt.PointingHandCursor))

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def _apply(self, which, f):
        if which == "band" and self._used is not None:
            span = self._used[1] - self._used[0]
            s = f - self._drag_dx
            if span <= 1.0:
                s = min(max(0.0, s), 1.0 - span)      # バー内に収める
            else:
                s = min(max(1.0 - span, s), 0.0)      # 軌道が映像より長い場合
            self._used = (s, s + span)
            self.usedRangeSlid.emit(s, s + span)
        else:
            self._pos = f
            self.playheadChanged.emit(self._pos)
        self.update()

    # --- 描画 ---
    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        bar_y, bar_h = 5, h - 10
        p.fillRect(2, bar_y, w - 4, bar_h, QColor(40, 40, 40))
        if self._used is not None:
            us, ue = self._used
            cs, ce = min(max(us, 0.0), 1.0), min(max(ue, 0.0), 1.0)
            x0, x1 = int(self._f2x(cs)), int(self._f2x(ce))
            p.fillRect(x0, bar_y, max(2, x1 - x0), bar_h, QColor(60, 200, 120, 160))
            # 端の線: 入力映像の範囲内なら緑、はみ出している側は警告色
            for x, out in ((x0, us < -1e-6), (x1, ue > 1.0 + 1e-6)):
                pen = QPen(QColor(230, 120, 40) if out else QColor(60, 220, 130))
                pen.setWidth(3 if out else 2)
                p.setPen(pen)
                p.drawLine(x, bar_y - 2, x, bar_y + bar_h + 2)
            # はみ出し量を端の三角で示す
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(230, 120, 40))
            if us < -1e-6:
                p.drawPolygon(*[QPoint(2, h // 2), QPoint(9, bar_y),
                                QPoint(9, bar_y + bar_h)])
            if ue > 1.0 + 1e-6:
                p.drawPolygon(*[QPoint(w - 2, h // 2), QPoint(w - 9, bar_y),
                                QPoint(w - 9, bar_y + bar_h)])
            p.setBrush(Qt.NoBrush)
        xp = int(self._f2x(self._pos))
        pen = QPen(QColor(255, 40, 40))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(xp, 0, xp, h)
        p.end()


# ======== メソッドレジストリ (イントロスペクションで自動生成) ========
# 定義元ミックスインでカテゴリを決める。imgtrans にメソッドが増えれば
# 自動的に選択肢へ現れる。
_MIXIN_CATEGORY = {
    "TransformsAddMixin": "add",
    "TransformsApplyMixin": "apply",
    "DataOpsMixin": "data",        # apply* で始まるものは apply へ振り替え
    "VisualizeMixin": "other",
    "AudioMixin": "other",
    "SurfaceRenderingMixin": "other",
}
CATEGORY_ORDER = ["add", "apply", "data", "expand", "other"]
CATEGORY_LABEL_KEY = {"add": "cat_add", "apply": "cat_apply",
                      "data": "cat_data", "expand": "cat_expand",
                      "other": "cat_other"}
# 種別ごとの色。ステップカードの枠と背景、種別コンボ、ヘルプのチップで共通。
CATEGORY_COLOR = {"add": "#2a6fd6", "apply": "#c07000",
                  "data": "#2a8f4f", "expand": "#8e44ad", "other": "#777777"}


def category_of(method_name):
    for c, lst in METHOD_REGISTRY.items():
        if any(n == method_name for n, _ in lst):
            return c
    return "other"


def category_icon(cat, size=12):
    pm = QPixmap(size, size)
    pm.fill(QColor(CATEGORY_COLOR.get(cat, "#888")))
    return QIcon(pm)


def _rgba(hex_color, alpha):
    c = QColor(hex_color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"

# スキャン方向のサイズ (data.shape[1]) を変えるメソッド群。
# 実行後は配列の形が変わるため、置ける位置に制約がある。
# 実測 (2026-08-11) で挙動が2種類に分かれることを確認した。
#
# ① 先頭専用 (EXPAND_FIRST):
#    新しいスキャン幅を定義し、scan_nums もその値へ更新する。
#    データが空の状態でのみ実行でき、**その後は新しい幅で他のメソッドを続けられる**。
#      Cut を先頭 → addFlat  : OK  (51, 1440)
#      addFlat → Cut         : NG  (形が合わない)
#    ※ addCycleTrans など width/height を直接参照するメソッドは
#      更新後の scan_nums を見ないため、続けても失敗する
EXPAND_FIRST = {
    "addWideKeyframeTrans",   # scan_nums = width * wide_scale
    "addCylinderCut",         # _cut_finalize が scan_nums = output_width に更新
    "addBoxUnfoldCut",        # 同上 (直方体の周長ぶんの幅になる)
}
# ② 末尾専用 (EXPAND_LAST):
#    既存データの左右を np.pad で広げる。scan_nums は据え置きなので、
#    **これ以降にステップを足すと必ず失敗する**。
#      addFlat → wide_expandB          : OK  (50, 1320)
#      addFlat → wide_expandB → addFlat : NG
EXPAND_LAST = {
    "wide_expandB",
}
EXPAND_METHODS = EXPAND_FIRST | EXPAND_LAST

# チェーンに載せない (アプリ側が別 UI で扱う / 本アプリの対象外)
_EXCLUDE = {
    "img_to_maneuver", "img_to_maneuver_rate_based",   # グレー画像適用は対象外
    "transprocess", "new_transprocess", "pretransprocess",  # レンダリングボタン
    "surfaceTransprocess", "setup_surface_audio",
    "print_progress", "maneuver_log", "info_setting",
    "extract_params_from_filename", "yuv420_to_rgb",
    "hlg_eotf", "pq_eotf", "hlg_oetf", "pq_oetf", "cross_point",
    "surface_grid_time_positions", "spline_interpolate",
}


def build_method_registry():
    """drawManeuver の公開メソッドをカテゴリ別に列挙する。

    returns: {category: [(name, signature), ...]}
    """
    reg = {c: [] for c in CATEGORY_ORDER}
    for name, fn in inspect.getmembers(drawManeuver, inspect.isfunction):
        if name.startswith("_") or name in _EXCLUDE:
            continue
        owner = fn.__qualname__.split(".")[0]
        cat = _MIXIN_CATEGORY.get(owner)
        if cat is None:
            continue
        if cat == "data" and name.startswith("apply"):
            cat = "apply"
        if name in EXPAND_METHODS:
            cat = "expand"      # 空間サイズを変える系は独立カテゴリへ
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            continue
        reg[cat].append((name, sig))
    for c in reg:
        reg[c].sort(key=lambda t: t[0].lower())
    return reg


METHOD_REGISTRY = build_method_registry()
METHOD_SIGS = {n: s for lst in METHOD_REGISTRY.values() for n, s in lst}


# ---- README からメソッド説明を引用 ----
_README_CACHE = {}


def _readme_paths():
    """imgtrans リポジトリの README を言語優先順で返す。"""
    try:
        root = Path(inspect.getfile(drawManeuver)).resolve().parent.parent
    except Exception:
        return []
    ja, en = root / "README_JA.md", root / "README.md"
    order = [ja, en] if LANG == "ja" else [en, ja]
    return [p for p in order if p.exists()]


_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_HEAD_RE = re.compile(r"^(#{2,6})\s+(.*?)\s*$")


def _parse_readme(path):
    """README を見出し単位のセクション辞書にする。

    値は {"text": 画像参照を除いた markdown, "images": [絶対パス, ...]}。
    索引は2段階で作る:
      1. 「## `メソッド名`」形式 … 変換メソッドの解説 (最優先)
      2. 見出し文にメソッド名を含む任意レベルの見出し … 音声まわりのように
         「##### Python内で完結する音声レンダリング audio_render」と
         書かれている節を拾うため
    セクションは「自分と同じか浅いレベルの見出し」が来るまでを範囲とする。
    """
    key = str(path)
    if key in _README_CACHE:
        return _README_CACHE[key]
    root = path.parent
    primary, secondary = {}, {}
    cur_name = cur_level = None
    cur_primary = False
    buf, imgs = [], []

    def flush():
        if not cur_name:
            return
        entry = {"text": "\n".join(buf).strip(), "images": list(imgs)}
        target = primary if cur_primary else secondary
        if cur_name not in target or len(entry["text"]) > len(
                target[cur_name]["text"]):
            target[cur_name] = entry

    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            hm = _HEAD_RE.match(line)
            if hm:
                level, title = len(hm.group(1)), hm.group(2)
                # 現在のセクションを終える (同じか浅いレベルの見出しで区切る)
                if cur_name and level <= cur_level:
                    flush()
                    cur_name = cur_level = None
                    cur_primary = False
                    buf, imgs = [], []
                m = re.fullmatch(r"`([A-Za-z_][A-Za-z0-9_]*)`", title)
                if level == 2 and m:
                    flush() if cur_name else None
                    cur_name, cur_level, cur_primary = m.group(1), level, True
                    buf, imgs = [], []
                    continue
                # 見出し文の末尾がメソッド名のもの (例: "… audio_render")
                tok = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", title)
                cand = next((t for t in reversed(tok) if t in METHOD_SIGS), None)
                if cand and cur_name is None:
                    cur_name, cur_level, cur_primary = cand, level, False
                    buf, imgs = [], []
                    continue
                if cur_name:
                    buf.append(line)
                continue
            if cur_name is not None:
                found = _IMG_RE.findall(line.strip())
                if found:
                    for rel in found:
                        rel = rel.split(" ")[0].strip()
                        fp = (root / rel).resolve()
                        if fp.exists() and str(fp) not in imgs:
                            imgs.append(str(fp))
                else:
                    buf.append(line)
        flush()
    except Exception:
        pass
    merged = dict(secondary)
    merged.update(primary)          # 「## `名前`」形式を優先
    _README_CACHE[key] = merged
    return merged


def readme_doc(method_name):
    """メソッド説明。{"text":…, "images":[…]} / 無ければ None。

    現在言語の README を優先し、そちらに画像が無ければもう一方の
    README の画像を補う (画像はどちらの版も同じものを指すため)。
    """
    result = None
    for path in _readme_paths():
        sec = _parse_readme(path).get(method_name)
        if not sec:
            continue
        if result is None:
            result = {"text": sec["text"], "images": list(sec["images"])}
        elif not result["images"]:
            result["images"] = list(sec["images"])
    return result


def method_params(name):
    """メソッドの (param_name, default, annotation) 一覧 (self を除く)。"""
    sig = METHOD_SIGS.get(name)
    if sig is None:
        return []
    out = []
    for pname, p in sig.parameters.items():
        if pname == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = None if p.default is inspect.Parameter.empty else p.default
        required = p.default is inspect.Parameter.empty
        ann = None if p.annotation is inspect.Parameter.empty else p.annotation
        out.append((pname, default, ann, required))
    return out


# 必須引数に入れる安全な仮値 (frame 系は一律 300)。既定値のない引数だけに使う。
PARAM_NAME_DEFAULTS = {
    "frame_nums": 300, "FRAME_NUMS": 300, "addframe": 300, "num_frames": 300,
    "outnums": 300, "frame_speed": 1,
    "settime": 0, "slide_time": 1, "trackslit": 0, "point_frame": 0,
    "i_direction": 0, "z_direction": 0, "axis_position": 0.5,
    "sblur": 5, "tblur": 5, "bl_time": 5, "maxgap": 10,
    "s_frame": 0, "e_frame": 300, "a_frame": 0, "b_frame": 300,
    "target_z": 0, "target_frame": 0, "center_time": 100,
    "zdepth": 100, "v": 1.0, "xypoint": 0.5, "start": 0, "end": 300,
    "connection_num": 30,
    # 拡張系
    "key_array": [[0, 0], [300, 300]], "add_size": 960,
    # サイクル/波形系の必須引数
    "cycle_degree": 360, "start_center": 0.5, "end_center": 0.5,
}


def literal_or_str(text):
    """テキストを Python リテラルとして解釈する (失敗したら文字列のまま)。"""
    t = text.strip()
    if t == "":
        return None
    try:
        return ast.literal_eval(t)
    except (ValueError, SyntaxError):
        return t


def fmt_value(v):
    """kwargs 値をコード表現にする (.py 書き出し用)。"""
    if isinstance(v, np.ndarray):
        return repr(v.tolist())
    return repr(v)


def method_doc_parts(method_name):
    """(署名, 本文, 図パス, 定義場所) を返す (README 優先 → docstring)。"""
    fn = getattr(drawManeuver, method_name, None)
    try:
        sig = str(inspect.signature(fn)).replace("self, ", "").replace("self", "")
    except Exception:
        sig = "()"
    try:
        where = (f"{os.path.basename(inspect.getfile(fn))}:"
                 f"{inspect.getsourcelines(fn)[1]}")
    except Exception:
        where = ""
    doc = readme_doc(method_name)
    if doc:
        text, images = doc["text"], doc["images"]
    else:
        text, images = (inspect.getdoc(fn) if fn else None) or tr("no_doc"), []
    return sig, text, images, where


class DocBody(QScrollArea):
    """README のメソッド説明を、図・GIF アニメーション付きで表示する本文部分。

    QTextEdit の markdown 表示は GIF を静止画としてしか扱えないため、
    本文と図を分離し、図は QLabel + QMovie で再生する。
    DocDialog (単体) と MethodHelpWindow (一覧付き) の両方で使う。
    """

    MAX_IMG_W = 900

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self._movies = []
        self._inner = None
        self._body = None

    def set_doc(self, title, subtitle, doc_text, images, where=""):
        for m in self._movies:
            try:
                m.stop()
            except Exception:
                pass
        self._movies = []
        inner = QWidget()
        self._inner = inner
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)

        head = QLabel(f"<b>{title}</b> <span style='color:#666;'>{subtitle}</span>"
                      + (f"<br><span style='color:#999; font-size:10px;'>{where}</span>"
                         if where else ""))
        head.setWordWrap(True)
        head.setTextInteractionFlags(Qt.TextSelectableByMouse)
        il.addWidget(head)

        # 本文は QLabel のリッチテキストで置く。QTextEdit だと内側に
        # 独自スクロールを持ってしまい、長い説明が狭い枠に閉じ込められる。
        # QLabel なら折り返し後の高さが自動で決まり、外側のスクロールに乗る。
        src = QTextDocument()
        try:
            src.setMarkdown(doc_text)
        except Exception:
            src.setPlainText(doc_text)
        self._restyle(src)
        body = QLabel(src.toHtml())
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setContentsMargins(2, 2, 2, 2)
        self._body = body
        il.addWidget(body)

        for path in images or []:
            cap = QLabel(os.path.basename(path))
            cap.setStyleSheet("color:#999; font-size:10px; margin-top:6px;")
            il.addWidget(cap)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#fff; border:1px solid #ddd;")
            if path.lower().endswith(".gif"):
                movie = QMovie(path)
                if movie.isValid():
                    native = QImageReader(path).size()
                    if native.width() > self.MAX_IMG_W:
                        sc = self.MAX_IMG_W / native.width()
                        movie.setScaledSize(QSize(self.MAX_IMG_W,
                                                  max(1, int(native.height() * sc))))
                    lbl.setMovie(movie)
                    movie.start()
                    self._movies.append(movie)
            else:
                pm = QPixmap(path)
                if not pm.isNull():
                    if pm.width() > self.MAX_IMG_W:
                        pm = pm.scaledToWidth(self.MAX_IMG_W, Qt.SmoothTransformation)
                    lbl.setPixmap(pm)
            il.addWidget(lbl)
        il.addStretch()
        self.setWidget(inner)
        self.verticalScrollBar().setValue(0)

    # 見出しの文字サイズ (pt)。README は #### 以下も使うが、Qt の
    # setMarkdown は h5/h6 を本文より小さく描くため読めなくなる。
    # setDefaultStyleSheet は setMarkdown に効かないので直接補正する。
    HEAD_PT = {1: 19.0, 2: 17.0, 3: 15.5, 4: 14.5, 5: 14.0, 6: 13.5}
    BODY_PT = 12.5

    def _restyle(self, doc):
        cur = QTextCursor(doc)
        blk = doc.begin()
        while blk.isValid():
            lvl = blk.blockFormat().headingLevel()
            fmt = QTextCharFormat()
            fmt.setFontPointSize(self.HEAD_PT.get(lvl, self.BODY_PT))
            if lvl:
                fmt.setFontWeight(QFont.Bold)
            cur.setPosition(blk.position())
            cur.setPosition(blk.position() + max(0, blk.length() - 1),
                            QTextCursor.KeepAnchor)
            cur.mergeCharFormat(fmt)
            blk = blk.next()

    def stop_movies(self):
        for m in self._movies:
            try:
                m.stop()
            except Exception:
                pass


def _size_for_screen(widget, frac_w=0.62, frac_h=0.85, max_w=1000, max_h=900):
    scr = QApplication.primaryScreen()
    avail = scr.availableGeometry() if scr is not None else None
    if avail is not None:
        widget.resize(min(max_w, max(700, int(avail.width() * frac_w))),
                      min(max_h, max(600, int(avail.height() * frac_h))))
    else:
        widget.resize(max_w, max_h)


class DocDialog(QDialog):
    """1 メソッドぶんの説明ダイアログ (ⓘ ボタン用)。"""

    def __init__(self, parent, title, subtitle, doc_text, images, where=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setSizeGripEnabled(True)
        _size_for_screen(self)
        v = QVBoxLayout(self)
        self.body = DocBody(self)
        self.body.set_doc(title, subtitle, doc_text, images, where)
        v.addWidget(self.body, 1)
        close = QPushButton("OK")
        close.clicked.connect(self.accept)
        v.addWidget(close, 0, Qt.AlignRight)

    def closeEvent(self, ev):
        self.body.stop_movies()
        super().closeEvent(ev)


class MethodHelpWindow(QDialog):
    """メソッド一覧 + ヘルプ。上段の種別チップで絞り込み、中段のコンボで選び、
    下段にそのメソッドの README 説明を出す (非モーダル)。"""
    add_requested = pyqtSignal(str, str)     # (category, method)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("help_title"))
        self.setSizeGripEnabled(True)
        self.setModal(False)
        _size_for_screen(self, 0.66, 0.85, 1100, 950)
        v = QVBoxLayout(self)

        # --- 種別チップ (横並び・トグル、未選択 = 全メソッド) ---
        chips = QHBoxLayout()
        self.chips = {}
        for c in CATEGORY_ORDER:
            b = QToolButton()
            b.setCheckable(True)
            b.setText(tr(CATEGORY_LABEL_KEY[c]).split(" (")[0])
            b.setToolTip(tr(CATEGORY_LABEL_KEY[c]))
            col = CATEGORY_COLOR[c]
            b.setStyleSheet(
                f"QToolButton {{ border:1px solid {col}; border-radius:10px;"
                f" padding:4px 12px; color:{col}; font-weight:bold;"
                f" background:{_rgba(col, 0.08)}; }}"
                f"QToolButton:checked {{ background:{col}; color:white; }}")
            b.clicked.connect(lambda _=False, cc=c: self._on_chip(cc))
            chips.addWidget(b)
            self.chips[c] = b
        chips.addStretch()
        v.addLayout(chips)
        hint = QLabel(tr("help_hint"))
        hint.setStyleSheet("color:gray; font-size:10px;")
        v.addWidget(hint)

        # --- メソッド選択 ---
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("lbl_method")))
        self.method_combo = QComboBox()
        self.method_combo.setMinimumWidth(320)
        self.method_combo.currentIndexChanged.connect(self._on_method)
        row.addWidget(self.method_combo, 1)
        self.add_btn = QPushButton(tr("btn_add_from_help"))
        self.add_btn.clicked.connect(self._on_add)
        row.addWidget(self.add_btn)
        v.addLayout(row)

        # --- 説明 ---
        self.body = DocBody(self)
        v.addWidget(self.body, 1)

        self._cat = None
        self._fill_methods()

    def _on_chip(self, cat):
        # 押した種別が既に選ばれていれば解除 (= 全メソッド)
        self._cat = None if self._cat == cat else cat
        for c, b in self.chips.items():
            b.setChecked(c == self._cat)
        self._fill_methods()

    def _fill_methods(self, keep=None):
        cur = keep or self.method_combo.currentData()
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        cats = [self._cat] if self._cat else CATEGORY_ORDER
        for c in cats:
            for name, sig in METHOD_REGISTRY.get(c, []):
                params = [p for p in sig.parameters if p != "self"]
                self.method_combo.addItem(category_icon(c), name, name)
                self.method_combo.setItemData(
                    self.method_combo.count() - 1,
                    f"[{c}] {name}({', '.join(params)})", Qt.ToolTipRole)
        self.method_combo.blockSignals(False)
        i = self.method_combo.findData(cur) if cur else -1
        self.method_combo.setCurrentIndex(i if i >= 0 else 0)
        self._on_method()

    def _on_method(self, *_):
        name = self.method_combo.currentData()
        if not name:
            return
        sig, text, images, where = method_doc_parts(name)
        self.body.set_doc(name, sig, text, images, where)
        self.add_btn.setEnabled(True)

    def _on_add(self):
        name = self.method_combo.currentData()
        if name:
            self.add_requested.emit(category_of(name), name)

    def select(self, category=None, method=None):
        """メインウィンドウで選ばれている種別/メソッドに合わせて開く。"""
        self._cat = category if category in CATEGORY_ORDER else None
        for c, b in self.chips.items():
            b.setChecked(c == self._cat)
        self._fill_methods(keep=method)

    def closeEvent(self, ev):
        self.body.stop_movies()
        super().closeEvent(ev)


def show_method_doc(parent, method_name):
    """メソッドの説明ダイアログを開く (README 優先 → docstring)。"""
    sig, text, images, where = method_doc_parts(method_name)
    DocDialog(parent, method_name, sig, text, images, where).exec_()


class WorkDirBrowser(QWidget):
    """作業ディレクトリの簡易ファイラ。クリックでファイルを既定のアプリで開き、
    フォルダはその中へ入る。「上へ」でルート (作業ディレクトリ) まで戻れる。"""

    ICON = {".mp4": "🎞", ".mov": "🎞", ".m4v": "🎞", ".avi": "🎞", ".mkv": "🎞",
            ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼", ".tif": "🖼", ".tiff": "🖼",
            ".gif": "🖼", ".bmp": "🖼", ".wav": "🔊", ".aif": "🔊", ".aiff": "🔊",
            ".py": "🐍", ".txt": "📄", ".log": "📄", ".csv": "📄", ".json": "📄",
            ".npy": "🧮", ".scd": "🎼"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root = ""
        self._cwd = ""
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        head = QHBoxLayout()
        self.back_btn = QToolButton()
        self.back_btn.setText(tr("btn_dir_back"))
        self.back_btn.clicked.connect(self.go_up)
        head.addWidget(self.back_btn)
        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color:#555; font-size:10px;")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        head.addWidget(self.path_label, 1)
        self.refresh_btn = QToolButton()
        self.refresh_btn.setText("↻")
        self.refresh_btn.clicked.connect(self.refresh)
        head.addWidget(self.refresh_btn)
        v.addLayout(head)
        self.list = QListWidget()
        self.list.setFixedHeight(170)
        self.list.setStyleSheet("QListWidget { font-size:11px; }")
        self.list.setToolTip(tr("tip_dir_click"))
        self.list.itemClicked.connect(self._on_item)
        v.addWidget(self.list)

    def set_root(self, path):
        self._root = os.path.abspath(path) if path else ""
        self._cwd = self._root
        self.refresh()

    def refresh(self):
        self.list.clear()
        if not self._cwd or not os.path.isdir(self._cwd):
            self.path_label.setText("")
            self.back_btn.setEnabled(False)
            return
        rel = os.path.relpath(self._cwd, self._root) if self._root else self._cwd
        self.path_label.setText(os.path.basename(self._root) + ("" if rel == "." else "/" + rel))
        self.back_btn.setEnabled(self._cwd != self._root)
        try:
            names = sorted(os.listdir(self._cwd), key=str.lower)
        except OSError:
            names = []
        dirs = [n for n in names if os.path.isdir(os.path.join(self._cwd, n))
                and not n.startswith(".")]
        files = [n for n in names if os.path.isfile(os.path.join(self._cwd, n))
                 and not n.startswith(".")]
        for n in dirs:
            it = QListWidgetItem(f"📁 {n}/")
            it.setData(Qt.UserRole, os.path.join(self._cwd, n))
            it.setData(Qt.UserRole + 1, "dir")
            self.list.addItem(it)
        for n in files:
            ext = os.path.splitext(n)[1].lower()
            full = os.path.join(self._cwd, n)
            try:
                sz = os.path.getsize(full)
            except OSError:
                sz = 0
            it = QListWidgetItem(f"{self.ICON.get(ext, '·')} {n}   ({_fmt_size(sz)})")
            it.setData(Qt.UserRole, full)
            it.setData(Qt.UserRole + 1, "file")
            it.setToolTip(full)
            self.list.addItem(it)
        if not dirs and not files:
            it = QListWidgetItem(tr("dir_empty"))
            it.setFlags(Qt.NoItemFlags)
            self.list.addItem(it)

    def go_up(self):
        if self._cwd and self._cwd != self._root:
            self._cwd = os.path.dirname(self._cwd)
            if not self._cwd.startswith(self._root):
                self._cwd = self._root
            self.refresh()

    def _on_item(self, it):
        path = it.data(Qt.UserRole)
        kind = it.data(Qt.UserRole + 1)
        if not path:
            return
        if kind == "dir":
            self._cwd = path
            self.refresh()
        elif os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# ======== 引数エディタ ========
class ParamEditor(QWidget):
    """1 メソッドぶんの引数入力欄を signature から自動生成する。"""
    changed = pyqtSignal()

    def __init__(self, method_name, values=None):
        super().__init__()
        self.method_name = method_name
        self._widgets = {}      # pname -> (kind, widget)
        self._required = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        values = values or {}

        params = method_params(method_name)
        if not params:
            lbl = QLabel("(引数なし)" if LANG == "ja" else "(no arguments)")
            lbl.setStyleSheet("color: gray; font-size: 10px;")
            v.addWidget(lbl)
            return

        # 2 列に詰めて縦を短くする
        row = None
        for i, (pname, default, ann, required) in enumerate(params):
            if i % 2 == 0:
                row = QHBoxLayout()
                row.setSpacing(6)
                row.setContentsMargins(0, 0, 0, 0)
                v.addLayout(row)
            cell = QHBoxLayout()
            cell.setSpacing(3)
            lab = QLabel(pname + ":")
            lab.setStyleSheet("font-size: 10px;"
                              + (" color:#c0392b;" if required else " color:#555;"))
            lab.setToolTip(f"{pname} (default={default!r})")
            cell.addWidget(lab)
            init = values.get(pname, default)
            if init is None and required and pname in PARAM_NAME_DEFAULTS:
                init = PARAM_NAME_DEFAULTS[pname]
                if default is None:
                    default = init      # widget 型推定にも仮値を使う
            kind, w = self._make_widget(pname, default, ann, required, init)
            self._widgets[pname] = (kind, w)
            self._required[pname] = required
            cell.addWidget(w, 1)
            row.addLayout(cell, 1)

    def _make_widget(self, pname, default, ann, required, value):
        """既定値/注釈から適切なウィジェットを選ぶ。"""
        is_bool = isinstance(default, bool) or ann is bool
        is_int = (isinstance(default, int) and not isinstance(default, bool)) or ann is int
        is_float = isinstance(default, float) or ann is float

        if is_bool:
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda *_: self.changed.emit())
            return "bool", w
        if is_int:
            w = QSpinBox()
            w.setRange(-9999999, 9999999)
            w.setValue(int(value) if isinstance(value, (int, float)) else 0)
            w.setMaximumWidth(96)
            w.valueChanged.connect(lambda *_: self.changed.emit())
            return "int", w
        if is_float:
            w = QDoubleSpinBox()
            w.setRange(-1e7, 1e7)
            w.setDecimals(4)
            w.setSingleStep(0.1)
            w.setValue(float(value) if isinstance(value, (int, float)) else 0.0)
            w.setMaximumWidth(110)
            w.valueChanged.connect(lambda *_: self.changed.emit())
            return "float", w
        # str / None / list / tuple / 必須(型不明) → リテラル入力
        w = QLineEdit()
        if value is None:
            w.setText("")
            w.setPlaceholderText(tr("param_required") if required else "None")
        elif isinstance(value, str):
            w.setText(value)
        else:
            w.setText(repr(value))
        w.setStyleSheet("font-size: 10px;")
        w.editingFinished.connect(lambda *_: self.changed.emit())
        return "literal", w

    def values(self):
        """現在の kwargs を返す (未入力の任意引数は省略 = 既定値を使う)。"""
        out = {}
        for pname, (kind, w) in self._widgets.items():
            if kind == "bool":
                out[pname] = w.isChecked()
            elif kind == "int":
                out[pname] = w.value()
            elif kind == "float":
                out[pname] = w.value()
            else:
                txt = w.text().strip()
                if txt == "":
                    if self._required.get(pname):
                        out[pname] = None      # 必須未入力 → 実行時にエラー表示
                    continue                    # 任意なら省略して既定値に任せる
                out[pname] = literal_or_str(txt)
        return out

    def missing_required(self):
        return [p for p, req in self._required.items()
                if req and self.values().get(p, None) is None]


# ======== ステップ ウィジェット ========
class StepWidget(QFrame):
    """チェーン 1 ステップぶんのカード。"""
    changed = pyqtSignal()
    move_requested = pyqtSignal(object, int)     # (self, delta)
    remove_requested = pyqtSignal(object)
    duplicate_requested = pyqtSignal(object)

    CAT_COLOR = CATEGORY_COLOR

    def __init__(self, category, method_name, values=None, enabled=True):
        super().__init__()
        self.category = category
        self.method_name = method_name
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._set_border()
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 3, 6, 3)
        v.setSpacing(2)

        head = QHBoxLayout()
        head.setSpacing(4)
        self.enable_chk = QCheckBox()
        self.enable_chk.setChecked(enabled)
        self.enable_chk.setToolTip(tr("step_disabled"))
        self.enable_chk.toggled.connect(lambda *_: (self._set_border(),
                                                    self.changed.emit()))
        head.addWidget(self.enable_chk)
        self.index_label = QLabel("1")
        self.index_label.setStyleSheet("color:#888; font-size:10px;")
        self.index_label.setMinimumWidth(16)
        head.addWidget(self.index_label)
        name = QLabel(method_name)
        name.setStyleSheet(
            f"font-weight:bold; color:{self.CAT_COLOR.get(category, '#333')};")
        head.addWidget(name)
        if method_name in EXPAND_METHODS:
            badge = QLabel("⤢先" if method_name in EXPAND_FIRST else "⤢末")
            badge.setToolTip(tr("expand_hint_first")
                             if method_name in EXPAND_FIRST
                             else tr("expand_hint_last"))
            badge.setStyleSheet(
                "color:#8e44ad; font-weight:bold; border:1px solid #8e44ad;"
                " border-radius:3px; padding:0 3px; font-size:10px;")
            head.addWidget(badge)
        head.addStretch()
        info_btn = QToolButton()
        info_btn.setText("ⓘ")
        info_btn.setToolTip(tr("tip_info"))
        info_btn.setAutoRaise(True)
        info_btn.clicked.connect(self._show_info)
        head.addWidget(info_btn)
        for txt, tip, cb in (
                ("▲", tr("tip_up"), lambda: self.move_requested.emit(self, -1)),
                ("▼", tr("tip_down"), lambda: self.move_requested.emit(self, 1)),
                ("⧉", tr("tip_dup"), lambda: self.duplicate_requested.emit(self)),
                ("✕", tr("tip_del"), lambda: self.remove_requested.emit(self))):
            b = QToolButton()
            b.setText(txt)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(cb)
            head.addWidget(b)
        v.addLayout(head)

        self.editor = ParamEditor(method_name, values)
        self.editor.changed.connect(self.changed.emit)
        v.addWidget(self.editor)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color:#c0392b; font-size:10px;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        v.addWidget(self.error_label)

    def _show_info(self):
        """メソッドの説明を README (図・GIF 込み) で表示する。"""
        show_method_doc(self, self.method_name)

    def _set_border(self):
        on = self.enable_chk.isChecked() if hasattr(self, "enable_chk") else True
        col = self.CAT_COLOR.get(self.category, "#888")
        # 背景も種別の色で薄く塗る (チェーン上で種別が一目で分かるように)
        self.setStyleSheet(
            f"StepWidget {{ border:1px solid {col if on else '#ccc'};"
            f" border-radius:4px;"
            f" background: {_rgba(col, 0.13) if on else 'rgba(128,128,128,0.04)'}; }}")

    def set_index(self, i):
        self.index_label.setText(str(i + 1))

    def is_enabled_step(self):
        return self.enable_chk.isChecked()

    def spec(self):
        return {"category": self.category, "method": self.method_name,
                "kwargs": self.editor.values(),
                "enabled": self.enable_chk.isChecked()}

    def set_param(self, name, value):
        """引数をプログラムから設定する (タイムライン連動で使用)。"""
        entry = self.editor._widgets.get(name)
        if entry is None:
            return False
        kind, w = entry
        w.blockSignals(True)
        try:
            if kind == "bool":
                w.setChecked(bool(value))
            elif kind == "int":
                w.setValue(int(value))
            elif kind == "float":
                w.setValue(float(value))
            else:
                w.setText("" if value is None else repr(value))
        finally:
            w.blockSignals(False)
        return True

    def show_error(self, msg):
        self.error_label.setText(msg or "")
        self.error_label.setVisible(bool(msg))


# ======== パイプライン実行 ========
class PipelineRunner:
    """ステップ列を dm に適用する。

    ステップごとに dm.data のスナップショットを保持し、編集時は
    「変わっていない先頭部分」までを復元して、そこから先だけ再実行する。
    スナップショットの総量が予算を超えたら古いものから捨て、必要なら
    先頭から再実行する (結果は常に同じ = 決定的)。
    """
    SNAPSHOT_BUDGET = 1_500_000_000     # 約 1.5GB

    def __init__(self):
        self._keys = []        # 実行済みステップの正規化キー
        self._snaps = []       # _snaps[i] = ステップ i 実行後の data (or None)

    @staticmethod
    def _key(specs):
        return [json.dumps(s, sort_keys=True, default=str) for s in specs]

    def _store(self, i, data):
        """ステップ i 実行後の状態を予算内で保存する。"""
        while len(self._snaps) <= i:
            self._snaps.append(None)
        arr = data.copy() if isinstance(data, np.ndarray) else None
        self._snaps[i] = arr
        # 予算超過なら古い側から解放 (先頭から再実行になるだけで結果は不変)
        total = sum(s.nbytes for s in self._snaps if s is not None)
        j = 0
        while total > self.SNAPSHOT_BUDGET and j < len(self._snaps) - 1:
            if self._snaps[j] is not None:
                total -= self._snaps[j].nbytes
                self._snaps[j] = None
            j += 1

    def run(self, dm, specs, scan_override=None):
        """specs (有効ステップのみ) を順に実行。(ok, error_index, message)。

        scan_override を与えると dm.scan_nums を一時的に差し替えて実行する
        (= プレビュー用の低解像度プロキシ。add 系が作る配列の列数が減るので
        チェーン全体が桁違いに速くなる)。
        """
        keys = [f"scan={scan_override}"] + self._key(specs)
        # 変更されていない先頭部分の長さ (keys[0] はヘッダ、keys[i+1] が step i)
        common = 0
        n = min(len(keys), len(self._keys))
        while common < n and keys[common] == self._keys[common]:
            common += 1
        common_steps = max(0, common - 1)   # 一致しているステップ数
        # その範囲内で復元できる最後のスナップショットを探す
        resume = 0
        base = None
        for i in range(min(common_steps, len(self._snaps)) - 1, -1, -1):
            if self._snaps[i] is not None:
                resume, base = i + 1, self._snaps[i]
                break

        # 副作用 (log / out_name_attr / 自動プロット) と次元関連の属性を退避。
        # add 系は scan_nums だけでなく width/height を直接参照し、時間振幅は
        # 「スキャン幅 × xyt_boxel_scale」で決まるため、プロキシ実行では
        # スキャン軸の次元を縮めつつ xyt_boxel_scale を逆数で補正して
        # z (time) 値を本番と一致させる。
        saved_log, saved_attr = dm.log, dm.out_name_attr
        saved_auto = getattr(dm, "auto_visualize_out", False)
        saved_scan = dm.scan_nums
        saved_w, saved_h = dm.width, dm.height
        saved_xyt = dm.xyt_boxel_scale
        dm.auto_visualize_out = False
        if scan_override:
            ratio = float(saved_scan) / float(scan_override)
            dm.scan_nums = int(scan_override)
            if dm.scan_direction % 2 == 1:
                dm.width = int(scan_override)
            else:
                dm.height = int(scan_override)
            dm.xyt_boxel_scale = saved_xyt * ratio
        try:
            dm.data = base.copy() if base is not None else []
            self._keys = keys[:resume + 1] if resume else keys[:1]
            self._snaps = self._snaps[:resume]
            for i in range(resume, len(specs)):
                s = specs[i]
                fn = getattr(dm, s["method"], None)
                if fn is None:
                    return False, i, f"メソッドがありません: {s['method']}"
                try:
                    fn(**s["kwargs"])
                except Exception as e:
                    return False, i, f"{type(e).__name__}: {e}"
                self._keys.append(keys[i + 1])
                self._store(i, dm.data)
        finally:
            dm.log, dm.out_name_attr = saved_log, saved_attr
            dm.auto_visualize_out = saved_auto
            dm.scan_nums = saved_scan
            dm.width, dm.height = saved_w, saved_h
            dm.xyt_boxel_scale = saved_xyt
        return True, -1, ""

    def invalidate(self):
        self._keys = []
        self._snaps = []


def run_specs_plain(dm, specs, log_cb=None):
    """specs をキャッシュなしで先頭から実行する (レンダリング前の本番適用用)。

    プレビューとは別経路。dm.scan_nums は本番値のまま。
    returns: (ok, message)
    """
    saved_log, saved_attr = dm.log, dm.out_name_attr
    saved_auto = getattr(dm, "auto_visualize_out", False)
    dm.auto_visualize_out = False
    try:
        dm.data = []
        for i, s in enumerate(specs):
            fn = getattr(dm, s["method"], None)
            if fn is None:
                return False, f"メソッドがありません: {s['method']}"
            if log_cb:
                log_cb(f"  step {i + 1}/{len(specs)}: {s['method']}")
            try:
                fn(**s["kwargs"])
            except Exception as e:
                return False, f"step {i + 1} {s['method']}: {type(e).__name__}: {e}"
    finally:
        dm.log, dm.out_name_attr = saved_log, saved_attr
        dm.auto_visualize_out = saved_auto
    return True, ""


# ======== 軽量 2D プロット ========
def render_maneuver_plot(data, outfps, recfps, w_px, h_px, threads=17):
    """dm.data から Space / Time の 2 段プロットを QPixmap で返す。

    maneuver_2dplot と同じ緑→赤のスレッド配色。PNG を書かずメモリ上で
    描くため、ライブ更新でも軽い。
    """
    if data is None or len(data) == 0:
        return None
    F, S = data.shape[0], data.shape[1]
    idx = np.unique(np.round(np.linspace(0, S - 1, min(threads, S))).astype(int))
    step = max(1, F // 2000)                      # 横方向のダウンサンプル
    x = np.arange(0, F, step)
    cmap = LinearSegmentedColormap.from_list(
        "gr", [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)], N=max(2, len(idx)))

    dpi = 100.0
    fig = plt.Figure(figsize=(max(2.0, w_px / dpi), max(1.5, h_px / dpi)), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    for k, si in enumerate(idx):
        c = cmap(k / max(1, len(idx) - 1))
        ax1.plot(x, data[::step, si, 0], color=c, linewidth=0.8)
        ax2.plot(x, data[::step, si, 1], color=c, linewidth=0.8)
    ax1.set_ylabel("Space(px)", fontsize=8)
    ax2.set_ylabel("Time(frame)", fontsize=8)
    ax2.set_xlabel(f"output frame  ({F/max(1e-6, outfps):.2f}s @ {outfps:.0f}fps)",
                   fontsize=8)
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.set_facecolor("white")
    ax1.tick_params(labelbottom=False)
    fig.tight_layout(pad=0.6)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    h, w = buf.shape[:2]
    rgb = np.ascontiguousarray(buf[..., :3])
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    plt.close(fig)
    return QPixmap.fromImage(qimg)


# ======== GPU プレビュー (dm.data を直接マップ化) ========
class ManeuverRTPreview(RealtimePreviewWidget):
    """RealtimePreviewWidget を dm.data 駆動にしたサブクラス。

    本家はグレー画像 (space/time/rate PNG) からマップを作るが、ここでは
    maneuver data がすでに「絶対入力フレーム + スキャン座標」なので、
    _build_abs_maps() をオーバーライドしてそのまま渡す。
    """

    def __init__(self, lang="ja"):
        super().__init__(lang=lang)
        self._man_space = None      # (T, S) 0..1
        self._man_time = None       # (T, S) 絶対入力フレーム

    def set_maneuver(self, data, scan_full):
        """dm.data をマップ解像度へリサンプルして保持する。"""
        if data is None or len(data) == 0:
            self._man_space = self._man_time = None
            return
        sp = data[:, :, 0].astype(np.float32) / max(1.0, float(scan_full) - 1.0)
        tm = data[:, :, 1].astype(np.float32)
        self._man_space = cv2.resize(sp, (MAP_S_RES, MAP_T_RES),
                                     interpolation=cv2.INTER_LINEAR)
        self._man_time = cv2.resize(tm, (MAP_S_RES, MAP_T_RES),
                                    interpolation=cv2.INTER_LINEAR)

    def _build_abs_maps(self):
        if self._man_time is None:
            return super()._build_abs_maps()
        return {"space": np.ascontiguousarray(self._man_space),
                "time": np.ascontiguousarray(self._man_time),
                "rate": np.ascontiguousarray(self._man_time)}


# ======== レンダリング ワーカー ========
class RenderWorker(QThread):
    """本番レンダリング: チェーンを本番スリット数で再実行 → transprocess → 音声。"""
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str)

    def __init__(self, dm, specs, out_type=1, separate_num=None,
                 audio_out=False, audio_kwargs=None, audio_voices=20,
                 audio_only=False, video_path="", full_data=None,
                 imgtype=None, still_split_minutes=None, scan_step=1):
        super().__init__()
        self.dm = dm
        self.specs = specs
        self.out_type = out_type
        self.separate_num = separate_num
        self.audio_out = audio_out
        # audio_render / audio_video_out へそのまま渡す全パラメータ
        self.audio_kwargs = dict(audio_kwargs or {})
        self.audio_mode = self.audio_kwargs.get("mode", "play")
        self.audio_voices = max(2, int(audio_voices))
        self.imgtype = imgtype     # out_type=0 (連番画像) のときの拡張子
        # 1 フレームのデータを時間で区切って複数枚出力するときの 1 枚あたりの分数
        self.still_split_minutes = still_split_minutes
        self.scan_step = max(1, int(scan_step))
        # 音声のみ再書き出し: チェーンが前回レンダリングと同一のとき、
        # 保存済みの映像 (video_path) に対して audio_video_out だけ実行する
        self.audio_only = audio_only
        self.video_path = video_path
        self.full_data = full_data
        self.video_only_path = ""     # 音声 mux 前の映像パス (呼び出し側が保存)

    def run(self):
        try:
            if self.audio_only:
                self._run_audio_only()
                return
            # 1) プレビューはプロキシ解像度なので、本番スリット数で作り直す
            self.log_signal.emit(
                tr("render_full_chain", n=self.dm.scan_nums))
            ok, msg = run_specs_plain(self.dm, self.specs,
                                      log_cb=self.log_signal.emit)
            if not ok:
                self.log_signal.emit("[ERROR] " + msg)
                self.done_signal.emit(False, "")
                return
            d = self.dm.data
            self.log_signal.emit(
                f"full data: {d.shape}  z {d[:, :, 1].min():.0f}"
                f"–{d[:, :, 1].max():.0f}")

            # 1 フレームのデータを時間で区切って複数枚に分ける。
            # 最後の区間だけ短くなるため幅が揃わず 1 回では出せないので、
            # SlitScan.py と同じくチャンクごとに transprocess を回す。
            if self.still_split_minutes and len(d) == 1 and self.out_type == 0:
                self._run_still_chunks(d)
                return

            # 2) レンダリング
            if self.imgtype:
                self.dm.imgtype = self.imgtype
            self.log_signal.emit(
                f"=== new_transprocess (out_type={self.out_type}"
                + (f", imgtype={self.imgtype}" if self.out_type == 0 else "")
                + ") ===")
            self.dm.new_transprocess(separate_num=self.separate_num,
                                     out_type=self.out_type, del_data=False)
            path = getattr(self.dm, "out_videopath", "") or ""
            if path and not os.path.isabs(path):
                path = os.path.abspath(path)
            self.video_only_path = path if os.path.exists(path) else ""
            # 動画出力なのにファイルが出来ていないなら失敗として扱う。
            # new_transprocess は z<0 などを内部で握り潰して戻るため、
            # ここで検知しないと「完了」と報告してしまう。
            if self.out_type != 0 and not self.video_only_path:
                self.log_signal.emit(tr("render_no_output"))
                self.done_signal.emit(False, "")
                return

            # 3) 音声 (プレビュー設定と同じく audio_video_out へ)
            if self.audio_out and path and os.path.exists(path):
                try:
                    self.log_signal.emit(
                        f"=== audio_video_out ({self._audio_desc()}) ===")
                    final = self.dm.audio_video_out(
                        thread_num=self.audio_voices, **self.audio_kwargs)
                    if final and os.path.exists(final):
                        path = os.path.abspath(final)
                except Exception as e:
                    self.log_signal.emit(f"[WARN] audio_video_out failed: {e} — "
                                         "音声なしの出力を使用します")
            self.done_signal.emit(True, path if os.path.exists(path) else "")
        except Exception as e:
            self.log_signal.emit("[ERROR] " + "".join(
                traceback.format_exception_only(type(e), e)).strip())
            self.done_signal.emit(False, "")
        finally:
            try:
                plt.close("all")
            except Exception:
                pass

    def _run_still_chunks(self, data):
        """1 枚ぶんの積み重ねを時間で区切り、区間ごとに 1 枚ずつ書き出す。"""
        chunks = plan_still_chunks(data, self.dm.recfps,
                                   self.still_split_minutes)
        if not chunks:
            self.log_signal.emit(tr("render_no_output"))
            self.done_signal.emit(False, "")
            return
        n, total, _each = still_chunk_summary(chunks)
        self.log_signal.emit(tr("still_split_applied",
                                total=_fmt_dur(total), n=n))
        if self.imgtype:
            self.dm.imgtype = self.imgtype
        for i, (i0, i1, s_sec, e_sec) in enumerate(chunks, 1):
            self.log_signal.emit(tr(
                "still_chunk_run", i=i, n=n, s=_fmt_dur(s_sec),
                e=_fmt_dur(e_sec), w=(i1 - i0) // self.scan_step))
            # SlitScan.py と同じ手順: 区間ごとに data と命名状態を作り直す
            self.dm.data = np.ascontiguousarray(data[:, i0:i1, :])
            self.dm.log = 0
            self.dm.out_name_attr = ""
            self.dm.sepVideoOut = 1
            self.dm.new_transprocess(
                out_type=0, scan_step=self.scan_step, del_data=False,
                use_pyav=True,
                title_atr=f"_{s_sec / 60:.0f}-{e_sec / 60:.0f}min")
        self.done_signal.emit(True, "")

    def _audio_desc(self):
        fx = [k.replace("depth_", "") for k, v in self.audio_kwargs.items()
              if k.startswith("depth_") and v is True]
        s = f"mode={self.audio_mode}, voices={self.audio_voices}"
        if fx:
            s += ", fx=" + "+".join(fx)
        return s

    def _run_audio_only(self):
        """映像の再レンダリングを省略し、保存済み映像へ音声だけ書き出す。"""
        try:
            self.log_signal.emit(
                tr("audio_only_render", v=os.path.basename(self.video_path)))
            # 音声軌跡は本番解像度の data が必要 (プレビューはプロキシのため)
            if self.full_data is not None:
                self.dm.data = self.full_data
            elif not (isinstance(getattr(self.dm, "data", None), np.ndarray)
                      and len(self.dm.data) > 0):
                ok, msg = run_specs_plain(self.dm, self.specs,
                                          log_cb=self.log_signal.emit)
                if not ok:
                    self.log_signal.emit("[ERROR] " + msg)
                    self.done_signal.emit(False, "")
                    return
            self.video_only_path = self.video_path
            self.log_signal.emit(
                f"=== audio_video_out ({self._audio_desc()}) ===")
            final = self.dm.audio_video_out(
                videopath=self.video_path,
                thread_num=self.audio_voices, **self.audio_kwargs)
            if final and os.path.exists(final):
                self.done_signal.emit(True, os.path.abspath(final))
            else:
                self.done_signal.emit(False, "")
        except Exception as e:
            self.log_signal.emit("[ERROR] " + "".join(
                traceback.format_exception_only(type(e), e)).strip())
            self.done_signal.emit(False, "")
        finally:
            try:
                plt.close("all")
            except Exception:
                pass


# ======== メイン GUI ========
class DrawManeuverGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.setAcceptDrops(True)   # Finder から動画をドロップできる
        self.resize(1560, 980)
        self.setMinimumSize(900, 640)

        self.videopath = None
        self.videopath_src = None
        self._work_dir = os.getcwd()
        self._status_active = False  # 進捗行を書き換え中か
        self._plot3d_worker = None   # 3D プロット書き出しスレッド
        self._saving_preview = False # プレビュー簡易保存の実行中フラグ
        self._was_single = None      # 直前が1フレームだったか
        self._auto_still = False     # 連番画像へ自動で切り替えたか
        self._last_3d_png = ""       # 直近の 3D プロット静止画
        self._plot_full = (None, None)  # 原寸表示のソース
        self._rt_built_key = None    # GPU プレビューを構築した時点のチェーン指紋
        self._output_path = ""      # 直近の書き出し先
        self._output_is_dir = False
        self.dm = None
        self.steps = []                 # StepWidget のリスト
        self.runner = PipelineRunner()
        self._vid_cap = None
        self._vid_info = None           # (w, h, fps, n, dur)
        self._vid_pos = -1
        self._vid_frame_cache = None
        self._vrot_worker = None
        self._render_worker = None
        self._i18n = []

        # チェーン変更 → デバウンスして再構築
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(350)
        self._rebuild_timer.timeout.connect(self.rebuild_pipeline)
        # 操作履歴 (ステップ列のスナップショット) — Undo/Redo 用
        self._history = []
        self._hist_pos = -1
        self._hist_lock = False
        self._dirty = False           # 手動更新モードでの未反映フラグ
        self._rendering = False
        # 前回レンダリングの記録 (チェーン不変なら音声のみ再書き出しに使う)
        self._last_render_key = None   # チェーン+レンダリング条件の指紋
        self._last_video_path = ""     # 音声 mux 前の映像パス
        self._last_full_data = None    # 本番解像度の data (音声軌跡用)

        self.init_ui()
        self._update_gates()

    # ---- i18n ----
    def _reg(self, fn):
        self._i18n.append(fn)
        fn()

    def _trlabel(self, key, **fmt):
        lbl = QLabel()
        self._reg(lambda l=lbl, k=key, f=fmt: l.setText(tr(k, **f)))
        return lbl

    def on_language_changed(self, *_):
        global LANG
        sel = self.lang_select.currentData()
        if sel in ("ja", "en") and sel != LANG:
            LANG = sel
            self.setWindowTitle(tr("window_title"))
            for fn in self._i18n:
                try:
                    fn()
                except Exception:
                    pass
            self._refresh_method_combo()
            if getattr(self, "rt_preview", None):
                self.rt_preview.set_lang(LANG)

    # ---- UI ----
    def init_ui(self):
        # --- 言語 ---
        self.lang_select = QComboBox()
        self.lang_select.addItem("日本語", "ja")
        self.lang_select.addItem("English", "en")
        self.lang_select.setCurrentIndex(0 if LANG == "ja" else 1)
        self.lang_select.currentIndexChanged.connect(self.on_language_changed)
        lang_row = QHBoxLayout()
        lang_row.addWidget(self._trlabel("lang_label"))
        lang_row.addWidget(self.lang_select)
        lang_row.addStretch()

        # --- 入力映像 (Shape_of_time_flow をトレース) ---
        setup = QGroupBox()
        self._reg(lambda b=setup: b.setTitle(tr("grp_setup")))
        sg = QVBoxLayout(setup)
        sg.addLayout(lang_row)

        self.video_btn = QPushButton()
        self._reg(lambda: (self.video_btn.setText(tr("btn_select_video")),
                           self.video_btn.setToolTip(tr("tip_drop_video"))))
        self.video_btn.clicked.connect(self.select_video)
        sg.addWidget(self.video_btn)
        self.video_label = QLabel(tr("no_video"))
        self.video_label.setWordWrap(True)
        self.video_label.setStyleSheet("color:gray; font-size:10px;")
        self._i18n.append(lambda: (None if self.videopath_src
                                   else self.video_label.setText(tr("no_video"))))
        sg.addWidget(self.video_label)

        # 未選択のあいだはドロップ先であることを示すプレースホルダを出す
        self.video_preview = QLabel(tr("drop_placeholder"))
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.video_preview.setFixedHeight(150)
        self.video_preview.setWordWrap(True)
        self._reg(lambda: (None if self._vid_info
                           else self.video_preview.setText(tr("drop_placeholder"))))
        self._style_video_preview(empty=True)
        sg.addWidget(self.video_preview)

        # 再生 / Finder で表示 (映像を読み込むまでは隠す)
        vp_row = QHBoxLayout()
        self.play_btn = QPushButton()
        self.play_btn.setCheckable(True)
        self._reg(lambda: (self.play_btn.setText(
                               tr("btn_pause") if self.play_btn.isChecked()
                               else tr("btn_play")),
                           self.play_btn.setToolTip(tr("tip_play"))))
        self.play_btn.toggled.connect(self._on_play_toggled)
        self.play_btn.setVisible(False)
        vp_row.addWidget(self.play_btn)
        self.reveal_btn = QPushButton()
        self._reg(lambda: (self.reveal_btn.setText(tr("btn_reveal_video")),
                           self.reveal_btn.setToolTip(tr("tip_reveal_video"))))
        self.reveal_btn.clicked.connect(self.reveal_video_in_finder)
        self.reveal_btn.setVisible(False)
        vp_row.addWidget(self.reveal_btn)
        vp_row.addStretch()
        sg.addLayout(vp_row)
        # 再生は壁時計駆動 (重い素材ではフレームを落として実時間を保つ)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._play_t0 = 0.0
        self._play_i0 = 0
        self.video_dim_label = QLabel("")
        self.video_dim_label.setStyleSheet("color:gray; font-size:10px;")
        sg.addWidget(self.video_dim_label)

        # 入力映像の時間軸 + 使用範囲バンド (Shape_of_time_flow からの移植)
        self.timeline_label = self._trlabel("lbl_timeline")
        self.timeline_label.setStyleSheet("font-size:11px;")
        self.timeline_label.setVisible(False)
        sg.addWidget(self.timeline_label)
        self.timeline = ManeuverTimelineSlider()
        self.timeline.playheadChanged.connect(self._on_scrub)
        self.timeline.usedRangeSlid.connect(self._on_used_range_slid)
        self.timeline.setVisible(False)
        sg.addWidget(self.timeline)
        self.timeline_readout = QLabel("")
        self.timeline_readout.setStyleSheet("color:gray; font-size:10px;")
        self.timeline_readout.setVisible(False)
        sg.addWidget(self.timeline_readout)
        self.timeline_hint = self._trlabel("timeline_hint")
        self.timeline_hint.setStyleSheet("color:gray; font-size:10px;")
        self.timeline_hint.setWordWrap(True)
        self.timeline_hint.setVisible(False)
        sg.addWidget(self.timeline_hint)

        self.slit_toggle = QCheckBox()
        self._reg(lambda: self.slit_toggle.setText(tr("chk_vertical")))
        self.slit_toggle.stateChanged.connect(self._on_slit_changed)
        sg.addWidget(self.slit_toggle)
        self.slit_label = QLabel(tr("slit_h"))
        self._reg(lambda: self.slit_label.setText(
            tr("slit_v") if self.slit_toggle.isChecked() else tr("slit_h")))
        sg.addWidget(self.slit_label)

        vrow = QHBoxLayout()
        vrow.addWidget(self._trlabel("lbl_video_rotate"))
        self.vrot_combo = QComboBox()
        for rid, key, _vf in VIDEO_ROTATIONS:
            self.vrot_combo.addItem(tr(key), rid)
        self._reg(lambda: [self.vrot_combo.setItemText(i, tr(k))
                           for i, (_r, k, _v) in enumerate(VIDEO_ROTATIONS)])
        self.vrot_combo.currentIndexChanged.connect(self._on_rotation_changed)
        vrow.addWidget(self.vrot_combo, 1)
        sg.addLayout(vrow)
        hint = self._trlabel("hint_video_rotate")
        hint.setStyleSheet("color:gray; font-size:10px;")
        hint.setWordWrap(True)
        sg.addWidget(hint)

        frow = QHBoxLayout()
        frow.addWidget(self._trlabel("lbl_outfps"))
        self.outfps_combo = QComboBox()
        for f in (10, 24, 25, 30, 60, 120):
            self.outfps_combo.addItem(str(f), f)
        self.outfps_combo.setCurrentIndex(3)
        self.outfps_combo.currentIndexChanged.connect(self._on_outfps_changed)
        frow.addWidget(self.outfps_combo)
        frow.addStretch()
        sg.addLayout(frow)

        self.init_btn = QPushButton()
        self._reg(lambda: self.init_btn.setText(tr("btn_initialize")))
        self.init_btn.clicked.connect(self.initialize_dm)
        self.init_btn.setEnabled(False)
        sg.addWidget(self.init_btn)
        # 初期化で作られた作業ディレクトリを Finder で開くリンク
        self.workdir_btn = QPushButton()
        self._reg(lambda: (self.workdir_btn.setText(tr("btn_open_workdir")),
                           self.workdir_btn.setToolTip(tr("tip_open_workdir"))))
        self.workdir_btn.setStyleSheet(
            "QPushButton { border:none; color:#2a6ebb; text-align:left;"
            " padding:2px 0; }"
            "QPushButton:hover { text-decoration:underline; }")
        self.workdir_btn.clicked.connect(self.open_work_dir)
        self.workdir_btn.setVisible(False)
        sg.addWidget(self.workdir_btn)
        self.workdir_files_label = self._trlabel("lbl_workdir_files")
        self.workdir_files_label.setStyleSheet("font-size:11px;")
        self.workdir_files_label.setVisible(False)
        sg.addWidget(self.workdir_files_label)
        self.workdir_browser = WorkDirBrowser()
        self.workdir_browser.setVisible(False)
        sg.addWidget(self.workdir_browser)
        self.info_label = QLabel(tr("video_not_init"))
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color:gray; font-size:10px;")
        self._i18n.append(lambda: (None if self.dm
                                   else self.info_label.setText(tr("video_not_init"))))
        sg.addWidget(self.info_label)
        sg.addStretch()

        # --- メソッドチェーン ---
        chain_group = QGroupBox()
        self._reg(lambda b=chain_group: b.setTitle(tr("grp_chain")))
        cg = QVBoxLayout(chain_group)

        self.steps_box = QVBoxLayout()
        self.steps_box.setSpacing(3)
        self.empty_label = QLabel(tr("chain_empty"))
        self.empty_label.setStyleSheet("color:gray; font-size:11px;")
        self._i18n.append(lambda: self.empty_label.setText(tr("chain_empty")))
        self.steps_box.addWidget(self.empty_label)   # index 0
        self.steps_box.addStretch(1)                 # 末尾: カードを上詰めにする
        steps_holder = QWidget()
        steps_holder.setLayout(self.steps_box)
        scroll = QScrollArea()
        scroll.setWidget(steps_holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        # 種別/メソッドの選択はチェーンの「上」に置く (追加 → 下に並ぶ流れ)
        addrow = QHBoxLayout()
        addrow.addWidget(self._trlabel("lbl_category"))
        self.cat_combo = QComboBox()
        for c in CATEGORY_ORDER:
            self.cat_combo.addItem(category_icon(c), tr(CATEGORY_LABEL_KEY[c]), c)
        self._reg(lambda: [self.cat_combo.setItemText(i, tr(CATEGORY_LABEL_KEY[c]))
                           for i, c in enumerate(CATEGORY_ORDER)])
        self.cat_combo.currentIndexChanged.connect(self._refresh_method_combo)
        addrow.addWidget(self.cat_combo, 1)
        addrow.addWidget(self._trlabel("lbl_method"))
        self.method_combo = QComboBox()
        self.method_combo.setMinimumWidth(220)
        addrow.addWidget(self.method_combo, 2)
        self.add_step_btn = QPushButton()
        self._reg(lambda: self.add_step_btn.setText(tr("btn_add_step")))
        self.add_step_btn.clicked.connect(self._add_step_from_combo)
        addrow.addWidget(self.add_step_btn)
        # 右端: メソッド一覧 + ヘルプの別ウィンドウ
        self.help_btn = QToolButton()
        self._reg(lambda: (self.help_btn.setText(tr("btn_help_methods")),
                           self.help_btn.setToolTip(tr("tip_help_methods"))))
        self.help_btn.setStyleSheet(
            "QToolButton { border:1px solid #888; border-radius:11px; min-width:22px;"
            " min-height:22px; font-weight:bold; }")
        self.help_btn.clicked.connect(self.open_method_help)
        addrow.addWidget(self.help_btn)
        cg.addLayout(addrow)
        cg.addWidget(scroll, 1)
        self._help_window = None
        self._refresh_method_combo()

        # --- 2D プロット + 実行 ---
        plot_group = QGroupBox()
        self.plot_group = plot_group   # rt_group とのスプリッタ配分調整で参照する
        self._reg(lambda b=plot_group: b.setTitle(tr("grp_plot")))
        pg = QVBoxLayout(plot_group)

        ctrl = QHBoxLayout()
        self.undo_btn = QToolButton()
        self.undo_btn.setText("↶")
        self._reg(lambda: self.undo_btn.setToolTip(tr("tip_undo")))
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setShortcut("Ctrl+Z")
        ctrl.addWidget(self.undo_btn)
        self.redo_btn = QToolButton()
        self.redo_btn.setText("↷")
        self._reg(lambda: self.redo_btn.setToolTip(tr("tip_redo")))
        self.redo_btn.clicked.connect(self.redo)
        self.redo_btn.setShortcut("Ctrl+Shift+Z")
        ctrl.addWidget(self.redo_btn)
        ctrl.addSpacing(10)
        self.auto_update_chk = QCheckBox()
        self._reg(lambda: self.auto_update_chk.setText(tr("chk_auto_update")))
        self.auto_update_chk.setChecked(True)
        self.auto_update_chk.toggled.connect(self._on_auto_toggled)
        ctrl.addWidget(self.auto_update_chk)
        self.update_btn = QPushButton()
        self._reg(lambda: self.update_btn.setText(
            tr("btn_update_dirty") if self._dirty else tr("btn_update_now")))
        self.update_btn.clicked.connect(self._manual_update)
        self.update_btn.setEnabled(False)
        ctrl.addWidget(self.update_btn)
        ctrl.addSpacing(10)
        ctrl.addWidget(self._trlabel("lbl_preview_scan"))
        self.preview_scan_spin = QSpinBox()
        self.preview_scan_spin.setRange(3, 1024)
        self.preview_scan_spin.setValue(19)
        self._reg(lambda: self.preview_scan_spin.setToolTip(tr("tip_preview_scan")))
        self.preview_scan_spin.valueChanged.connect(
            lambda *_: self._schedule_rebuild())
        ctrl.addWidget(self.preview_scan_spin)
        ctrl.addStretch()
        pg.addLayout(ctrl)

        # 出力データの形状。2D プロットの「上」に常時表示する。
        # データが組み上がって以降、何を書き出そうとしているのかが
        # 常に目に入るようにするため、プロット画像とは分けて出す。
        self.data_info = QLabel("")
        self.data_info.setAlignment(Qt.AlignCenter)
        self.data_info.setWordWrap(True)
        self.data_info.setTextFormat(Qt.RichText)
        self.data_info.setStyleSheet(
            "QLabel { background:#eef4fb; border:1px solid #b8cfe6;"
            " border-radius:4px; color:#1f3d5c; font-size:12px;"
            " font-weight:bold; padding:5px 8px; }")
        self.data_info.setVisible(False)
        pg.addWidget(self.data_info)

        self.plot_label = ClickableLabel(tr("plot_waiting"))
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.doubleClicked.connect(self.open_plot_fullsize)
        self._reg(lambda: self.plot_label.setToolTip(tr("tip_plot_dblclick")))
        # 縦が足りないときは設定パネルより先にプロットが縮むようにする
        self.plot_label.setMinimumSize(360, 180)
        self.plot_label.setStyleSheet(
            "QLabel { background:#ffffff; border:1px solid #555; color:#888; }")
        self._i18n.append(lambda: (None if self.plot_label.pixmap()
                                   else self.plot_label.setText(tr("plot_waiting"))))
        pg.addWidget(self.plot_label, 1)

        # zPointCheck 警告 (time が負/範囲超過のときだけ出現し、
        # 押すとチェーン末尾に zPointCheck ステップを追加する)
        self.zcheck_btn = QPushButton()
        self._reg(lambda: self.zcheck_btn.setText(tr("zcheck_warn")))
        self.zcheck_btn.setStyleSheet(
            "QPushButton { background:#fdf2cc; color:#9a6b00;"
            " border:1px solid #e0b34d; border-radius:4px; padding:4px; }"
            "QPushButton:hover { background:#f7e3a1; }")
        self.zcheck_btn.clicked.connect(self.append_zcheck_step)
        self.zcheck_btn.setVisible(False)
        pg.addWidget(self.zcheck_btn)

        actions = QHBoxLayout()
        self.full2d_btn = QPushButton()
        self._reg(lambda: self.full2d_btn.setText(tr("btn_full2d")))
        self.full2d_btn.clicked.connect(self.run_full_2dplot)
        actions.addWidget(self.full2d_btn)
        # 3D プロットアニメーション: フレームが複数あるときだけ出す
        self.full3d_btn = QPushButton()
        self._reg(lambda: self.full3d_btn.setText(tr("btn_full3d")))
        self.full3d_btn.clicked.connect(self.run_full_3dplot)
        self.full3d_btn.setVisible(False)
        actions.addWidget(self.full3d_btn)
        self.export_btn = QPushButton()
        self._reg(lambda: self.export_btn.setText(tr("btn_export")))
        self.export_btn.clicked.connect(self.export_script)
        actions.addWidget(self.export_btn)
        pg.addLayout(actions)

        # --- 出力設定 / 音声設定 ---
        # 縦が足りないときは中身を潰さずスクロールさせる。低い画面でも
        # 全項目に手が届くようにするため (レンダリングボタンは外に置く)。
        self.settings_host = QWidget()
        sh = QVBoxLayout(self.settings_host)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.setSpacing(4)
        sh.addWidget(self._build_output_panel())
        sh.addWidget(self._build_audio_panel())
        sh.addStretch()
        self.settings_scroll = FitScrollArea()
        self.settings_scroll.setWidget(self.settings_host)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_scroll.setSizePolicy(QSizePolicy.Preferred,
                                           QSizePolicy.Maximum)
        for acc in (self.settings_host.findChildren(Accordion)):
            acc.button.toggled.connect(lambda *_: self._sync_settings_height())
        pg.addWidget(self.settings_scroll)
        self._sync_settings_height()

        render_row = QHBoxLayout()
        self.render_btn = QPushButton()
        self._reg(lambda: self.render_btn.setText(tr("btn_render")))
        self.render_btn.clicked.connect(self.start_render)
        render_row.addWidget(self.render_btn, 1)
        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 0)      # 不定 (走行中のみ表示)
        self.render_progress.setVisible(False)
        render_row.addWidget(self.render_progress, 1)
        pg.addLayout(render_row)

        # 書き出したものを既定のアプリで開くボタン。
        # レンダリングが成功したときだけ出す。
        self.open_output_btn = QPushButton()
        self._reg(lambda: self.open_output_btn.setText(
            tr("btn_open_output_dir") if self._output_is_dir
            else tr("btn_open_output")))
        self.open_output_btn.setStyleSheet(
            "QPushButton { background:#e8f4ea; color:#1f5c33;"
            " border:1px solid #a9cfb5; border-radius:4px; padding:5px; }"
            "QPushButton:hover { background:#d5ebdb; }")
        self.open_output_btn.clicked.connect(self.open_rendered_output)
        self.open_output_btn.setVisible(False)

        # GPU プレビューの見た目をそのまま作業ディレクトリへ簡易保存する
        self.save_preview_btn = QPushButton()
        self._reg(lambda: (self.save_preview_btn.setText(tr("btn_save_preview")),
                           self.save_preview_btn.setToolTip(tr("tip_save_preview"))))
        self.save_preview_btn.clicked.connect(self.save_preview_video)

        out_row = QHBoxLayout()
        out_row.addWidget(self.save_preview_btn, 1)
        out_row.addWidget(self.open_output_btn, 1)
        pg.addLayout(out_row)

        # --- GPU プレビュー ---
        if _HAS_RT_PREVIEW:
            self.rt_group = QGroupBox()
            self._reg(lambda: self.rt_group.setTitle(tr("grp_realtime")))
            rv = QVBoxLayout(self.rt_group)
            rv.setContentsMargins(4, 4, 4, 4)
            self.rt_preview = ManeuverRTPreview(lang=LANG)
            self.rt_preview.previewAspectReady.connect(self._on_preview_aspect)
            self.rt_preview.rebuilt.connect(self._on_rt_rebuilt)
            rv.addWidget(self.rt_preview)
        else:
            self.rt_group = None
            self.rt_preview = None

        # --- レイアウト: [入力] [チェーン] [プロット + GPU] ---
        cols = QSplitter(Qt.Horizontal)
        left = QWidget(); ll = QVBoxLayout(left); ll.addWidget(setup)
        cols.addWidget(left)
        cols.addWidget(chain_group)
        # プロット/設定 と GPU プレビューの配分は可動にする。固定比だと
        # 画面が低いときに設定パネル側が必要な高さを貰えず潰れてしまう。
        if self.rt_group is not None:
            right = QSplitter(Qt.Vertical)
            right.addWidget(plot_group)
            right.addWidget(self.rt_group)
            right.setStretchFactor(0, 3)
            right.setStretchFactor(1, 4)
            self._right_split = right
        else:
            right = QWidget(); rl = QVBoxLayout(right)
            rl.addWidget(plot_group)
            self._right_split = None
        cols.addWidget(right)
        cols.setStretchFactor(0, 2)
        cols.setStretchFactor(1, 3)
        cols.setStretchFactor(2, 4)

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setMaximumHeight(140)
        log_box = QWidget()
        lb = QVBoxLayout(log_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.addWidget(self._trlabel("lbl_log"))
        lb.addWidget(self.log_window)

        outer_split = QSplitter(Qt.Vertical)
        outer_split.addWidget(cols)
        outer_split.addWidget(log_box)
        outer_split.setStretchFactor(0, 6)
        outer_split.setStretchFactor(1, 1)

        v = QVBoxLayout()
        v.setContentsMargins(6, 6, 6, 6)
        v.addWidget(outer_split)
        self.setLayout(v)

    def _sync_settings_height(self):
        """設定スクロール領域の高さを中身に合わせる。

        中身が収まるときはスクロールバーを出さずそのまま表示し、
        画面が低いときだけスクロールに切り替わるようにする。
        """
        need = self.settings_host.sizeHint().height()
        self.settings_scroll.setMaximumHeight(need)
        # 低い画面ではここまで縮み、中身はスクロールで辿れる
        self.settings_scroll.setMinimumHeight(min(need, 120))
        self.settings_scroll.updateGeometry()
        self._grow_settings_pane(need)

    def _grow_settings_pane(self, need):
        """アコーディオンを開いた分の高さを GPU プレビュー側から回してくる。

        右列は可動スプリッタなので、開いたときに自動で配分を寄せておかないと
        既定の比率のままスクロール表示になり、設定項目が見えにくい。
        GPU プレビューの最小高は割らない (割る分はスクロールで辿れる)。
        """
        sp = getattr(self, "_right_split", None)
        if sp is None or self.rt_group is None:
            return
        sizes = sp.sizes()
        if len(sizes) != 2:
            return
        short = need - self.settings_scroll.height()
        spare = sizes[1] - self.rt_group.minimumSizeHint().height()
        move = max(0, min(short, spare))
        if move:
            sp.setSizes([sizes[0] + move, sizes[1] - move])

    # 右列スプリッタの既定配分 (plot_group : rt_group)。setStretchFactor と揃える。
    _RIGHT_SPLIT_DEFAULT = (3, 4)

    def _reset_right_split(self):
        """右列の高さ配分を既定比率へ戻す。

        _on_preview_aspect は縦長素材のぶんだけ rt_group 側を広げるが、
        戻す判断はしない (常時判定すると横長へ切り替えた瞬間に縮んで
        ちらつく)。新しい映像を読み込むタイミングでまとめてリセットする。
        """
        sp = getattr(self, "_right_split", None)
        if sp is None:
            return
        total = sum(sp.sizes())
        if total <= 0:
            return
        a, b = self._RIGHT_SPLIT_DEFAULT
        sp.setSizes([int(total * a / (a + b)), int(total * b / (a + b))])

    def _on_preview_aspect(self, ratio):
        """GPU プレビューの実際の映像アスペクト比が判明したら、縦長素材で
        プレビューが横長の箱の中に小さく押し込まれたままにならないよう、
        rt_group の高さの取り分を広げる。

        箱の幅はそのまま (横に並ぶ他パネルの構成を崩さない)、高さだけを
        映像の縦横比に近づける。横長/正方形の素材では現状より縮めない
        (plot_group から奪うのは、映像を大きく見せるために必要な分だけ)。
        """
        sp = getattr(self, "_right_split", None)
        if sp is None or self.rt_group is None or ratio <= 0:
            return
        sizes = sp.sizes()
        if len(sizes) != 2 or sum(sizes) <= 0:
            return
        total = sum(sizes)
        col_w = max(200, self.rt_group.width())
        want_h = int(col_w / ratio)
        floor_h = self.plot_group.minimumSizeHint().height()
        max_rt_h = max(1, total - floor_h)
        new_rt_h = min(max(want_h, sizes[1]), max_rt_h)
        if new_rt_h > sizes[1]:
            sp.setSizes([total - new_rt_h, new_rt_h])

    # ---- 出力設定パネル ----
    def _build_output_panel(self):
        """出力形式 (動画コーデック / 連番画像) のアコーディオン。"""
        acc = Accordion(tr("grp_output"))
        self._reg(lambda a=acc: a.set_title(tr("grp_output")))

        r1 = QHBoxLayout()
        r1.addWidget(self._trlabel("lbl_out_kind"))
        self.out_kind_combo = QComboBox()
        self.out_kind_combo.addItem(tr("out_video"), "video")
        self.out_kind_combo.addItem(tr("out_still"), "still")
        self._reg(lambda: (self.out_kind_combo.setItemText(0, tr("out_video")),
                           self.out_kind_combo.setItemText(1, tr("out_still"))))
        self.out_kind_combo.currentIndexChanged.connect(self._on_out_kind)
        r1.addWidget(self.out_kind_combo, 1)
        acc.addLayout(r1)

        self.out_type_row = QHBoxLayout()
        self.out_type_row.addWidget(self._trlabel("lbl_out_type"))
        self.out_type_combo = QComboBox()
        for val, label, ext, note in OUT_TYPES:
            self.out_type_combo.addItem(f"{label}  {ext}", val)
            self.out_type_combo.setItemData(
                self.out_type_combo.count() - 1, note, Qt.ToolTipRole)
        self.out_type_row.addWidget(self.out_type_combo, 1)
        acc.addLayout(self.out_type_row)

        self.img_row = QHBoxLayout()
        self.img_row.addWidget(self._trlabel("lbl_img_format"))
        self.img_format_combo = QComboBox()
        for ext, label in IMG_FORMATS:
            self.img_format_combo.addItem(label, ext)
        self.img_row.addWidget(self.img_format_combo, 1)
        acc.addLayout(self.img_row)
        self.img_hint = self._trlabel("hint_img_format")
        self.img_hint.setStyleSheet("color:gray; font-size:10px;")
        acc.addWidget(self.img_hint)

        # 1 フレームだけのデータ用: 全尺を 1 枚にするか時間で分割するか
        self.split_row = QHBoxLayout()
        self.split_row.addWidget(self._trlabel("lbl_still_split"))
        self.still_split_combo = QComboBox()
        self.still_split_combo.addItem(tr("still_split_one"), "one")
        self.still_split_combo.addItem(tr("still_split_time"), "time")
        self._reg(lambda: (
            self.still_split_combo.setItemText(0, tr("still_split_one")),
            self.still_split_combo.setItemText(1, tr("still_split_time"))))
        self.still_split_combo.currentIndexChanged.connect(
            self._sync_still_split)
        self.split_row.addWidget(self.still_split_combo, 1)
        self.split_row.addWidget(self._trlabel("lbl_split_minutes"))
        self.split_min_spin = QDoubleSpinBox()
        self.split_min_spin.setRange(0.05, 120.0)
        self.split_min_spin.setDecimals(2)
        self.split_min_spin.setSingleStep(0.5)
        self.split_min_spin.setValue(1.0)
        self.split_min_spin.valueChanged.connect(lambda *_: self._sync_still_split())
        self.split_row.addWidget(self.split_min_spin)
        self.split_unit_label = self._trlabel("unit_minutes")
        self.split_row.addWidget(self.split_unit_label)
        # 時間軸の間引き (出力幅 = 区間フレーム数 / scan_step)
        self.scan_step_label = self._trlabel("lbl_scan_step")
        self.split_row.addWidget(self.scan_step_label)
        self.scan_step_spin = QSpinBox()
        self.scan_step_spin.setRange(1, 64)
        self.scan_step_spin.setValue(1)
        self._reg(lambda: self.scan_step_spin.setToolTip(tr("tip_scan_step")))
        self.scan_step_spin.valueChanged.connect(lambda *_: self._sync_still_split())
        self.split_row.addWidget(self.scan_step_spin)
        self.split_row.addStretch()
        acc.addLayout(self.split_row)
        self.split_info = QLabel("")
        self.split_info.setStyleSheet("color:#1f3d5c; font-size:11px;")
        acc.addWidget(self.split_info)
        self.split_hint = self._trlabel("hint_still_split")
        self.split_hint.setStyleSheet("color:gray; font-size:10px;")
        self.split_hint.setWordWrap(True)
        acc.addWidget(self.split_hint)

        r3 = QHBoxLayout()
        r3.addWidget(self._trlabel("lbl_separate"))
        self.separate_spin = QSpinBox()
        self.separate_spin.setRange(0, 999)
        self.separate_spin.setValue(0)
        r3.addWidget(self.separate_spin)
        h = self._trlabel("hint_separate")
        h.setStyleSheet("color:gray; font-size:10px;")
        r3.addWidget(h)
        r3.addStretch()
        acc.addLayout(r3)

        self._on_out_kind()
        return acc

    def _sync_still_split(self, *_):
        """1枚データ用の分割設定の表示と、分割枚数の見積もりを更新する。"""
        single = bool(self._was_single)
        still = self.out_kind_combo.currentData() == "still"
        show = single and still
        for i in range(self.split_row.count()):
            w = self.split_row.itemAt(i).widget()
            if w:
                w.setVisible(show)
        self.split_hint.setVisible(show)
        by_time = self.still_split_combo.currentData() == "time"
        for w in (self.split_min_spin, self.split_unit_label,
                  self.scan_step_label, self.scan_step_spin):
            w.setVisible(show and by_time)
        if not (show and by_time):
            self.split_info.setVisible(False)
            return
        data = getattr(self.dm, "data", None) if self.dm else None
        chunks = plan_still_chunks(
            data, getattr(self.dm, "recfps", 30.0), self.split_min_spin.value())
        n, total, each = still_chunk_summary(chunks)
        if n == 0:
            self.split_info.setVisible(False)
            return
        # 最後の区間だけ短くなる (SlitScan.py と同じ区切り方)
        tail = ""
        if n > 1 and abs(each[-1] - each[0]) > 0.05:
            tail = tr("still_split_tail", last=_fmt_dur(each[-1]))
        self.split_info.setText(
            tr("still_split_info", total=_fmt_dur(total),
               each=_fmt_dur(each[0]), n=n) + tail)
        self.split_info.setVisible(True)

    def still_split_minutes(self):
        """分割して出力するときの 1 枚あたりの分数 (分割しないなら None)。"""
        if not self._was_single:
            return None
        if self.out_kind_combo.currentData() != "still":
            return None
        if self.still_split_combo.currentData() != "time":
            return None
        return float(self.split_min_spin.value())

    def _on_out_kind(self, *_):
        """動画 / 連番画像 で表示する設定を切り替える。"""
        still = self.out_kind_combo.currentData() == "still"
        for i in range(self.out_type_row.count()):
            w = self.out_type_row.itemAt(i).widget()
            if w:
                w.setVisible(not still)
        for i in range(self.img_row.count()):
            w = self.img_row.itemAt(i).widget()
            if w:
                w.setVisible(still)
        self.img_hint.setVisible(still)
        self._sync_still_split()

    def selected_out_type(self):
        if self.out_kind_combo.currentData() == "still":
            return 0
        return int(self.out_type_combo.currentData())

    @staticmethod
    def _fx_box(layouts, *widgets):
        """FX の詳細行をまとめて表示/非表示できる入れ物にする。"""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        for lay in layouts:
            lay.setContentsMargins(0, 0, 0, 0)
            v.addLayout(lay)
        for w in widgets:
            v.addWidget(w)
        return box

    # ---- 音声設定パネル ----
    def _build_audio_panel(self):
        """audio_render の全パラメータを扱うアコーディオン。"""
        acc = Accordion(tr("grp_audio"))
        self._reg(lambda a=acc: a.set_title(tr("grp_audio")))

        r0 = QHBoxLayout()
        self.audio_chk = QCheckBox()
        self._reg(lambda: self.audio_chk.setText(tr("chk_audio")))
        self.audio_chk.toggled.connect(self._on_audio_toggled)
        r0.addWidget(self.audio_chk)
        self.audio_mode_combo = QComboBox()
        self.audio_mode_combo.addItem(tr("audio_mode_play"), "play")
        self.audio_mode_combo.addItem(tr("audio_mode_grain"), "grain")
        self._reg(lambda: (
            self.audio_mode_combo.setItemText(0, tr("audio_mode_play")),
            self.audio_mode_combo.setItemText(1, tr("audio_mode_grain"))))
        r0.addWidget(self.audio_mode_combo, 1)
        info = QToolButton()
        info.setText("ⓘ")
        info.setAutoRaise(True)
        self._reg(lambda b=info: b.setToolTip(tr("tip_info")))
        info.clicked.connect(lambda: show_method_doc(self, AUDIO_PARAM_DOC))
        r0.addWidget(info)
        acc.addLayout(r0)

        r1 = QHBoxLayout()
        r1.addWidget(self._trlabel("lbl_audio_voices"))
        self.audio_voices_spin = QSpinBox()
        self.audio_voices_spin.setRange(2, 128)
        self.audio_voices_spin.setValue(20)
        r1.addWidget(self.audio_voices_spin)
        r1.addWidget(self._trlabel("lbl_audio_gain"))
        self.audio_gain_spin = QDoubleSpinBox()
        self.audio_gain_spin.setRange(0.0, 8.0)
        self.audio_gain_spin.setSingleStep(0.1)
        self.audio_gain_spin.setValue(1.0)
        r1.addWidget(self.audio_gain_spin)
        self.audio_normalize_chk = QCheckBox()
        self._reg(lambda: self.audio_normalize_chk.setText(
            tr("chk_audio_normalize")))
        self.audio_normalize_chk.setChecked(True)
        r1.addWidget(self.audio_normalize_chk)
        r1.addStretch()
        acc.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(self._trlabel("lbl_audio_smooth"))
        self.audio_smooth_combo = QComboBox()
        self.audio_smooth_combo.addItem("fourier", "fourier")
        self.audio_smooth_combo.addItem("spline", "spline")
        r2.addWidget(self.audio_smooth_combo)
        r2.addWidget(self._trlabel("lbl_audio_harmonics"))
        self.audio_harmonics_spin = QSpinBox()
        self.audio_harmonics_spin.setRange(0, 4096)
        self.audio_harmonics_spin.setValue(0)      # 0 = None (全成分)
        self.audio_harmonics_spin.setSpecialValueText("all")
        r2.addWidget(self.audio_harmonics_spin)
        r2.addWidget(self._trlabel("lbl_audio_inpan"))
        self.audio_inpan_combo = QComboBox()
        for v in ("balance", "gain", "none"):
            self.audio_inpan_combo.addItem(v, v)
        r2.addWidget(self.audio_inpan_combo)
        r2.addStretch()
        acc.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(self._trlabel("lbl_audio_grain"))
        self.audio_grain_spin = QDoubleSpinBox()
        self.audio_grain_spin.setRange(0.005, 1.0)
        self.audio_grain_spin.setDecimals(3)
        self.audio_grain_spin.setSingleStep(0.01)
        self.audio_grain_spin.setValue(0.1)
        r3.addWidget(self.audio_grain_spin)
        r3.addWidget(self._trlabel("lbl_audio_jump"))
        self.audio_jump_spin = QDoubleSpinBox()
        self.audio_jump_spin.setRange(0.01, 5.0)
        self.audio_jump_spin.setDecimals(3)
        self.audio_jump_spin.setSingleStep(0.05)
        self.audio_jump_spin.setValue(0.25)
        r3.addWidget(self.audio_jump_spin)
        r3.addStretch()
        acc.addLayout(r3)

        # --- now depth 駆動の変調 ---
        fxg = self._trlabel("grp_audio_fx")
        fxg.setStyleSheet("color:#555; font-size:10px; margin-top:4px;")
        acc.addWidget(fxg)
        f1 = QHBoxLayout()
        self.fx_reverb_chk = QCheckBox()
        self.fx_lpf_chk = QCheckBox()
        self.fx_width_chk = QCheckBox()
        self.fx_detune_chk = QCheckBox()
        for c, key in ((self.fx_reverb_chk, "fx_reverb"),
                       (self.fx_lpf_chk, "fx_lpf"),
                       (self.fx_width_chk, "fx_width"),
                       (self.fx_detune_chk, "fx_detune")):
            self._reg(lambda cc=c, k=key: cc.setText(tr(k)))
            f1.addWidget(c)
        f1.addStretch()
        acc.addLayout(f1)

        # --- リバーブ: 広さと残響時間は別の軸として並べる ---
        f2 = QHBoxLayout()
        f2.addWidget(self._trlabel("lbl_reverb_wet"))
        self.reverb_wet_spin = QDoubleSpinBox()
        self.reverb_wet_spin.setRange(0.0, 1.0)
        self.reverb_wet_spin.setSingleStep(0.05)
        self.reverb_wet_spin.setValue(0.4)
        f2.addWidget(self.reverb_wet_spin)
        f2.addWidget(self._trlabel("lbl_reverb_time"))
        self.reverb_time_spin = QDoubleSpinBox()
        self.reverb_time_spin.setRange(0.1, 20.0)
        self.reverb_time_spin.setSingleStep(0.1)
        self.reverb_time_spin.setValue(2.5)
        f2.addWidget(self.reverb_time_spin)
        f2.addWidget(self._trlabel("lbl_reverb_predelay"))
        self.reverb_predelay_spin = QDoubleSpinBox()
        self.reverb_predelay_spin.setRange(0.0, 0.5)
        self.reverb_predelay_spin.setDecimals(3)
        self.reverb_predelay_spin.setSingleStep(0.005)
        self.reverb_predelay_spin.setValue(0.048)
        f2.addWidget(self.reverb_predelay_spin)
        f2.addStretch()

        # 空間の広さ: 感覚的に効くのでスライダーで
        fr = QHBoxLayout()
        lab_room = self._trlabel("lbl_reverb_room")
        self._reg(lambda l=lab_room: l.setToolTip(tr("tip_reverb_room")))
        fr.addWidget(lab_room)
        self.reverb_room_slider = QSlider(Qt.Horizontal)
        self.reverb_room_slider.setRange(20, 400)      # 0.20 .. 4.00
        self.reverb_room_slider.setValue(100)          # 1.00
        self._reg(lambda s=self.reverb_room_slider: s.setToolTip(
            tr("tip_reverb_room")))
        self.reverb_room_slider.valueChanged.connect(
            lambda v: self.reverb_room_val.setText(f"{v / 100.0:.2f}×"))
        fr.addWidget(self.reverb_room_slider, 1)
        self.reverb_room_val = QLabel("1.00×")
        self.reverb_room_val.setMinimumWidth(48)
        fr.addWidget(self.reverb_room_val)
        fr.addWidget(self._trlabel("lbl_reverb_duck"))
        self.reverb_duck_spin = QDoubleSpinBox()
        self.reverb_duck_spin.setRange(0.0, 1.0)
        self.reverb_duck_spin.setSingleStep(0.05)
        self.reverb_duck_spin.setValue(0.0)
        fr.addWidget(self.reverb_duck_spin)
        rh = self._trlabel("hint_reverb")
        rh.setStyleSheet("color:gray; font-size:10px;")
        rh.setWordWrap(True)
        self.reverb_box = self._fx_box([f2, fr], rh)
        acc.addWidget(self.reverb_box)

        fd = QHBoxLayout()
        fd.addWidget(self._trlabel("lbl_detune_cents"))
        self.detune_cents_spin = QDoubleSpinBox()
        self.detune_cents_spin.setRange(0.0, 200.0)
        self.detune_cents_spin.setValue(18.0)
        fd.addWidget(self.detune_cents_spin)
        fd.addWidget(QLabel("LFO(Hz):"))
        self.detune_rate_spin = QDoubleSpinBox()
        self.detune_rate_spin.setRange(0.01, 10.0)
        self.detune_rate_spin.setDecimals(2)
        self.detune_rate_spin.setSingleStep(0.05)
        self.detune_rate_spin.setValue(0.15)
        fd.addWidget(self.detune_rate_spin)
        fd.addStretch()
        self.detune_box = self._fx_box([fd])
        acc.addWidget(self.detune_box)

        fl = QHBoxLayout()
        fl.addWidget(self._trlabel("lbl_lpf_range"))
        self.lpf_hi_spin = QSpinBox()
        self.lpf_hi_spin.setRange(100, 22000)
        self.lpf_hi_spin.setValue(18000)
        fl.addWidget(self.lpf_hi_spin)
        self.lpf_lo_spin = QSpinBox()
        self.lpf_lo_spin.setRange(20, 22000)
        self.lpf_lo_spin.setValue(600)
        fl.addWidget(self.lpf_lo_spin)
        fl.addStretch()
        self.lpf_box = self._fx_box([fl])
        acc.addWidget(self.lpf_box)

        fw = QHBoxLayout()
        fw.addWidget(self._trlabel("lbl_width_range"))
        self.width_lo_spin = QDoubleSpinBox()
        self.width_lo_spin.setRange(0.0, 4.0)
        self.width_lo_spin.setSingleStep(0.1)
        self.width_lo_spin.setValue(1.0)
        fw.addWidget(self.width_lo_spin)
        self.width_hi_spin = QDoubleSpinBox()
        self.width_hi_spin.setRange(0.0, 4.0)
        self.width_hi_spin.setSingleStep(0.1)
        self.width_hi_spin.setValue(1.8)
        fw.addWidget(self.width_hi_spin)
        fw.addStretch()
        self.width_box = self._fx_box([fw])
        acc.addWidget(self.width_box)

        # 各 FX の詳細は、そのチェックが入っているときだけ出す。
        # 常時出していると音声パネルが縦に長くなりすぎるため。
        for chk, box in ((self.fx_reverb_chk, self.reverb_box),
                         (self.fx_detune_chk, self.detune_box),
                         (self.fx_lpf_chk, self.lpf_box),
                         (self.fx_width_chk, self.width_box)):
            box.setVisible(chk.isChecked())
            chk.toggled.connect(box.setVisible)
            chk.toggled.connect(lambda *_: self._sync_settings_height())

        self._audio_widgets = [
            self.audio_mode_combo, self.audio_voices_spin, self.audio_gain_spin,
            self.audio_normalize_chk, self.audio_smooth_combo,
            self.audio_harmonics_spin, self.audio_inpan_combo,
            self.audio_grain_spin, self.audio_jump_spin,
            self.fx_reverb_chk, self.fx_lpf_chk, self.fx_width_chk,
            self.fx_detune_chk, self.reverb_wet_spin, self.reverb_time_spin,
            self.reverb_predelay_spin, self.reverb_room_slider,
            self.reverb_duck_spin, self.detune_cents_spin,
            self.detune_rate_spin, self.lpf_hi_spin, self.lpf_lo_spin,
            self.width_lo_spin, self.width_hi_spin,
        ]
        self._on_audio_toggled(False)
        return acc

    def _on_audio_toggled(self, on):
        for w in getattr(self, "_audio_widgets", []):
            w.setEnabled(bool(on))

    def audio_kwargs(self):
        """audio_render / audio_video_out へ渡す全パラメータ。"""
        harm = self.audio_harmonics_spin.value()
        return {
            "mode": self.audio_mode_combo.currentData() or "play",
            "smooth": self.audio_smooth_combo.currentData() or "fourier",
            "n_harmonics": None if harm == 0 else harm,
            "inpan_mode": self.audio_inpan_combo.currentData() or "balance",
            "grain_dur": float(self.audio_grain_spin.value()),
            "jump_thresh_sec": float(self.audio_jump_spin.value()),
            "normalize": self.audio_normalize_chk.isChecked(),
            "gain": float(self.audio_gain_spin.value()),
            "depth_reverb": self.fx_reverb_chk.isChecked(),
            "reverb_wet": float(self.reverb_wet_spin.value()),
            "reverb_time": float(self.reverb_time_spin.value()),
            "reverb_predelay": float(self.reverb_predelay_spin.value()),
            "reverb_room_size": self.reverb_room_slider.value() / 100.0,
            "reverb_dry_duck": float(self.reverb_duck_spin.value()),
            "depth_lpf": self.fx_lpf_chk.isChecked(),
            "lpf_range": (float(self.lpf_hi_spin.value()),
                          float(self.lpf_lo_spin.value())),
            "depth_width": self.fx_width_chk.isChecked(),
            "width_range": (float(self.width_lo_spin.value()),
                            float(self.width_hi_spin.value())),
            "depth_detune": self.fx_detune_chk.isChecked(),
            "detune_cents": float(self.detune_cents_spin.value()),
            "detune_rate": float(self.detune_rate_spin.value()),
        }

    # ---- 入力映像 ----
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video file", "", VIDEO_FILE_FILTER)
        if path:
            self._apply_selected_video(path)

    def _apply_selected_video(self, path):
        """選択/ドロップされた映像を入力として設定する。"""
        self.videopath_src = self.videopath = path
        self.video_label.setText(f"Selected: {path}")
        self.log(f"Video selected: {path}")
        self._open_video_preview(path)
        self.init_btn.setEnabled(True)

    # ---- ドラッグ＆ドロップ (Finder から動画を放り込む) ----
    def dragEnterEvent(self, ev):
        if dropped_video_path(ev.mimeData()):
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        if dropped_video_path(ev.mimeData()):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        path = dropped_video_path(ev.mimeData())
        if not path:
            return
        ev.acceptProposedAction()
        self._apply_selected_video(path)

    def _style_video_preview(self, empty):
        if empty:
            self.video_preview.setStyleSheet(
                "QLabel { background:#f4f6f9; color:#6b7a8c; font-size:12px;"
                " border:2px dashed #9fb3c8; border-radius:6px; }")
        else:
            self.video_preview.setStyleSheet(
                "QLabel { background:#111; border:1px solid #555; }")

    # ---- 入力映像のその場再生 ----
    _play_from_tick = False

    def _on_play_toggled(self, on):
        self.play_btn.setText(tr("btn_pause") if on else tr("btn_play"))
        if on:
            self._start_play()
        else:
            self._stop_play()

    def _play_bounds(self):
        """再生する入力フレーム範囲 [i0, i1)。使用範囲があればその中をループ。"""
        n = int(self._vid_info[3])
        used = self.timeline.used_range()
        if used:
            i0 = int(used[0] * max(0, n - 1))
            i1 = max(i0 + 1, int(used[1] * max(0, n - 1)) + 1)
        else:
            i0, i1 = 0, n
        return i0, min(i1, n)

    def _start_play(self):
        if self._vid_info is None:
            self.play_btn.setChecked(False)
            return
        i0, i1 = self._play_bounds()
        cur = int(self.timeline.playhead() * max(0, self._vid_info[3] - 1))
        if not (i0 <= cur < i1 - 1):
            cur = i0
        self._play_i0 = cur
        self._play_t0 = time.time()
        fps = float(self._vid_info[2]) or 30.0
        self._play_timer.setInterval(int(max(16, 1000.0 / min(fps, 30.0))))
        self._play_timer.start()

    def _stop_play(self):
        self._play_timer.stop()
        if self.play_btn.isChecked():
            self.play_btn.blockSignals(True)
            self.play_btn.setChecked(False)
            self.play_btn.blockSignals(False)
        self.play_btn.setText(tr("btn_play"))

    def _on_play_tick(self):
        if self._vid_info is None:
            self._stop_play()
            return
        i0, i1 = self._play_bounds()
        fps = float(self._vid_info[2]) or 30.0
        span = max(1, i1 - i0)
        # 壁時計から現在フレームを決める → 追いつけないぶんは自然に飛ぶ
        idx = self._play_i0 + int((time.time() - self._play_t0) * fps)
        idx = i0 + (idx - i0) % span
        self._read_video_frame(idx)
        self._present_video_frame()
        self._play_from_tick = True
        try:
            self.timeline.set_playhead(idx / max(1, self._vid_info[3] - 1))
        finally:
            self._play_from_tick = False
        self._update_timeline_readout()

    def reveal_video_in_finder(self):
        """入力映像を Finder (macOS) で選択状態にして開く。他 OS はフォルダを開く。"""
        path = self.videopath_src or self.videopath
        if not path or not os.path.exists(path):
            self.log(tr("open_output_gone", p=path or "(不明)"))
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))

    def _open_video_preview(self, path):
        if self._vid_cap is not None:
            try:
                self._vid_cap.release()
            except Exception:
                pass
        self._vid_cap = None
        self._vid_info = None
        self._vid_pos = -1
        self._vid_frame_cache = None
        self._stop_play()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.video_preview.setPixmap(QPixmap())
            self.video_preview.setText(tr("drop_placeholder"))
            self._style_video_preview(empty=True)
            for wgt in (self.timeline_label, self.timeline, self.play_btn,
                        self.reveal_btn, self.timeline_readout, self.timeline_hint):
                wgt.setVisible(False)
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vid_cap = cap
        self._vid_info = (w, h, fps, n, n / max(1e-6, fps))
        self.video_dim_label.setText(
            tr("vid_info", w=w, h=h, n=n, fps=fps, dur=n / max(1e-6, fps)))
        self._style_video_preview(empty=False)
        for wgt in (self.timeline_label, self.timeline, self.play_btn,
                    self.reveal_btn, self.timeline_readout, self.timeline_hint):
            wgt.setVisible(True)
        self.timeline.set_used_range(None, None)
        self.timeline.set_playhead(0.0)
        self._update_timeline_readout()
        self._read_video_frame(0)
        self._present_video_frame()

    def _read_video_frame(self, idx):
        """1 フレームだけデコードしてキャッシュ (4K でも軽い)。"""
        if self._vid_cap is None or self._vid_info is None:
            return
        n = self._vid_info[3]
        idx = min(max(0, int(idx)), max(0, n - 1))
        if self._vid_frame_cache is not None and self._vid_frame_cache[0] == idx:
            return
        gap = idx - self._vid_pos
        if gap < 0 or gap > 12:
            self._vid_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        else:
            for _ in range(gap):
                self._vid_cap.grab()
        ret, frame = self._vid_cap.read()
        self._vid_pos = idx + 1
        if ret and frame is not None:
            fh, fw = frame.shape[:2]
            sc = min(1024.0 / max(1, fw), 1024.0 / max(1, fh), 1.0)
            if sc < 1.0:
                frame = cv2.resize(frame, (int(fw * sc), int(fh * sc)),
                                   interpolation=cv2.INTER_AREA)
            self._vid_frame_cache = (idx, frame)

    def _present_video_frame(self):
        """キャッシュから回転 + スリットガイドを適用して表示 (再デコードなし)。"""
        if self._vid_frame_cache is None:
            return
        idx, frame = self._vid_frame_cache
        frame = apply_rotation_cv2(frame, self.vrot_combo.currentData())
        tw = max(1, self.video_preview.width() - 2)
        th = max(1, self.video_preview.height() - 2)
        fh, fw = frame.shape[:2]
        sc = min(tw / max(1, fw), th / max(1, fh), 1.0)
        if sc < 1.0:
            frame = cv2.resize(frame, (max(2, int(fw * sc)), max(2, int(fh * sc))),
                               interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        pm = QPixmap.fromImage(qimg)
        self._draw_slit_overlay(pm)
        self.video_preview.setPixmap(pm)
        if self._vid_info:
            self.video_preview.setToolTip(f"frame {idx} / {self._vid_info[3]}")

    def _draw_slit_overlay(self, pm):
        """スリット方向を緑→赤のガイドラインで可視化する。"""
        N = 12
        w, h = pm.width(), pm.height()
        if w < 8 or h < 8:
            return
        vertical = self._current_sd() == 1
        p = QPainter(pm)
        for i in range(N):
            f = i / (N - 1)
            pen = QPen(QColor(int(255 * f), int(255 * (1 - f)), 0, 170))
            pen.setWidth(1)
            p.setPen(pen)
            if vertical:
                x = int(f * (w - 1))
                p.drawLine(x, 0, x, h)
            else:
                y = int(f * (h - 1))
                p.drawLine(0, y, w, y)
        p.end()

    def _on_scrub(self, frac):
        if self._vid_info is None:
            return
        if self._play_timer.isActive() and not self._play_from_tick:
            self._stop_play()          # 手でつまみを動かしたら再生は止める
        self._read_video_frame(frac * max(0, self._vid_info[3] - 1))
        self._present_video_frame()
        self._update_timeline_readout()

    # ---- タイムライン (使用範囲バンド ⇄ zCenterArange) ----
    def _update_timeline_band(self):
        """軌道データが参照している入力時間範囲を緑バンドに反映する。

        z 値はスリット本数に依存しないため、プロキシ計算の結果をそのまま使える。
        """
        if self._vid_info is None:
            return
        if not self._has_data():
            self.timeline.set_used_range(None, None)
            self._update_timeline_readout()
            return
        n = max(1, int(self.dm.count))
        z = self.dm.data[:, :, 1]
        # 入力範囲外 (負 / count 超過) もそのまま渡す → バー端に警告表示される
        self.timeline.set_used_range(float(z.min()) / n, float(z.max()) / n)
        self._update_timeline_readout()

    def _update_timeline_readout(self):
        if self._vid_info is None:
            self.timeline_readout.setText("")
            return
        fps = self._vid_info[2]
        n = self._vid_info[3]
        used = self.timeline.used_range()
        if used is None:
            self.timeline_readout.setText(
                f"(軌道データなし)   |   再生位置 "
                f"{self.timeline.playhead() * n / max(1e-6, fps):.2f}s"
                if LANG == "ja" else
                f"(no trajectory)   |   playhead "
                f"{self.timeline.playhead() * n / max(1e-6, fps):.2f}s")
            return
        sf, ef = int(round(used[0] * n)), int(round(used[1] * n))
        txt = tr("timeline_readout",
                 s=sf / max(1e-6, fps), e=ef / max(1e-6, fps),
                 sf=sf, ef=ef, n=n,
                 p=self.timeline.playhead() * n / max(1e-6, fps))
        if sf < 0 or ef > n:
            txt += "  ⚠ 範囲外" if LANG == "ja" else "  ⚠ out of range"
        self.timeline_readout.setText(txt)

    def _on_used_range_slid(self, s_frac, e_frac):
        """緑バンドのスライド → チェーン末尾の zCenterArange を自動調整する。

        zCenterArange(center_time_frame) は「軌道の z min/max の中央」を
        指定フレームへ持っていくメソッドなので、バンドの中央フレームを
        そのまま引数に入れれば操作と1対1で対応する。
        末尾に zCenterArange が無ければ追加し、あれば値を更新する。
        """
        if self.dm is None or self._vid_info is None:
            return
        n = int(self.dm.count)
        center = int(round((s_frac + e_frac) / 2.0 * n))
        center = max(0, min(n, center))
        self._update_timeline_readout()

        last = self.steps[-1] if self.steps else None
        if last is not None and last.method_name == "zCenterArange":
            last.set_param("center_time_frame", center)
            self.log(tr("zcenter_updated", f=center))
            self._push_history()
            self._request_rebuild()
        else:
            self.add_step("data", "zCenterArange",
                          {"center_time_frame": center})
            self.log(tr("zcenter_added", f=center))

    def _current_sd(self):
        if self.dm is not None:
            return int(getattr(self.dm, "scan_direction", 1))
        return 1 if self.slit_toggle.isChecked() else 0

    def _on_slit_changed(self, *_):
        self.slit_label.setText(
            tr("slit_v") if self.slit_toggle.isChecked() else tr("slit_h"))
        self._present_video_frame()
        if self.dm is not None:
            self.info_label.setText(tr("vrot_reinit"))
            self.init_btn.setEnabled(True)

    def _on_rotation_changed(self, *_):
        self._present_video_frame()
        if self.dm is not None:
            self.info_label.setText(tr("vrot_reinit"))
            self.init_btn.setEnabled(True)

    def _on_outfps_changed(self, *_):
        if self.dm is not None:
            self.dm.outfps = self.outfps_combo.currentData()
            self.log(f"outfps = {self.dm.outfps}")
            self._refresh_plot()

    # ---- 初期化 ----
    def initialize_dm(self):
        src = self.videopath_src
        if not src:
            QMessageBox.warning(self, "Error", tr("need_init"))
            return
        rot = self.vrot_combo.currentData()
        vf = VIDEO_ROTATION_VF.get(rot)
        if not vf:
            self.videopath = src
            self._init_now()
            return
        out = rotated_video_path(src, rot)
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            self.videopath = out
            self._init_now()
            return
        if shutil.which("ffmpeg") is None:
            QMessageBox.critical(self, "Error", "ffmpeg not found")
            return
        self.info_label.setText(tr("vrot_working"))
        self.init_btn.setEnabled(False)
        self._vrot_worker = VideoRotateWorker(
            src, out, vf, rot_angle=VIDEO_ROTATION_ANGLE.get(rot))
        self._vrot_worker.log_signal.connect(self.log)
        self._vrot_worker.done_signal.connect(self._on_rotated)
        self._vrot_worker.start()

    def _on_rotated(self, ok, out):
        self.init_btn.setEnabled(True)
        if not ok:
            self.info_label.setText(tr("vrot_failed"))
            QMessageBox.critical(self, "Error", tr("vrot_failed"))
            return
        self.videopath = out
        self._init_now()

    def _init_now(self):
        sd = bool(self.slit_toggle.isChecked())
        self.log("Initializing drawManeuver...")
        try:
            self.dm = drawManeuver(videopath=self.videopath, sd=sd)
            self.dm.outfps = self.outfps_combo.currentData()
            self.dm.auto_visualize_out = False   # ステップ実行のたびの PNG 出力を抑止
            info = (f"Video info: {self.dm.width}x{self.dm.height}, "
                    f"Frames: {self.dm.count}, FPS: {self.dm.recfps:.2f}, "
                    f"scan_nums: {self.dm.scan_nums}")
            self.info_label.setText(info)
            self.log(info)
            self._work_dir = os.getcwd()   # 実行用 .py の書き出し先
            self.workdir_btn.setToolTip(self._work_dir)
            self.workdir_btn.setVisible(True)
            self.workdir_browser.set_root(self._work_dir)
            self.workdir_files_label.setVisible(True)
            self.workdir_browser.setVisible(True)
            self.log(f"作業ディレクトリ: {self._work_dir}")
            if self.rt_preview is not None:
                self.rt_preview.set_video(self.videopath)
                self.rt_preview.set_params(sd=int(self.dm.scan_direction),
                                           rec_fps=float(self.dm.recfps),
                                           out_fps=int(self.dm.outfps))
            # 新しい映像を読み込んだら、前の映像のアスペクト比に合わせて
            # 広げていた分をいったん既定比率へ戻す (_on_preview_aspect 参照)
            self._reset_right_split()
            self.runner.invalidate()
            self._last_render_key = None
            self._last_video_path = ""
            self._last_full_data = None
            self.rebuild_pipeline()
            self._update_timeline_band()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.log("[ERROR] " + str(e))
        self._update_gates()

    # ---- ステップ管理 ----
    def _refresh_method_combo(self, *_):
        cat = self.cat_combo.currentData()
        self.method_combo.clear()
        for name, sig in METHOD_REGISTRY.get(cat, []):
            params = [p for p in sig.parameters if p != "self"]
            self.method_combo.addItem(category_icon(cat), name, name)
            self.method_combo.setItemData(
                self.method_combo.count() - 1,
                f"{name}({', '.join(params)})", Qt.ToolTipRole)

    def open_method_help(self):
        """メソッド一覧 + ヘルプの別ウィンドウ (非モーダル・1 枚だけ)。"""
        if self._help_window is None:
            self._help_window = MethodHelpWindow(self)
            self._help_window.add_requested.connect(self._add_from_help)
        self._help_window.select(self.cat_combo.currentData(),
                                 self.method_combo.currentData())
        self._help_window.show()
        self._help_window.raise_()
        self._help_window.activateWindow()

    def _add_from_help(self, category, name):
        self.add_step(category, name)
        # メインの選択も追従させる (続けて同じ系統を足しやすいように)
        i = self.cat_combo.findData(category)
        if i >= 0:
            self.cat_combo.setCurrentIndex(i)
        j = self.method_combo.findData(name)
        if j >= 0:
            self.method_combo.setCurrentIndex(j)

    def _add_step_from_combo(self):
        name = self.method_combo.currentData()
        if not name:
            return
        self.add_step(self.cat_combo.currentData(), name)

    def add_step(self, category, method_name, values=None, enabled=True):
        w = StepWidget(category, method_name, values, enabled)
        w.changed.connect(self._schedule_rebuild)
        w.move_requested.connect(self._move_step)
        w.remove_requested.connect(self._remove_step)
        w.duplicate_requested.connect(self._duplicate_step)
        # 先頭専用の拡張系 (新しいスキャン幅を定義する) は自動で先頭へ入れる
        if method_name in EXPAND_FIRST:
            self.steps.insert(0, w)
            self.steps_box.insertWidget(1, w)      # index 0 は empty_label
            pos = 1
        else:
            self.steps.append(w)
            self.steps_box.insertWidget(len(self.steps), w)
            pos = len(self.steps)
        self._renumber()
        self.log(f"+ step {pos}: {method_name}")
        self._schedule_rebuild()
        return w

    def _move_step(self, w, delta):
        i = self.steps.index(w)
        j = i + delta
        if not (0 <= j < len(self.steps)):
            return
        self.steps.insert(j, self.steps.pop(i))
        self.steps_box.removeWidget(w)
        self.steps_box.insertWidget(j + 1, w)
        self._renumber()
        self._schedule_rebuild()

    def _remove_step(self, w):
        if w not in self.steps:
            return
        self.steps.remove(w)
        self.steps_box.removeWidget(w)
        w.setParent(None)
        w.deleteLater()
        self._renumber()
        self._schedule_rebuild()

    def _duplicate_step(self, w):
        s = w.spec()
        i = self.steps.index(w)
        nw = StepWidget(s["category"], s["method"], s["kwargs"], s["enabled"])
        nw.changed.connect(self._schedule_rebuild)
        nw.move_requested.connect(self._move_step)
        nw.remove_requested.connect(self._remove_step)
        nw.duplicate_requested.connect(self._duplicate_step)
        self.steps.insert(i + 1, nw)
        self.steps_box.insertWidget(i + 2, nw)
        self._renumber()
        self._schedule_rebuild()

    def _renumber(self):
        for i, w in enumerate(self.steps):
            w.set_index(i)
        self.empty_label.setVisible(not self.steps)

    # ---- 履歴 (Undo/Redo) ----
    def _chain_state(self):
        return [w.spec() for w in self.steps]

    def _push_history(self):
        if self._hist_lock:
            return
        state = json.dumps(self._chain_state(), sort_keys=True, default=str)
        if 0 <= self._hist_pos < len(self._history) \
                and self._history[self._hist_pos] == state:
            return
        self._history = self._history[:self._hist_pos + 1]
        self._history.append(state)
        if len(self._history) > 100:
            self._history.pop(0)
        self._hist_pos = len(self._history) - 1
        self._update_history_btns()

    def _update_history_btns(self):
        self.undo_btn.setEnabled(self._hist_pos > 0)
        self.redo_btn.setEnabled(self._hist_pos < len(self._history) - 1)

    def _restore_state(self, state_json):
        specs = json.loads(state_json)
        self._hist_lock = True
        try:
            for w in list(self.steps):
                self.steps.remove(w)
                self.steps_box.removeWidget(w)
                w.setParent(None)
                w.deleteLater()
            for s in specs:
                w = StepWidget(s["category"], s["method"],
                               s.get("kwargs"), s.get("enabled", True))
                w.changed.connect(self._schedule_rebuild)
                w.move_requested.connect(self._move_step)
                w.remove_requested.connect(self._remove_step)
                w.duplicate_requested.connect(self._duplicate_step)
                self.steps.append(w)
                self.steps_box.insertWidget(len(self.steps), w)
            self._renumber()
        finally:
            self._hist_lock = False
        self._request_rebuild()

    def undo(self):
        if self._hist_pos > 0:
            self._hist_pos -= 1
            self._restore_state(self._history[self._hist_pos])
            self._update_history_btns()
            self.log("undo")

    def redo(self):
        if self._hist_pos < len(self._history) - 1:
            self._hist_pos += 1
            self._restore_state(self._history[self._hist_pos])
            self._update_history_btns()
            self.log("redo")

    # ---- パイプライン ----
    def _on_auto_toggled(self, on):
        self.update_btn.setEnabled(not on or self._dirty)
        if on and self._dirty:
            self._request_rebuild()

    def _set_dirty(self, dirty):
        self._dirty = dirty
        self.update_btn.setEnabled(dirty or not self.auto_update_chk.isChecked())
        self.update_btn.setText(
            tr("btn_update_dirty") if dirty else tr("btn_update_now"))

    def _manual_update(self):
        self._rebuild_timer.stop()
        self.rebuild_pipeline()

    def _request_rebuild(self):
        if self.dm is None or self._rendering:
            return
        if self.auto_update_chk.isChecked():
            self._rebuild_timer.start()
        else:
            self._set_dirty(True)

    def _schedule_rebuild(self):
        self._push_history()
        self._request_rebuild()

    def enabled_specs(self):
        return [w.spec() for w in self.steps if w.is_enabled_step()]

    def _warn_expand_position(self):
        """拡張系が置ける位置に無いステップへ注意書きを出す。

        先頭専用 (新しいスキャン幅を定義する) は先頭以外だと必ず失敗し、
        末尾専用 (既存データを広げる) は末尾以外だと後続が必ず失敗する。
        """
        enabled = [w for w in self.steps if w.is_enabled_step()]
        for i, w in enumerate(enabled):
            if w.method_name in EXPAND_FIRST and i != 0:
                w.show_error(tr("expand_must_be_first"))
            elif w.method_name in EXPAND_LAST and i != len(enabled) - 1:
                w.show_error(tr("expand_must_be_last"))

    def rebuild_pipeline(self):
        """全ステップを再実行して dm.data を作り直し、プロットを更新する。"""
        if self.dm is None:
            return
        for w in self.steps:
            w.show_error("")
        self._warn_expand_position()
        specs = self.enabled_specs()
        if not specs:
            self.dm.data = []
            self.runner.invalidate()
            self.plot_label.setPixmap(QPixmap())
            self.plot_label.setText(tr("plot_waiting"))
            self.full2d_btn.setVisible(True)
            self.full3d_btn.setVisible(False)
            self._set_data_info("")
            self._set_dirty(False)
            self.zcheck_btn.setVisible(False)
            self._update_timeline_band()
            self._update_gates()
            return
        self.plot_label.setText(tr("plot_building"))
        QApplication.processEvents()
        t0 = time.time()
        proxy = int(self.preview_scan_spin.value())
        ok, err_i, msg = self.runner.run(self.dm, specs, scan_override=proxy)
        if not ok:
            # 有効ステップの index を実ウィジェットへ写す
            enabled_widgets = [w for w in self.steps if w.is_enabled_step()]
            if 0 <= err_i < len(enabled_widgets):
                enabled_widgets[err_i].show_error(msg)
            self._warn_expand_position()
            self.log(f"[ERROR] step {err_i + 1}: {msg}")
            self.plot_label.setText(tr("plot_error", e=msg))
            self._set_data_info("")
            self._update_gates()
            return
        self.log(f"[chain] {len(specs)} steps → data "
                 f"{getattr(self.dm.data, 'shape', None)}  "
                 f"({time.time() - t0:.2f}s, preview {proxy} slits)")
        self._set_dirty(False)
        self._refresh_plot()
        self._refresh_zcheck_warning()
        self._update_timeline_band()
        self._push_to_rt_preview()
        self._update_gates()

    def _refresh_plot(self):
        data = getattr(self.dm, "data", None) if self.dm else None
        if data is None or len(data) == 0:
            return
        # プロキシの空間座標 (0..preview_scan-1) を本番ピクセルへ換算して表示
        proxy = max(2, int(self.preview_scan_spin.value()))
        real = int(self.dm.scan_nums)
        disp = data
        factor = 1.0
        if data.shape[1] <= proxy and real > proxy:
            factor = (real - 1) / max(1, proxy - 1)
            disp = data.copy()
            disp[:, :, 0] *= factor
        # 1 フレームしか無いデータ (addSlicePlane など) では 2D プロットが
        # 意味を持たない。代わりに 3D プロットを 1 枚だけ描いて見せる。
        single = int(data.shape[0]) <= 1
        self.full2d_btn.setVisible(not single)
        self.full3d_btn.setVisible(not single)
        if single != self._was_single:
            self._was_single = single
            # 1 フレームだけのデータは動画にならないので連番画像を既定にする。
            # 複数フレームへ戻ったときは、こちらが勝手に変えた場合にかぎり
            # 動画へ戻す (ユーザーが自分で選んだ連番設定は尊重する)。
            kind = self.out_kind_combo.currentData()
            if single and kind != "still":
                self._auto_still = True
                self.out_kind_combo.setCurrentIndex(
                    self.out_kind_combo.findData("still"))
                self.log(tr("still_default"))
            elif not single and self._auto_still and kind == "still":
                self._auto_still = False
                self.out_kind_combo.setCurrentIndex(
                    self.out_kind_combo.findData("video"))
                self.log(tr("video_restored"))
        self._sync_still_split()
        if single:
            pm = self._render_3d_still()
            self._plot_full = ("3d", self._last_3d_png)
            if pm is not None:
                self.plot_label.setPixmap(pm)
            else:
                self.plot_label.setPixmap(QPixmap())
                self.plot_label.setText(tr("plot3d_fail"))
            self._finish_data_info(data, disp, factor, real,
                                   note=tr("plot_single_note"))
            return
        self._plot_full = ("2d", disp)
        pm = render_maneuver_plot(
            disp, float(self.dm.outfps), float(self.dm.recfps),
            max(320, self.plot_label.width()), max(240, self.plot_label.height()))
        if pm is not None:
            self.plot_label.setPixmap(pm)
        self._finish_data_info(data, disp, factor, real)

    def _finish_data_info(self, data, disp, factor, real, note=""):
        # 表示するのは「書き出される data の形状」。プレビューは間引いた
        # 本数で計算しているので、そのままだと slits が実際と食い違う。
        out_slits = real if factor != 1.0 else int(data.shape[1])
        # 出力画像の実ピクセル寸法: 積み重ね軸が out_slits、もう一方は映像そのまま
        if int(self.dm.scan_direction) % 2 == 1:
            out_w, out_h = out_slits, int(self.dm.height)
        else:
            out_w, out_h = int(self.dm.width), out_slits
        big = tr("lbl_out_size", w=out_w, h=out_h, f=int(data.shape[0]))
        detail = tr(
            "lbl_data_info", f=int(data.shape[0]), s=out_slits,
            zmin=float(data[:, :, 1].min()), zmax=float(data[:, :, 1].max()),
            smin=float(disp[:, :, 0].min()), smax=float(disp[:, :, 0].max()))
        if factor != 1.0:
            detail += tr("proxy_note", p=int(data.shape[1]), r=real)
        lines = [f'<span style="font-size:17px; font-weight:bold;">{html.escape(big)}</span>',
                 f'<span style="font-size:11px; font-weight:normal;">{html.escape(detail)}</span>']
        if note:
            lines.append(f'<span style="font-size:11px; font-weight:normal;">{html.escape(note)}</span>')
        self._set_data_info("<br>".join(lines))

    def _render_3d_still(self):
        """1 フレームぶんの 3D プロットを描いて QPixmap で返す。"""
        self.plot_label.setText(tr("plot3d_still"))
        QApplication.processEvents()
        since = time.time() - 1
        try:
            self.dm.maneuver_3dplot(out_framenums=1, out_fps=1, dpi=90)
        except Exception as e:
            self.log("[ERROR] maneuver_3dplot: " + str(e))
            return None
        finally:
            plt.close("all")
        out = _newest_plot_output(self._work_dir, since, exts=(".png",))
        if not out:
            return None
        self._last_3d_png = out
        pm = QPixmap(out)
        if pm.isNull():
            return None
        return pm.scaled(max(320, self.plot_label.width()),
                         max(240, self.plot_label.height()),
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def open_plot_fullsize(self):
        """プロットを原寸で別ウィンドウに開く (ダブルクリック)。"""
        kind, src = self._plot_full
        if kind == "3d":
            if not src or not os.path.exists(src):
                return
            pm = QPixmap(src)
            title = "3D plot"
        elif kind == "2d":
            scr = QApplication.primaryScreen()
            avail = scr.availableGeometry() if scr is not None else None
            w_px = int(avail.width() * 0.85) if avail else 1400
            h_px = int(avail.height() * 0.8) if avail else 900
            pm = render_maneuver_plot(src, float(self.dm.outfps),
                                      float(self.dm.recfps), w_px, h_px)
            title = "2D plot"
        else:
            return
        if pm is None or pm.isNull():
            return
        ImageWindow(self, title, pm).exec_()

    def run_full_3dplot(self):
        """3D プロットのアニメーションを書き出す (別スレッド)。"""
        if not self._has_data() or self._plot3d_worker is not None:
            return
        self._set_busy(True)
        self.log(tr("plot3d_running"))
        self._plot3d_worker = Plot3DWorker(
            self.dm, self._work_dir,
            out_framenums=int(min(120, max(24, self.dm.data.shape[0]))),
            out_fps=int(self.dm.outfps or 25), dpi=150)
        self._plot3d_worker.log_signal.connect(self.log)
        self._plot3d_worker.done_signal.connect(self._on_plot3d_done)
        self._plot3d_worker.start()

    def _on_plot3d_done(self, ok, path):
        self._plot3d_worker = None
        self._set_busy(False)
        if ok:
            self.log(tr("plot3d_done", p=path))
            self.workdir_browser.refresh()
            self._set_output_target(path)
        else:
            self.log(tr("plot3d_fail"))

    def _set_data_info(self, text):
        """出力データ形状の表示を更新する (空なら枠ごと隠す)。"""
        self.data_info.setText(text)
        self.data_info.setVisible(bool(text))

    def _push_to_rt_preview(self):
        if self.rt_preview is None or self.dm is None:
            return
        data = getattr(self.dm, "data", None)
        if data is None or len(data) == 0:
            return
        # プレビューデータはプロキシ解像度: 空間座標は 0..(preview_scan-1)
        scan_full = int(self.preview_scan_spin.value())
        self.rt_preview.set_maneuver(data, scan_full)
        # 出力の積み重ね軸の本番ピクセル数 (data_info の表示と同じ求め方)。
        # プロキシ本数のままだと、addSlicePlane(aspect_mode='fix') のように
        # 出力の幅が入力と異なるチェーンで、入力映像の形のまま描いてしまう。
        proxy = max(2, scan_full)
        real = int(self.dm.scan_nums)
        out_stack = real if (data.shape[1] <= proxy and real > proxy) \
            else int(data.shape[1])
        self.rt_preview.set_params(time_size=int(data.shape[0]),
                                   out_fps=int(self.dm.outfps),
                                   out_stack=out_stack)
        if self.rt_preview._backend is not None:
            self.rt_preview.refresh_maps()
            # maps は差し替わるが常駐している映像の時間範囲は古いまま。
            # チェーンが前回の構築時から変わっていれば「更新」を促す。
            if self._chain_key() != self._rt_built_key:
                self.rt_preview.mark_stale()

    def _chain_key(self):
        return json.dumps(self.enabled_specs(), sort_keys=True, default=str)

    def _on_rt_rebuilt(self):
        """GPU プレビューを構築したら、そのときのチェーンを覚えておく。"""
        self._rt_built_key = self._chain_key()
        self._stop_play()               # GPU 再生と入力再生を同時には回さない

    # ---- アクション ----
    def _is_busy(self):
        """レンダリング/3D プロット/プレビュー保存のいずれかが走っているか。"""
        return bool(self._rendering or self._plot3d_worker is not None
                    or self._saving_preview)

    def _set_busy(self, busy):
        """重い処理どうしがぶつからないよう、実行系のボタンをまとめて塞ぐ。

        レンダリング中にプレビュー保存を始めると、進捗を出すための
        processEvents でレンダリング完了処理が入れ子で走り、
        GPU プレビューのデータが差し替わってしまう。
        """
        self.render_progress.setVisible(bool(busy))
        for b in (self.render_btn, self.full2d_btn, self.full3d_btn,
                  self.save_preview_btn, self.export_btn):
            b.setEnabled(not busy)
        if not busy:
            self._update_gates()

    def _has_data(self):
        return (self.dm is not None
                and isinstance(getattr(self.dm, "data", None), np.ndarray)
                and len(self.dm.data) > 0)

    def _update_gates(self):
        ok = self._has_data() and not self._is_busy()
        for b in (self.render_btn, self.export_btn, self.full2d_btn,
                  self.full3d_btn):
            b.setEnabled(ok)
        self.save_preview_btn.setEnabled(not self._is_busy())
        self.add_step_btn.setEnabled(self.dm is not None)

    def _refresh_zcheck_warning(self):
        """time (z) が負 or 総フレーム超過なら警告ボタンを出す。

        z 値はスリット数に依存しないため、プロキシデータでの判定が
        本番データと一致する。
        """
        show = False
        if self._has_data():
            z = self.dm.data[:, :, 1]
            show = float(z.min()) < 0 or float(z.max()) > float(self.dm.count)
        # 既に末尾が zPointCheck なら出さない
        if show and self.steps:
            last_enabled = [w for w in self.steps if w.is_enabled_step()]
            if last_enabled and last_enabled[-1].method_name == "zPointCheck":
                show = False
        self.zcheck_btn.setVisible(show)

    def append_zcheck_step(self):
        """警告ボタン: チェーンの現状の最後に zPointCheck ステップを追加する。"""
        self.add_step("data", "zPointCheck", {})
        self.zcheck_btn.setVisible(False)

    def run_full_2dplot(self):
        if not self._has_data():
            return
        try:
            self.dm.maneuver_2dplot()
            self.log("maneuver_2dplot: PNG を出力しました → " + os.getcwd())
        except Exception as e:
            self.log("[ERROR] maneuver_2dplot: " + str(e))
        finally:
            plt.close("all")

    def _render_key(self, specs):
        """映像レンダリング結果を左右する条件の指紋 (音声設定は含めない)。

        音声だけを変えた再書き出しでは映像を作り直さないため、
        出力形式や分割数など映像に効く設定はすべてここに含める。
        """
        return json.dumps({
            "video": self.videopath,
            "sd": bool(self.slit_toggle.isChecked()),
            "outfps": int(self.outfps_combo.currentData()),
            "out_type": self.selected_out_type(),
            "imgtype": self.img_format_combo.currentData(),
            "separate": int(self.separate_spin.value()),
            "specs": specs,
        }, sort_keys=True, default=str)

    def start_render(self):
        if not self._has_data():
            QMessageBox.warning(self, "Error", tr("need_steps"))
            return
        specs = self.enabled_specs()
        key = self._render_key(specs)
        # チェーンが前回レンダリングと同一 + 映像が残っている + 音声ON
        # → 映像の再レンダリングを省略して音声のみ書き出す
        out_type = self.selected_out_type()
        # 連番画像出力では音声を後付けできないので通常経路のみ
        audio_only = (self.audio_chk.isChecked() and out_type != 0
                      and key == self._last_render_key
                      and bool(self._last_video_path)
                      and os.path.exists(self._last_video_path))
        self._rendering = True
        self._rebuild_timer.stop()
        self._stop_play()
        self._set_busy(True)
        self.open_output_btn.setVisible(False)
        self.render_progress.setVisible(True)
        self.log(tr("rendering"))
        sep = int(self.separate_spin.value())
        self._render_worker = RenderWorker(
            self.dm, specs,
            out_type=out_type,
            separate_num=sep if sep > 0 else None,
            imgtype=(self.img_format_combo.currentData()
                     if out_type == 0 else None),
            audio_out=self.audio_chk.isChecked() and out_type != 0,
            audio_kwargs=self.audio_kwargs(),
            audio_voices=self.audio_voices_spin.value(),
            audio_only=audio_only,
            video_path=self._last_video_path,
            full_data=self._last_full_data,
            still_split_minutes=self.still_split_minutes(),
            scan_step=int(self.scan_step_spin.value()),
        )
        self._pending_render_key = key
        self._render_worker.log_signal.connect(self.log)
        self._render_worker.done_signal.connect(self._on_render_done)
        self._render_worker.start()

    def _on_render_done(self, ok, path):
        self._rendering = False
        self._set_busy(False)
        if ok:
            # 音声のみ再書き出しに備えて、映像のみのパスと本番解像度の
            # data を記録する (プロキシ再構築で dm.data が上書きされる前に)
            w = self._render_worker
            if w is not None and w.video_only_path:
                self._last_video_path = w.video_only_path
                self._last_render_key = getattr(self, "_pending_render_key", None)
                d = getattr(self.dm, "data", None)
                if isinstance(d, np.ndarray) and len(d) > 0:
                    self._last_full_data = d.copy()
            self.log(tr("render_done", p=path or "(不明)"))
            self.workdir_browser.refresh()
            # 連番画像出力は単一ファイルにならないので書き出し先フォルダを開く
            img_dir = os.path.join(self._work_dir, "img")
            self._set_output_target(
                path or (img_dir if os.path.isdir(img_dir) else self._work_dir))
        else:
            self.log(tr("render_failed"))
        # プレビューキャッシュを破棄し、プロキシデータへ戻す
        self.runner.invalidate()
        self._request_rebuild()
        self._update_gates()

    def open_work_dir(self):
        """初期化で作られた作業ディレクトリを Finder (既定のファイラ) で開く。"""
        d = self._work_dir
        if not d or not os.path.isdir(d):
            self.log(tr("open_output_gone", p=d or "(不明)"))
            return
        self.log(tr("open_output", p=d))
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(d)):
            self.log(tr("open_output_gone", p=d))

    def save_preview_video(self):
        """GPU プレビューの内容を作業ディレクトリへ簡易書き出しする。"""
        if self._is_busy():
            self.log(tr("busy_now"))
            QMessageBox.information(self, "Info", tr("busy_now"))
            return
        rt = self.rt_preview
        if rt is None or not rt.is_preview_ready():
            QMessageBox.information(self, "Info", tr("save_preview_need"))
            return
        path = self._unique_out_path("preview")
        self._saving_preview = True
        self._set_busy(True)
        try:
            out = rt.export_preview_video(
                path, progress_cb=lambda i, n: self._preview_progress(i, n))
        except Exception as e:
            out = ""
            self.log("[ERROR] " + str(e))
        finally:
            self._saving_preview = False
            self._set_busy(False)
        if out:
            self.log(tr("save_preview_done", p=out))
            self.workdir_browser.refresh()
            self._set_output_target(out)
        else:
            self.log(tr("save_preview_fail"))

    def _preview_progress(self, i, n):
        self.log_status(tr("save_preview_running", i=i, n=n))
        QApplication.processEvents()

    def log_status(self, text):
        """進捗表示。連続して呼ばれた分は最終行を書き換えて行を増やさない。"""
        if self._status_active:
            cur = self.log_window.textCursor()
            cur.movePosition(QTextCursor.End)
            cur.select(QTextCursor.LineUnderCursor)
            cur.insertText(text)
            self.log_window.setTextCursor(cur)
            self.log_window.ensureCursorVisible()
        else:
            self.log(text)
            self._status_active = True

    def _unique_out_path(self, kind):
        """作業ディレクトリ内で衝突しないパスを返す。"""
        stem = os.path.splitext(os.path.basename(self.videopath or ""))[0]
        base = f"{kind}_{stem}" if stem else kind
        path = os.path.join(self._work_dir, base + ".mp4")
        n = 2
        while os.path.exists(path):
            path = os.path.join(self._work_dir, f"{base}_{n}.mp4")
            n += 1
        return path

    def _set_output_target(self, path):
        """「開く」ボタンの対象を設定する (無ければボタンを隠す)。"""
        path = os.path.abspath(path) if path else ""
        self._output_path = path if path and os.path.exists(path) else ""
        self._output_is_dir = bool(self._output_path
                                   and os.path.isdir(self._output_path))
        self.open_output_btn.setText(
            tr("btn_open_output_dir") if self._output_is_dir
            else tr("btn_open_output"))
        self.open_output_btn.setToolTip(self._output_path)
        self.open_output_btn.setVisible(bool(self._output_path))

    def open_rendered_output(self):
        """書き出したものを OS の既定アプリで開く。

        動画なら QuickTime 等のプレーヤーで再生が始まり、連番画像なら
        書き出し先フォルダが開く。
        """
        path = self._output_path
        if not path or not os.path.exists(path):
            self.log(tr("open_output_gone", p=path or "(不明)"))
            self.open_output_btn.setVisible(False)
            return
        self.log(tr("open_output", p=path))
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            self.log(tr("open_output_gone", p=path))

    # ---- .py 書き出し ----
    def build_script(self):
        """現在のチェーンを実行する単体 Python スクリプトを生成する。"""
        sd = "True" if self.slit_toggle.isChecked() else "False"
        lines = [
            '"""drawmaneuver_gui が書き出した実行用スクリプト',
            "",
            f"生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            '"""',
            "from imgtrans import drawManeuver",
            "",
            f"VIDEO = {self.videopath!r}",
            "",
            f"dm = drawManeuver(videopath=VIDEO, sd={sd})",
            f"dm.outfps = {int(self.outfps_combo.currentData())}",
            "",
            "# ==== メソッドチェーン ====",
        ]
        for i, w in enumerate(self.steps):
            s = w.spec()
            args = ", ".join(f"{k}={fmt_value(v)}" for k, v in s["kwargs"].items())
            call = f"dm.{s['method']}({args})"
            if s["enabled"]:
                lines.append(f"{call}    # step {i + 1}")
            else:
                lines.append(f"# [無効] {call}    # step {i + 1}")
        out_type = self.selected_out_type()
        sep = int(self.separate_spin.value())
        sep_arg = f", separate_num={sep}" if sep > 0 else ""
        lines += ["", "# ==== 検証 & 出力 ===="]
        lines.append("dm.maneuver_2dplot()")
        if out_type == 0:
            lines.append(f"dm.imgtype = {self.img_format_combo.currentData()!r}")
        lines.append(
            f"dm.new_transprocess(out_type={out_type}{sep_arg}, del_data=False)")
        if self.audio_chk.isChecked() and out_type != 0:
            kw = self.audio_kwargs()
            args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
            lines.append(
                f"dm.audio_video_out(thread_num="
                f"{int(self.audio_voices_spin.value())}, {args})")
        lines.append("")
        return "\n".join(lines)

    def export_script(self):
        if not self.steps:
            QMessageBox.warning(self, "Error", tr("need_steps"))
            return
        # 保存先は聞かず、初期化した作業ディレクトリへそのまま書き出す。
        # 既存の書き出しを黙って上書きしないよう、名前が衝突したら連番を付ける。
        stem = os.path.splitext(os.path.basename(self.videopath or ""))[0]
        base = f"maneuver_{stem}" if stem else "maneuver_script"
        path = os.path.join(self._work_dir, base + ".py")
        n = 2
        while os.path.exists(path):
            path = os.path.join(self._work_dir, f"{base}_{n}.py")
            n += 1
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.build_script())
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.log("[ERROR] " + tr("export_fail", p=path, e=str(e)))
            return
        self.log(tr("export_done", p=os.path.abspath(path)))
        self.workdir_browser.refresh()

    # ---- log ----
    def log(self, text):
        self._status_active = False
        self.log_window.append(str(text))
        self.log_window.ensureCursorVisible()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DrawManeuverGUI()
    win.show()
    sys.exit(app.exec_())
