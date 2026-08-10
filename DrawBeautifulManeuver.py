"""DrawBeautifulManeuver — drawManeuver メソッドチェーン エディタ (2026-08)

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
import ast
import copy
import time
import json
import shutil
import inspect
import subprocess
import traceback
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QComboBox, QTextEdit, QCheckBox, QMessageBox, QSpinBox, QDoubleSpinBox,
    QFrame, QGroupBox, QScrollArea, QSplitter, QProgressBar, QSizePolicy,
    QSlider, QLineEdit, QFileDialog, QToolButton, QDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

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
    "window_title": {"ja": "DrawBeautifulManeuver", "en": "DrawBeautifulManeuver"},
    "grp_setup": {"ja": "入力 (Setup)", "en": "Setup"},
    "lang_label": {"ja": "言語 / Language:", "en": "Language / 言語:"},
    "btn_select_video": {"ja": "動画を選択 / Select Video File",
                          "en": "Select Video File"},
    "no_video": {"ja": "動画が未選択です", "en": "No video file selected"},
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
    "lbl_method": {"ja": "メソッド:", "en": "Method:"},
    "btn_add_step": {"ja": "＋ ステップを追加", "en": "+ Add step"},
    "cat_add": {"ja": "Add 系 (データを作る/継ぎ足す)", "en": "Add (create/append)"},
    "cat_apply": {"ja": "Apply 系 (全体に効果を適用)", "en": "Apply (transform all)"},
    "cat_data": {"ja": "データ操作 (整列/抽出/チェック)", "en": "Data ops"},
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
    "lbl_data_info": {"ja": "data: {f} frames × {s} slits   |   z {zmin:.0f}–{zmax:.0f}"
                            "   |   space {smin:.0f}–{smax:.0f}",
                       "en": "data: {f} frames × {s} slits   |   z {zmin:.0f}–{zmax:.0f}"
                             "   |   space {smin:.0f}–{smax:.0f}"},
    "grp_realtime": {"ja": "リアルタイム軸間変換プレビュー (GPU)",
                      "en": "Realtime axis-transform preview (GPU)"},
    "btn_render": {"ja": "レンダリング開始 / Start Rendering", "en": "Start Rendering"},
    "btn_export": {"ja": "実行用 .py を書き出し", "en": "Export .py"},
    "btn_zcheck": {"ja": "zPointCheck を実行", "en": "Run zPointCheck"},
    "btn_full2d": {"ja": "詳細 2D プロット (PNG)", "en": "Full 2D plot (PNG)"},
    "lbl_log": {"ja": "Log:", "en": "Log:"},
    "need_init": {"ja": "先に動画を選択して「初期化」してください。",
                   "en": "Select a video and press Initialize first."},
    "need_steps": {"ja": "マニューバデータがありません (Add 系のステップが必要です)。",
                    "en": "No maneuver data — add an Add-category step first."},
    "export_done": {"ja": "書き出しました: {p}", "en": "Exported: {p}"},
    "rendering": {"ja": "レンダリング中…", "en": "Rendering…"},
    "render_done": {"ja": "レンダリング完了: {p}", "en": "Rendering done: {p}"},
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
CATEGORY_ORDER = ["add", "apply", "data", "other"]
CATEGORY_LABEL_KEY = {"add": "cat_add", "apply": "cat_apply",
                      "data": "cat_data", "other": "cat_other"}

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


def _parse_readme(path):
    """README を「## `メソッド名`」単位のセクション辞書にする。"""
    key = str(path)
    if key in _README_CACHE:
        return _README_CACHE[key]
    sections = {}
    name = None
    buf = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = line.strip()
            if m.startswith("## `") and m.endswith("`"):
                if name:
                    sections[name] = "\n".join(buf).strip()
                name = m[4:-1]
                buf = []
            elif m.startswith("## ") and name:
                sections[name] = "\n".join(buf).strip()
                name = None
                buf = []
            elif name is not None:
                # 画像参照はダイアログで表示できないので除く
                if not m.startswith("!["):
                    buf.append(line)
        if name:
            sections[name] = "\n".join(buf).strip()
    except Exception:
        pass
    _README_CACHE[key] = sections
    return sections


def readme_doc(method_name):
    """README のメソッド説明 (markdown)。無ければ None。"""
    for path in _readme_paths():
        doc = _parse_readme(path).get(method_name)
        if doc:
            return doc
    return None


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

    CAT_COLOR = {"add": "#2a6fd6", "apply": "#c07000",
                 "data": "#2a8f4f", "other": "#777777"}

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
        """メソッドの説明をダイアログ表示する。

        優先順: imgtrans リポジトリの README (現在言語 → もう一方) の
        「## `メソッド名`」セクション → docstring → (説明なし)。
        """
        fn = getattr(drawManeuver, self.method_name, None)
        try:
            sig = str(inspect.signature(fn)).replace("self, ", "").replace("self", "")
        except Exception:
            sig = "()"
        try:
            srcfile = os.path.basename(inspect.getfile(fn))
            line = inspect.getsourcelines(fn)[1]
            where = f"{srcfile}:{line}"
        except Exception:
            where = ""

        doc_md = readme_doc(self.method_name)
        if doc_md is None:
            doc_md = inspect.getdoc(fn) if fn else None
        if not doc_md:
            doc_md = tr("no_doc")

        dlg = QDialog(self)
        dlg.setWindowTitle(self.method_name)
        dlg.resize(640, 480)
        v = QVBoxLayout(dlg)
        head = QLabel(f"<b>{self.method_name}</b><span style='color:#666;'>"
                      f"{sig}</span>"
                      + (f"<br><span style='color:#999; font-size:10px;'>"
                         f"{where}</span>" if where else ""))
        head.setWordWrap(True)
        head.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(head)
        body = QTextEdit()
        body.setReadOnly(True)
        try:
            body.setMarkdown(doc_md)
        except Exception:
            body.setPlainText(doc_md)
        v.addWidget(body, 1)
        close = QPushButton("OK")
        close.clicked.connect(dlg.accept)
        v.addWidget(close, 0, Qt.AlignRight)
        dlg.exec_()

    def _set_border(self):
        on = self.enable_chk.isChecked() if hasattr(self, "enable_chk") else True
        col = self.CAT_COLOR.get(self.category, "#888")
        self.setStyleSheet(
            f"StepWidget {{ border:1px solid {col if on else '#ccc'};"
            f" border-radius:4px; background: rgba(128,128,128,{18 if on else 6}); }}")

    def set_index(self, i):
        self.index_label.setText(str(i + 1))

    def is_enabled_step(self):
        return self.enable_chk.isChecked()

    def spec(self):
        return {"category": self.category, "method": self.method_name,
                "kwargs": self.editor.values(),
                "enabled": self.enable_chk.isChecked()}

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
                 audio_out=False, audio_mode="play", audio_voices=20,
                 audio_only=False, video_path="", full_data=None):
        super().__init__()
        self.dm = dm
        self.specs = specs
        self.out_type = out_type
        self.separate_num = separate_num
        self.audio_out = audio_out
        self.audio_mode = audio_mode
        self.audio_voices = max(2, int(audio_voices))
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

            # 2) レンダリング
            self.log_signal.emit("=== new_transprocess ===")
            self.dm.new_transprocess(separate_num=self.separate_num,
                                     out_type=self.out_type, del_data=False)
            path = getattr(self.dm, "out_videopath", "") or ""
            if path and not os.path.isabs(path):
                path = os.path.abspath(path)
            self.video_only_path = path if os.path.exists(path) else ""

            # 3) 音声 (プレビュー設定と同じく audio_video_out へ)
            if self.audio_out and path and os.path.exists(path):
                try:
                    self.log_signal.emit(
                        f"=== audio_video_out (mode={self.audio_mode}, "
                        f"voices={self.audio_voices}) ===")
                    final = self.dm.audio_video_out(
                        mode=self.audio_mode, thread_num=self.audio_voices)
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
                f"=== audio_video_out (mode={self.audio_mode}, "
                f"voices={self.audio_voices}) ===")
            final = self.dm.audio_video_out(
                videopath=self.video_path,
                mode=self.audio_mode, thread_num=self.audio_voices)
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
class DrawBeautifulManeuverApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(1560, 980)
        self.setMinimumSize(900, 640)

        self.videopath = None
        self.videopath_src = None
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
        self._reg(lambda: self.video_btn.setText(tr("btn_select_video")))
        self.video_btn.clicked.connect(self.select_video)
        sg.addWidget(self.video_btn)
        self.video_label = QLabel(tr("no_video"))
        self.video_label.setWordWrap(True)
        self.video_label.setStyleSheet("color:gray; font-size:10px;")
        self._i18n.append(lambda: (None if self.videopath_src
                                   else self.video_label.setText(tr("no_video"))))
        sg.addWidget(self.video_label)

        self.video_preview = QLabel()
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.video_preview.setFixedHeight(150)
        self.video_preview.setStyleSheet(
            "QLabel { background:#111; border:1px solid #555; }")
        self.video_preview.setVisible(False)
        sg.addWidget(self.video_preview)
        self.video_scrub = QSlider(Qt.Horizontal)
        self.video_scrub.setRange(0, 1000)
        self.video_scrub.valueChanged.connect(self._on_scrub)
        self.video_scrub.setVisible(False)
        sg.addWidget(self.video_scrub)
        self.video_dim_label = QLabel("")
        self.video_dim_label.setStyleSheet("color:gray; font-size:10px;")
        sg.addWidget(self.video_dim_label)

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
        cg.addWidget(scroll, 1)

        addrow = QHBoxLayout()
        addrow.addWidget(self._trlabel("lbl_category"))
        self.cat_combo = QComboBox()
        for c in CATEGORY_ORDER:
            self.cat_combo.addItem(tr(CATEGORY_LABEL_KEY[c]), c)
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
        cg.addLayout(addrow)
        self._refresh_method_combo()

        # --- 2D プロット + 実行 ---
        plot_group = QGroupBox()
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

        self.plot_label = QLabel(tr("plot_waiting"))
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumSize(360, 260)
        self.plot_label.setStyleSheet(
            "QLabel { background:#ffffff; border:1px solid #555; color:#888; }")
        self._i18n.append(lambda: (None if self.plot_label.pixmap()
                                   else self.plot_label.setText(tr("plot_waiting"))))
        pg.addWidget(self.plot_label, 1)
        self.data_info = QLabel("")
        self.data_info.setStyleSheet("color:#555; font-size:10px;")
        pg.addWidget(self.data_info)

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
        self.export_btn = QPushButton()
        self._reg(lambda: self.export_btn.setText(tr("btn_export")))
        self.export_btn.clicked.connect(self.export_script)
        actions.addWidget(self.export_btn)
        pg.addLayout(actions)

        render_row = QHBoxLayout()
        self.audio_chk = QCheckBox()
        self._reg(lambda: self.audio_chk.setText(tr("chk_audio")))
        render_row.addWidget(self.audio_chk)
        self.audio_mode_combo = QComboBox()
        self.audio_mode_combo.addItem(tr("audio_mode_play"), "play")
        self.audio_mode_combo.addItem(tr("audio_mode_grain"), "grain")
        self._reg(lambda: (
            self.audio_mode_combo.setItemText(0, tr("audio_mode_play")),
            self.audio_mode_combo.setItemText(1, tr("audio_mode_grain"))))
        self.audio_mode_combo.setEnabled(False)
        self.audio_chk.toggled.connect(self.audio_mode_combo.setEnabled)
        render_row.addWidget(self.audio_mode_combo)
        render_row.addWidget(self._trlabel("lbl_audio_voices"))
        self.audio_voices_spin = QSpinBox()
        self.audio_voices_spin.setRange(2, 64)
        self.audio_voices_spin.setValue(20)
        render_row.addWidget(self.audio_voices_spin)
        self.render_btn = QPushButton()
        self._reg(lambda: self.render_btn.setText(tr("btn_render")))
        self.render_btn.clicked.connect(self.start_render)
        render_row.addWidget(self.render_btn, 1)
        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 0)      # 不定 (走行中のみ表示)
        self.render_progress.setVisible(False)
        render_row.addWidget(self.render_progress, 1)
        pg.addLayout(render_row)

        # --- GPU プレビュー ---
        if _HAS_RT_PREVIEW:
            self.rt_group = QGroupBox()
            self._reg(lambda: self.rt_group.setTitle(tr("grp_realtime")))
            rv = QVBoxLayout(self.rt_group)
            rv.setContentsMargins(4, 4, 4, 4)
            self.rt_preview = ManeuverRTPreview(lang=LANG)
            rv.addWidget(self.rt_preview)
        else:
            self.rt_group = None
            self.rt_preview = None

        # --- レイアウト: [入力] [チェーン] [プロット + GPU] ---
        cols = QSplitter(Qt.Horizontal)
        left = QWidget(); ll = QVBoxLayout(left); ll.addWidget(setup)
        cols.addWidget(left)
        cols.addWidget(chain_group)
        right = QWidget(); rl = QVBoxLayout(right)
        rl.addWidget(plot_group, 3)
        if self.rt_group is not None:
            rl.addWidget(self.rt_group, 4)
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

    # ---- 入力映像 ----
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video file", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.videopath_src = self.videopath = path
        self.video_label.setText(f"Selected: {path}")
        self.log(f"Video selected: {path}")
        self._open_video_preview(path)
        self.init_btn.setEnabled(True)

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
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.video_preview.setVisible(False)
            self.video_scrub.setVisible(False)
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vid_cap = cap
        self._vid_info = (w, h, fps, n, n / max(1e-6, fps))
        self.video_dim_label.setText(
            tr("vid_info", w=w, h=h, n=n, fps=fps, dur=n / max(1e-6, fps)))
        self.video_preview.setVisible(True)
        self.video_scrub.setVisible(True)
        self.video_scrub.blockSignals(True)
        self.video_scrub.setValue(0)
        self.video_scrub.blockSignals(False)
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

    def _on_scrub(self, val):
        if self._vid_info is None:
            return
        self._read_video_frame(val / 1000.0 * max(0, self._vid_info[3] - 1))
        self._present_video_frame()

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
            self.log(f"作業ディレクトリ: {os.getcwd()}")
            if self.rt_preview is not None:
                self.rt_preview.set_video(self.videopath)
                self.rt_preview.set_params(sd=int(self.dm.scan_direction),
                                           rec_fps=float(self.dm.recfps),
                                           out_fps=int(self.dm.outfps))
            self.runner.invalidate()
            self._last_render_key = None
            self._last_video_path = ""
            self._last_full_data = None
            self.rebuild_pipeline()
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
            self.method_combo.addItem(name, name)
            self.method_combo.setItemData(
                self.method_combo.count() - 1,
                f"{name}({', '.join(params)})", Qt.ToolTipRole)

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
        self.steps.append(w)
        self.steps_box.insertWidget(len(self.steps), w)
        self._renumber()
        self.log(f"+ step {len(self.steps)}: {method_name}")
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

    def rebuild_pipeline(self):
        """全ステップを再実行して dm.data を作り直し、プロットを更新する。"""
        if self.dm is None:
            return
        for w in self.steps:
            w.show_error("")
        specs = self.enabled_specs()
        if not specs:
            self.dm.data = []
            self.runner.invalidate()
            self.plot_label.setPixmap(QPixmap())
            self.plot_label.setText(tr("plot_waiting"))
            self.data_info.setText("")
            self._set_dirty(False)
            self.zcheck_btn.setVisible(False)
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
            self.log(f"[ERROR] step {err_i + 1}: {msg}")
            self.plot_label.setText(tr("plot_error", e=msg))
            self.data_info.setText("")
            self._update_gates()
            return
        self.log(f"[chain] {len(specs)} steps → data "
                 f"{getattr(self.dm.data, 'shape', None)}  "
                 f"({time.time() - t0:.2f}s, preview {proxy} slits)")
        self._set_dirty(False)
        self._refresh_plot()
        self._refresh_zcheck_warning()
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
        pm = render_maneuver_plot(
            disp, float(self.dm.outfps), float(self.dm.recfps),
            max(320, self.plot_label.width()), max(240, self.plot_label.height()))
        if pm is not None:
            self.plot_label.setPixmap(pm)
        info = tr(
            "lbl_data_info", f=data.shape[0], s=data.shape[1],
            zmin=float(data[:, :, 1].min()), zmax=float(data[:, :, 1].max()),
            smin=float(disp[:, :, 0].min()), smax=float(disp[:, :, 0].max()))
        if factor != 1.0:
            info += tr("proxy_note", p=data.shape[1], r=real)
        self.data_info.setText(info)

    def _push_to_rt_preview(self):
        if self.rt_preview is None or self.dm is None:
            return
        data = getattr(self.dm, "data", None)
        if data is None or len(data) == 0:
            return
        # プレビューデータはプロキシ解像度: 空間座標は 0..(preview_scan-1)
        scan_full = int(self.preview_scan_spin.value())
        self.rt_preview.set_maneuver(data, scan_full)
        self.rt_preview.set_params(time_size=int(data.shape[0]),
                                   out_fps=int(self.dm.outfps))
        if self.rt_preview._backend is not None:
            self.rt_preview.refresh_maps()

    # ---- アクション ----
    def _has_data(self):
        return (self.dm is not None
                and isinstance(getattr(self.dm, "data", None), np.ndarray)
                and len(self.dm.data) > 0)

    def _update_gates(self):
        ok = self._has_data()
        for b in (self.render_btn, self.export_btn, self.full2d_btn):
            b.setEnabled(ok)
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
        """映像レンダリング結果を左右する条件の指紋 (音声設定は含めない)。"""
        return json.dumps({
            "video": self.videopath,
            "sd": bool(self.slit_toggle.isChecked()),
            "outfps": int(self.outfps_combo.currentData()),
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
        audio_only = (self.audio_chk.isChecked()
                      and key == self._last_render_key
                      and bool(self._last_video_path)
                      and os.path.exists(self._last_video_path))
        self._rendering = True
        self._rebuild_timer.stop()
        self.render_btn.setEnabled(False)
        self.render_progress.setVisible(True)
        self.log(tr("rendering"))
        self._render_worker = RenderWorker(
            self.dm, specs,
            audio_out=self.audio_chk.isChecked(),
            audio_mode=self.audio_mode_combo.currentData() or "play",
            audio_voices=self.audio_voices_spin.value(),
            audio_only=audio_only,
            video_path=self._last_video_path,
            full_data=self._last_full_data,
        )
        self._pending_render_key = key
        self._render_worker.log_signal.connect(self.log)
        self._render_worker.done_signal.connect(self._on_render_done)
        self._render_worker.start()

    def _on_render_done(self, ok, path):
        self._rendering = False
        self.render_progress.setVisible(False)
        self.render_btn.setEnabled(True)
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
        else:
            self.log(tr("render_failed"))
        # プレビューキャッシュを破棄し、プロキシデータへ戻す
        self.runner.invalidate()
        self._request_rebuild()
        self._update_gates()

    # ---- .py 書き出し ----
    def build_script(self):
        """現在のチェーンを実行する単体 Python スクリプトを生成する。"""
        sd = "True" if self.slit_toggle.isChecked() else "False"
        lines = [
            '"""DrawBeautifulManeuver が書き出した実行用スクリプト',
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
        lines += [
            "",
            "# ==== 検証 & 出力 ====",
            "dm.zPointCheck()",
            "dm.maneuver_2dplot()",
            "dm.new_transprocess(out_type=1, del_data=False)",
            "",
        ]
        return "\n".join(lines)

    def export_script(self):
        if not self.steps:
            QMessageBox.warning(self, "Error", tr("need_steps"))
            return
        default = os.path.join(os.getcwd(), "maneuver_script.py")
        path, _ = QFileDialog.getSaveFileName(
            self, tr("btn_export"), default, "Python (*.py)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.build_script())
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        self.log(tr("export_done", p=path))

    # ---- log ----
    def log(self, text):
        self.log_window.append(str(text))
        self.log_window.ensureCursorVisible()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DrawBeautifulManeuverApp()
    win.show()
    sys.exit(app.exec_())
