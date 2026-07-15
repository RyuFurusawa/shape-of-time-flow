"""リアルタイム軸間変換プレビュー (Phase 1)

読み込んだ動画の直近フレーム群を GPU に常駐させ、space/time/rate マップに従って
per-pixel gather する「軸間変換」を実時間で表示する。

- 変換自体は GPU (wgpu → Metal/D3D12/Vulkan) で実質ゼロコスト。
- ボトルネックは初回のデコード + 転送とメモリ (フレーム数 F × プレビュー解像度 S)。
- wgpu が使えない環境では numpy ベクトル化 gather に自動フォールバック (低速だが動く)。

意味論は「芸術的に十分な近似」:
    time モード: srcFrame(x,y) = playhead - timeMap01(x,y) * span
    rate モード: srcFrame(x,y) = playhead * (baseline + (rateMap01-0.5)*2*maxdev)
    共通       : srcX(x,y)     = spaceMap01(x,y) * (W-1)
playhead をループさせることでプレビューが「動く」。
"""

import os
import threading
import time

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QDoubleSpinBox, QSlider, QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt5.QtGui import QImage, QPixmap

try:
    import wgpu
    _HAS_WGPU = True
except Exception:
    _HAS_WGPU = False

FULLHD_W = 1920
DEFAULT_BUDGET_MB = 1024      # フレームボリュームのメモリ予算 (可変)
MIN_PREVIEW_W = 320           # ダウンスケールの下限幅
MAX_FRAMES = 512              # 常駐フレーム数の上限 (時間方向の細かさ)

# ウィジェット内 i18n (アプリ本体とは独立)
_T = {
    "placeholder": {"ja": "リアルタイムプレビュー: Initialize 後、画像を設定して再構築",
                     "en": "Realtime preview: after Initialize, set images and rebuild"},
    "play": {"ja": "▶ 再生", "en": "▶ Play"},
    "pause": {"ja": "⏸ 一時停止", "en": "⏸ Pause"},
    "speed": {"ja": "速度:", "en": "Speed:"},
    "rebuild": {"ja": "再構築 / Rebuild", "en": "Rebuild"},
    "no_video": {"ja": "動画が未設定です", "en": "No video set"},
    "read_fail": {"ja": "動画を読めませんでした", "en": "Could not read the video"},
    "decoding_prog": {
        "ja": "デコード中 {n}/{F} — 済んだ時間領域から色が付きます  (S={sw}×{sh}, ≈{mb:.0f} MB, {kind})",
        "en": "Decoding {n}/{F} — pixels fill in as frames arrive  (S={sw}×{sh}, ≈{mb:.0f} MB, {kind})"},
    "decode_fail": {"ja": "デコード失敗: {err}", "en": "Decode failed: {err}"},
    "ready": {"ja": "準備完了  S={sw}×{sh}, F={F}  ≈{mb:.0f} MB  [{kind}]",
              "en": "Ready  S={sw}×{sh}, F={F}  ≈{mb:.0f} MB  [{kind}]"},
    "build_center": {"ja": "▶ プレビューを構築 / Build Preview",
                      "en": "▶ Build Preview"},
    "mode_info": {"ja": "適用: {m}", "en": "mode: {m}"},
}


# ---- ボリューム設計 (メモリ予算から S, F を決める) ----
def plan_volume(in_w, in_h, total_frames, budget_mb=DEFAULT_BUDGET_MB):
    """(sw, sh, F) を返す。S は min(入力, FullHD) から予算内に収まるまで縮小。"""
    budget = budget_mb * 1024 * 1024
    sw = min(in_w, FULLHD_W)
    sh = max(1, int(round(in_h * sw / in_w)))
    F_target = min(total_frames, MAX_FRAMES)
    # S を優先的に保ち、収まらなければ S を段階縮小
    while sw > MIN_PREVIEW_W and sw * sh * 4 * F_target > budget:
        sw = int(sw * 0.8)
        sh = max(1, int(round(in_h * sw / in_w)))
    # それでも収まらなければ F を間引く (最後の手段)
    F_max = max(2, int(budget // (sw * sh * 4)))
    F = min(F_target, F_max)
    return sw, sh, F


def decode_volume_progressive(video_path, sw, sh, F, batch_cb,
                              cancel=None, batch=16):
    """動画を先頭から順次デコードし、等間隔サンプリングした F フレームを
    batch ごとに batch_cb(start_index, frames_array) で通知する。

    以前のランダムシーク方式 (cap.set POS_FRAMES) は HEVC 等ではシークごとに
    キーフレームから再デコードが走り極端に遅かった。ここでは grab() で
    順次読み進め、必要なフレームだけ retrieve() する (入力映像の前半から
    座標変換が進行していくプログレッシブ表示の供給源)。
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or F
    idxs = np.linspace(0, max(0, total - 1), F).astype(int)
    buf = []
    start = 0
    j = 0
    src_i = 0
    while j < F:
        if cancel is not None and cancel.is_set():
            break
        if not cap.grab():
            break
        if src_i == idxs[j]:
            ok, fr = cap.retrieve()
            if ok:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                if (fr.shape[1], fr.shape[0]) != (sw, sh):
                    fr = cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA)
                rgba = np.empty((sh, sw, 4), np.uint8)
                rgba[..., :3] = fr
                rgba[..., 3] = 255
            else:
                rgba = np.zeros((sh, sw, 4), np.uint8)
            buf.append(rgba)
            j += 1
            # linspace の丸めで同一 src インデックスが連続するケース
            while j < F and idxs[j] == src_i:
                buf.append(rgba)
                j += 1
            if len(buf) >= batch:
                batch_cb(start, np.stack(buf))
                start += len(buf)
                buf = []
        src_i += 1
    if buf and not (cancel is not None and cancel.is_set()):
        batch_cb(start, np.stack(buf))
    cap.release()


def load_map01(path, ow, oh, default="gray", transpose=False):
    """マップ PNG を出力解像度 (ow, oh) の float32 [0,1] にして返す。

    transpose=True: 横スリット (sd=0) 用。マップファイルは (scan, time) 形状で
    保存されているため (img_to_maneuver が .T で読む規約)、ビデオ座標系に
    合わせて転置してからリサイズする。default のランプはビデオ座標系で
    直接生成するので転置しない。
    """
    m = None
    if path and os.path.exists(path):
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        if default == "h":       # 左→右 ramp
            return np.tile(np.linspace(0, 1, ow, dtype=np.float32), (oh, 1))
        if default == "v":       # 上→下 ramp
            return np.tile(np.linspace(0, 1, oh, dtype=np.float32)[:, None], (1, ow))
        return np.full((oh, ow), 0.5, np.float32)   # gray (rate normal)
    if m.ndim == 3:
        m = m[..., 0]
    if transpose:
        m = m.T
    m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_LINEAR)
    mx = 65535.0 if m.dtype == np.uint16 else 255.0
    return np.ascontiguousarray(m.astype(np.float32) / mx)


# ---- uniform 構造 (16byte 整列) ----
# sd: 1=縦スリット (space で X をリマップ) / 0=横スリット (space で Y をリマップ)
_PARAMS_DTYPE = np.dtype({
    "names": ["F", "OW", "OH", "srcW", "mode", "playhead", "span",
              "baseline", "maxdev", "sd", "srcH", "_p2"],
    "formats": ["<u4", "<u4", "<u4", "<u4", "<u4", "<f4", "<f4",
                "<f4", "<f4", "<u4", "<u4", "<f4"],
})

_WGSL = """
struct P { F:u32, OW:u32, OH:u32, srcW:u32, mode:u32,
           playhead:f32, span:f32, baseline:f32, maxdev:f32,
           sd:u32, srcH:u32, _p2:f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var vol: texture_2d_array<f32>;
@group(0) @binding(2) var smap: texture_2d<f32>;
@group(0) @binding(3) var tmap: texture_2d<f32>;
@group(0) @binding(4) var rmap: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> outbuf: array<u32>;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.OW || gid.y >= p.OH) { return; }
  let xy = vec2<i32>(i32(gid.x), i32(gid.y));
  let s = clamp(textureLoad(smap, xy, 0).r, 0.0, 1.0);
  var sx: i32;
  var sy: i32;
  if (p.sd == 1u) {
    // 縦スリット: スリット=縦線。space は水平位置をリマップ、Y は素通し
    sx = i32(s * f32(p.srcW - 1u));
    sy = xy.y;
  } else {
    // 横スリット: スリット=横線。space は垂直位置をリマップ、X は素通し
    sx = xy.x;
    sy = i32(s * f32(p.srcH - 1u));
  }
  var srcF: f32;
  if (p.mode == 0u) {
    let t = textureLoad(tmap, xy, 0).r;
    srcF = p.playhead - t * p.span;
  } else {
    let r = textureLoad(rmap, xy, 0).r;
    let rate = p.baseline + (r - 0.5) * 2.0 * p.maxdev;
    srcF = p.playhead * rate;
  }
  let Ff = f32(p.F);
  srcF = srcF - floor(srcF / Ff) * Ff;          // positive modulo
  let f0 = i32(floor(srcF));
  let f1 = (f0 + 1) % i32(p.F);
  let fr = srcF - floor(srcF);
  let c0 = textureLoad(vol, vec2<i32>(sx, sy), f0, 0);
  let c1 = textureLoad(vol, vec2<i32>(sx, sy), f1, 0);
  let c = mix(c0, c1, fr);
  outbuf[gid.y * p.OW + gid.x] = pack4x8unorm(vec4<f32>(c.rgb, 1.0));
}
"""


class _WgpuBackend:
    """wgpu 実装。利用不可なら .ok == False。"""
    def __init__(self):
        self.ok = False
        self.device = None
        self._vol_tex = None
        self._map_tex = {}
        self._out = None
        self._read = None
        self._ubo = None
        self._pipeline = None
        self._bg = None
        self._dims = None
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            self.device = adapter.request_device_sync()
            self._shader = self.device.create_shader_module(code=_WGSL)
            self._pipeline = self.device.create_compute_pipeline(
                layout="auto", compute={"module": self._shader, "entry_point": "main"})
            self.ok = True
        except Exception:
            self.ok = False

    def alloc_volume(self, F, H, W):
        """フレームボリュームのテクスチャを確保する (中身はゼロ=黒で初期化)。

        WebGPU はテクスチャをゼロ初期化するため、未アップロード領域を参照する
        出力ピクセルは自動的に黒になる → プログレッシブ表示がそのまま成立する。
        """
        self._vol_shape = (F, H, W)
        self._vol_tex = self.device.create_texture(
            size=(W, H, F), format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d")

    def upload_frames(self, start, frames):
        """デコード済みバッチ (n, H, W, 4) をレイヤー start から書き込む。"""
        n, H, W, _ = frames.shape
        self.device.queue.write_texture(
            {"texture": self._vol_tex, "mip_level": 0, "origin": (0, 0, start)},
            np.ascontiguousarray(frames),
            {"offset": 0, "bytes_per_row": W * 4, "rows_per_image": H}, (W, H, n))

    def set_map(self, name, arr):
        oh, ow = arr.shape
        dev = self.device
        t = dev.create_texture(
            size=(ow, oh, 1), format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d")
        dev.queue.write_texture(
            {"texture": t, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(arr, np.float32),
            {"offset": 0, "bytes_per_row": ow * 4, "rows_per_image": oh}, (ow, oh, 1))
        self._map_tex[name] = t

    def finalize(self, ow, oh):
        dev = self.device
        self._dims = (ow, oh)
        out_bytes = ow * oh * 4
        self._out = dev.create_buffer(
            size=out_bytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        self._read = dev.create_buffer(
            size=out_bytes,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        self._ubo = dev.create_buffer(
            size=_PARAMS_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._bg = dev.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": {"buffer": self._ubo, "offset": 0,
                                            "size": _PARAMS_DTYPE.itemsize}},
                {"binding": 1, "resource": self._vol_tex.create_view(dimension="2d-array")},
                {"binding": 2, "resource": self._map_tex["space"].create_view()},
                {"binding": 3, "resource": self._map_tex["time"].create_view()},
                {"binding": 4, "resource": self._map_tex["rate"].create_view()},
                {"binding": 5, "resource": {"buffer": self._out, "offset": 0,
                                            "size": out_bytes}},
            ])

    def render(self, params):
        dev = self.device
        ow, oh = self._dims
        dev.queue.write_buffer(self._ubo, 0, params.tobytes())
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._pipeline)
        cp.set_bind_group(0, self._bg)
        cp.dispatch_workgroups((ow + 7) // 8, (oh + 7) // 8, 1)
        cp.end()
        enc.copy_buffer_to_buffer(self._out, 0, self._read, 0, ow * oh * 4)
        dev.queue.submit([enc.finish()])
        self._read.map_sync(wgpu.MapMode.READ)
        mv = self._read.read_mapped()
        img = np.frombuffer(bytes(mv), np.uint8).reshape(oh, ow, 4)[..., :3].copy()
        self._read.unmap()
        return img


class _NumpyBackend:
    """wgpu が無い環境向けのベクトル化 gather フォールバック。"""
    ok = True

    def __init__(self):
        self._vol = None
        self._maps = {}
        self._yy = None
        self._dims = None

    def alloc_volume(self, F, H, W):
        self._vol = np.zeros((F, H, W, 4), np.uint8)   # 黒で初期化

    def upload_frames(self, start, frames):
        self._vol[start:start + frames.shape[0]] = frames

    def set_map(self, name, arr):
        self._maps[name] = arr

    def finalize(self, ow, oh):
        self._dims = (ow, oh)
        self._yy = np.arange(oh)[:, None]

    def render(self, params):
        vol = self._vol
        F, H, W, _ = vol.shape
        ow, oh = self._dims
        s = np.clip(self._maps["space"], 0.0, 1.0)
        if int(params["sd"]) == 1:
            # 縦スリット: X をリマップ、Y 素通し
            ix = (s * (W - 1)).astype(np.int32)
            iy = np.broadcast_to(self._yy, (oh, ow))
        else:
            # 横スリット: Y をリマップ、X 素通し
            ix = np.broadcast_to(np.arange(ow)[None, :], (oh, ow))
            iy = (s * (H - 1)).astype(np.int32)
        if int(params["mode"]) == 0:
            t = self._maps["time"]
            srcF = float(params["playhead"]) - t * float(params["span"])
        else:
            r = self._maps["rate"]
            rate = float(params["baseline"]) + (r - 0.5) * 2.0 * float(params["maxdev"])
            srcF = float(params["playhead"]) * rate
        srcF = np.mod(srcF, F)
        f0 = np.floor(srcF).astype(np.int32)
        f1 = (f0 + 1) % F
        fr = (srcF - f0)[..., None]
        c0 = vol[f0, iy, ix, :3].astype(np.float32)
        c1 = vol[f1, iy, ix, :3].astype(np.float32)
        return (c0 * (1 - fr) + c1 * fr).astype(np.uint8)


class _DecodeWorker(threading.Thread):
    """順次デコード + バッチ通知のワーカー。cancel イベントで途中停止できる。"""

    def __init__(self, video, sw, sh, F, batch_cb, done_cb):
        super().__init__(daemon=True)
        self.video, self.sw, self.sh, self.F = video, sw, sh, F
        self.batch_cb = batch_cb
        self.done_cb = done_cb
        self.cancel = threading.Event()

    def run(self):
        try:
            decode_volume_progressive(
                self.video, self.sw, self.sh, self.F,
                batch_cb=self.batch_cb, cancel=self.cancel)
            if not self.cancel.is_set():
                self.done_cb(None)
        except Exception as e:  # noqa
            if not self.cancel.is_set():
                self.done_cb(str(e))


class RealtimePreviewWidget(QWidget):
    """Tab3 に埋め込む GPU リアルタイム軸間変換プレビュー。"""

    _batch_decoded = pyqtSignal(int, int, object)  # (gen, start, frames)
    _decode_finished = pyqtSignal(int, object)     # (gen, error)

    def __init__(self, lang="ja"):
        super().__init__()
        self.lang = lang if lang in ("ja", "en") else "ja"
        self.video_path = None
        self.space_path = None
        self.time_path = None
        self.rate_path = None
        self.mode = "time"
        self.scan_direction = 1     # 1=縦スリット / 0=横スリット (アプリから同期)
        self.space_set = None
        self.vmin = 0
        self.vmax = 100
        self.baseline = 1.0
        self.maxdev = 0.5

        self._backend = None
        self._gpu = None       # _WgpuBackend のキャッシュ (device 再利用)
        self._worker = None    # 実行中の _DecodeWorker
        self._gen = 0          # rebuild 世代 (古いワーカーのバッチを弾く)
        self._F = None         # 常駐フレーム数
        self._srcW = None
        self._srcH = None
        self._loaded = 0       # デコード/アップロード済みフレーム数
        self._vol_mb = 0.0
        self._dims = None      # (ow, oh)
        self._playhead = 0.0   # 常駐ボリューム内の位置 [0, F)
        self._qimg_buf = None

        # 出力タイムライン (タブ2の「時間方向サイズ」「出力FPS」に追従)
        # 実時間 = time_size / out_fps 秒。プレビューはこのタイムラインを
        # 低フレームレート (適応) でなぞる。
        self.time_size = 120
        self.out_fps = 30
        self._t_out = 0.0      # 出力フレーム位置 [0, time_size)
        self._scrub_was_playing = False

        # 速度優先: プレビューの描画は最大 ~15fps、描画が重ければ自動で更に
        # 間引く (最終書き出しの fps とは独立)。
        self._base_interval_ms = 66      # ≈15fps
        self._max_interval_ms = 250      # ≈4fps まで自動降下
        self._render_ema = 0.0           # 1フレーム描画時間の移動平均 (sec)

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(self._base_interval_ms)
        self._timer.timeout.connect(self._tick)
        self._batch_decoded.connect(self._on_batch_decoded)
        self._decode_finished.connect(self._on_decode_finished)

    def _t(self, key, **kw):
        d = _T.get(key, {})
        s = d.get(self.lang) or d.get("ja") or key
        return s.format(**kw) if kw else s

    def set_lang(self, lang):
        if lang in ("ja", "en"):
            self.lang = lang
            self._speed_label.setText(self._t("speed"))
            self.rebuild_btn.setText(self._t("rebuild"))
            self.center_btn.setText(self._t("build_center"))
            self.mode_label.setText(self._t("mode_info", m=self.mode))
            self._center_overlays()
            self.play_btn.setText(self._t("pause") if self._timer.isActive()
                                  else self._t("play"))
            if not self._backend:
                self.view.setText(self._t("placeholder"))

    # ---- UI ----
    def _build_ui(self):
        v = QVBoxLayout(self)
        self.view = QLabel(self._t("placeholder"))
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(480, 270)
        self.view.setStyleSheet(
            "QLabel { background: #111; color: #888; border: 1px solid #555; }")
        v.addWidget(self.view, 1)

        # --- 黒画面中央の「構築」ボタン (未構築のとき唯一の開始点) ---
        self.center_btn = QPushButton(self._t("build_center"), self.view)
        self.center_btn.setCursor(Qt.PointingHandCursor)
        self.center_btn.setStyleSheet(
            "QPushButton { background: #2a6fd6; color: white; border-radius: 8px;"
            " padding: 14px 28px; font-size: 15px; font-weight: bold; }"
            "QPushButton:hover { background: #3b82e6; }"
            "QPushButton:disabled { background: #444; color: #999; }")
        self.center_btn.adjustSize()
        self.center_btn.clicked.connect(self.rebuild)

        # --- デコード進捗バー (映像を隠さないよう下部に細く表示) ---
        # プログレッシブ表示: 再生は即開始し、デコード済み領域から色が付く。
        self.decode_bar = QProgressBar()
        self.decode_bar.setRange(0, 100)
        self.decode_bar.setValue(0)
        self.decode_bar.setFixedHeight(6)
        self.decode_bar.setTextVisible(False)
        self.decode_bar.setStyleSheet(
            "QProgressBar { background: #222; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background: #2a6fd6; border-radius: 3px; }")
        self.decode_bar.hide()
        v.addWidget(self.decode_bar)

        # view のリサイズに追従してオーバーレイを中央へ
        self.view.installEventFilter(self)

        # --- トランスポート行 (QuickTime 風): ▶ [====|----] 0:04 / 0:12 ---
        tl = QHBoxLayout()
        self.play_btn = QPushButton(self._t("play"))
        self.play_btn.setFixedWidth(96)
        self.play_btn.clicked.connect(self._toggle_play)
        tl.addWidget(self.play_btn)

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, max(1, self.time_size - 1))
        self.timeline.sliderPressed.connect(self._on_scrub_start)
        self.timeline.sliderMoved.connect(self._on_scrub_move)
        self.timeline.sliderReleased.connect(self._on_scrub_end)
        tl.addWidget(self.timeline, 1)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        tl.addWidget(self.time_label)
        v.addLayout(tl)

        # --- 設定行 ---
        ctl = QHBoxLayout()
        # 適用モードはタブ2の「適用方法」に自動追従 (ここでは表示のみ)
        self.mode_label = QLabel(self._t("mode_info", m=self.mode))
        self.mode_label.setStyleSheet("color: gray; font-size: 11px;")
        ctl.addWidget(self.mode_label)

        self._speed_label = QLabel(self._t("speed"))
        ctl.addWidget(self._speed_label)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.0, 8.0)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        ctl.addWidget(self.speed_spin)

        # 構築後の再構築用 (未構築時は中央ボタンが主役)
        self.rebuild_btn = QPushButton(self._t("rebuild"))
        self.rebuild_btn.clicked.connect(self.rebuild)
        ctl.addWidget(self.rebuild_btn)
        ctl.addStretch()
        v.addLayout(ctl)

        self.status = QLabel("")
        self.status.setStyleSheet("color: gray; font-size: 11px;")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        self._update_time_label()

    # --- 中央オーバーレイの配置/演算中アニメ ---
    def eventFilter(self, obj, ev):
        if obj is self.view and ev.type() == QEvent.Resize:
            self._center_overlays()
        return super().eventFilter(obj, ev)

    def _center_overlays(self):
        w = self.center_btn
        w.adjustSize()
        w.move((self.view.width() - w.width()) // 2,
               (self.view.height() - w.height()) // 2)

    # ---- 外部 API ----
    def set_video(self, path):
        changed = (path != self.video_path)
        self.video_path = path
        if changed:
            # 動画が変わったら既存ボリュームは無効 → 中央の構築ボタンに戻す
            self.stop()
            if self._worker is not None:
                self._worker.cancel.set()
            self._gen += 1
            self._backend = None
            self._F = None
            self.decode_bar.hide()
            self.view.setPixmap(QPixmap())
            self.view.setText(self._t("placeholder"))
            self.center_btn.show()
            self._center_overlays()

    def set_maps(self, space_path=None, time_path=None, rate_path=None):
        self.space_path = space_path
        self.time_path = time_path
        self.rate_path = rate_path

    def set_params(self, mode=None, space_set=None, vmin=None, vmax=None,
                   baseline=None, maxdev=None, time_size=None, out_fps=None,
                   sd=None):
        if sd is not None and int(sd) in (0, 1) and int(sd) != self.scan_direction:
            self.scan_direction = int(sd)
            # マップの向き (転置/デフォルトランプ) が変わるため再アップロード
            self.refresh_maps()
        if mode in ("time", "rate"):
            # モードはタブ2「適用方法」に追従する (ウィジェット内に切替 UI は無い)
            changed = (mode != self.mode)
            self.mode = mode
            self.mode_label.setText(self._t("mode_info", m=mode))
            if changed:
                self._render_once()
        if space_set is not None: self.space_set = space_set
        if vmin is not None: self.vmin = vmin
        if vmax is not None: self.vmax = vmax
        if baseline is not None: self.baseline = baseline
        if maxdev is not None: self.maxdev = maxdev
        if time_size is not None and int(time_size) > 0:
            self.time_size = int(time_size)
            self.timeline.blockSignals(True)
            self.timeline.setRange(0, max(1, self.time_size - 1))
            self.timeline.blockSignals(False)
            self._t_out = min(self._t_out, self.time_size - 1)
        if out_fps is not None and int(out_fps) > 0:
            self.out_fps = int(out_fps)
        if time_size is not None or out_fps is not None:
            self._update_time_label()

    # ---- タイムライン (出力時間軸) ----
    @staticmethod
    def _fmt_time(sec):
        sec = max(0, int(round(sec)))
        return f"{sec // 60}:{sec % 60:02d}"

    def _update_time_label(self):
        cur = self._t_out / max(1, self.out_fps)
        total = self.time_size / max(1, self.out_fps)
        self.time_label.setText(f"{self._fmt_time(cur)} / {self._fmt_time(total)}")
        self.timeline.blockSignals(True)
        self.timeline.setValue(int(self._t_out))
        self.timeline.blockSignals(False)

    def _playhead_from_tout(self):
        """出力フレーム位置 → 常駐ボリューム内 playhead へ写像。"""
        if self._F is None:
            return 0.0
        return (self._t_out / max(1, self.time_size)) * self._F

    def _on_scrub_start(self):
        self._scrub_was_playing = self._timer.isActive()
        self.stop()

    def _on_scrub_move(self, value):
        self._t_out = float(value)
        self._playhead = self._playhead_from_tout()
        cur = self._t_out / max(1, self.out_fps)
        total = self.time_size / max(1, self.out_fps)
        self.time_label.setText(f"{self._fmt_time(cur)} / {self._fmt_time(total)}")
        self._render_once()

    def _on_scrub_end(self):
        if self._scrub_was_playing:
            self.start()

    def _backend_kind(self):
        return "GPU/wgpu" if isinstance(self._backend, _WgpuBackend) else "CPU/numpy"

    def rebuild(self):
        """プログレッシブ再構築: 黒ボリュームを確保して即再生を開始し、
        バックグラウンドの順次デコードが進むにつれてピクセルに色が付いていく。"""
        if not self.video_path or not os.path.exists(self.video_path):
            self.status.setText(self._t("no_video"))
            return
        cap = cv2.VideoCapture(self.video_path)
        in_w = int(cap.get(3)); in_h = int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if in_w <= 0 or in_h <= 0:
            self.status.setText(self._t("read_fail"))
            return

        # 前世代のワーカーを止める (バッチは世代番号で弾く)
        if self._worker is not None:
            self._worker.cancel.set()
        self._gen += 1

        sw, sh, F = plan_volume(in_w, in_h, total)
        ow, oh = sw, sh
        self._F = F
        self._srcW, self._srcH = sw, sh
        self._dims = (ow, oh)
        self._loaded = 0
        self._vol_mb = sw * sh * 4 * F / 1e6

        # backend 準備 (GPU device はキャッシュして再利用)
        if _HAS_WGPU:
            if self._gpu is None:
                self._gpu = _WgpuBackend()
            be = self._gpu if self._gpu.ok else _NumpyBackend()
        else:
            be = _NumpyBackend()
        self._backend = be
        be.alloc_volume(F, sh, sw)      # ゼロ初期化 = 黒
        self._upload_maps(be, ow, oh)
        be.finalize(ow, oh)

        # 即再生開始 (黒画面から始まり、デコード済み領域から色が付く)
        self._playhead = 0.0
        self._t_out = 0.0
        self._update_time_label()
        self.center_btn.hide()
        self.decode_bar.setValue(0)
        self.decode_bar.show()
        self.status.setText(self._t(
            "decoding_prog", n=0, F=F, sw=sw, sh=sh, mb=self._vol_mb,
            kind=self._backend_kind()))
        self._render_once()
        self.start()

        gen = self._gen
        self._worker = _DecodeWorker(
            self.video_path, sw, sh, F,
            batch_cb=lambda s, fr, g=gen: self._batch_decoded.emit(g, s, fr),
            done_cb=lambda err, g=gen: self._decode_finished.emit(g, err))
        self._worker.start()

    def _on_batch_decoded(self, gen, start, frames):
        if gen != self._gen or self._backend is None:
            return      # 旧世代ワーカーの残りバッチは無視
        self._backend.upload_frames(start, frames)
        self._loaded = max(self._loaded, start + frames.shape[0])
        F = self._F or 1
        self.decode_bar.setValue(int(self._loaded * 100 / F))
        self.status.setText(self._t(
            "decoding_prog", n=self._loaded, F=F,
            sw=self._srcW, sh=self._srcH, mb=self._vol_mb,
            kind=self._backend_kind()))
        if not self._timer.isActive():
            self._render_once()     # 一時停止中でも埋まっていく様子を反映

    def _on_decode_finished(self, gen, err):
        if gen != self._gen:
            return
        self.decode_bar.hide()
        if err:
            self.status.setText(self._t("decode_fail", err=err))
            return
        ow, oh = self._dims
        self.status.setText(self._t(
            "ready", sw=self._srcW, sh=self._srcH, F=self._F,
            mb=self._vol_mb, kind=self._backend_kind()))

    def _upload_maps(self, be, ow, oh):
        """3 マップをスリット方向に応じた向きでバックエンドへ転送する。

        縦スリット (sd=1): ファイル形状 (time, scan) がビデオ座標と一致 → そのまま。
            通常再生の既定: space=左→右 ramp, time=上→下 ramp
        横スリット (sd=0): ファイル形状 (scan, time) → ビデオ座標へ転置が必要。
            通常再生の既定: space=上→下 ramp, time=左→右 ramp
        """
        vertical = (self.scan_direction == 1)
        tp = not vertical
        sp_def = "h" if vertical else "v"
        tm_def = "v" if vertical else "h"
        be.set_map("space", load_map01(self.space_path, ow, oh, default=sp_def, transpose=tp))
        be.set_map("time", load_map01(self.time_path, ow, oh, default=tm_def, transpose=tp))
        be.set_map("rate", load_map01(self.rate_path, ow, oh, default="gray", transpose=tp))

    def refresh_maps(self):
        """マップだけ差し替え (ボリューム再デコードなし)。"""
        if not self._backend or not self._dims:
            return
        ow, oh = self._dims
        self._upload_maps(self._backend, ow, oh)
        self._render_once()

    # ---- 再生 ----
    def start(self):
        if self._backend and not self._timer.isActive():
            self._timer.start()
            self.play_btn.setText(self._t("pause"))

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText(self._t("play"))

    def _toggle_play(self):
        if self._timer.isActive():
            self.stop()
        else:
            self.start()

    def _params(self):
        F = self._F
        p = np.zeros((), _PARAMS_DTYPE)
        p["F"] = F
        p["OW"], p["OH"] = self._dims
        p["srcW"] = self._srcW
        p["mode"] = 0 if self.mode == "time" else 1
        p["playhead"] = self._playhead
        p["span"] = float(F)
        p["baseline"] = float(self.baseline)
        p["maxdev"] = float(self.maxdev)
        p["sd"] = int(self.scan_direction)
        p["srcH"] = self._srcH
        return p

    def _tick(self):
        if not self._backend:
            return
        # 出力タイムライン上を実時間で進める:
        #   1 tick = interval 秒 → Δt_out = out_fps × speed × interval
        interval_sec = max(1, self._timer.interval()) / 1000.0
        self._t_out = (self._t_out +
                       self.out_fps * self.speed_spin.value() * interval_sec) \
                      % max(1, self.time_size)
        self._playhead = self._playhead_from_tout()
        self._render_once()
        self._update_time_label()

    def _render_once(self):
        if not self._backend or self._F is None:
            return
        t0 = time.time()
        img = self._backend.render(self._params())
        ow, oh = self._dims
        self._qimg_buf = np.ascontiguousarray(img)
        qimg = QImage(self._qimg_buf.data, ow, oh, ow * 3, QImage.Format_RGB888)
        # 速度優先: 拡縮は FastTransformation (SmoothTransformation は CPU コスト大)
        pm = QPixmap.fromImage(qimg).scaled(
            self.view.width(), self.view.height(),
            Qt.KeepAspectRatio, Qt.FastTransformation)
        self.view.setPixmap(pm)
        # --- 適応フレームレート: 描画が重いときは自動で間引く ---
        dt = time.time() - t0
        self._render_ema = dt if self._render_ema == 0 else \
            0.8 * self._render_ema + 0.2 * dt
        desired = int(min(self._max_interval_ms,
                          max(self._base_interval_ms, self._render_ema * 1500)))
        if abs(desired - self._timer.interval()) > 15:
            self._timer.setInterval(desired)
