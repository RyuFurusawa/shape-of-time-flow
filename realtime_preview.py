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

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
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
    "decoding": {"ja": "デコード中… S={sw}×{sh}, F={F}  (≈{mb:.0f} MB, backend={be})",
                 "en": "Decoding… S={sw}×{sh}, F={F}  (≈{mb:.0f} MB, backend={be})"},
    "decode_fail": {"ja": "デコード失敗: {err}", "en": "Decode failed: {err}"},
    "ready": {"ja": "準備完了  S={sw}×{sh}, F={F}  ≈{mb:.0f} MB  [{kind}]",
              "en": "Ready  S={sw}×{sh}, F={F}  ≈{mb:.0f} MB  [{kind}]"},
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


def decode_volume(video_path, sw, sh, F):
    """動画から F フレームを等間隔サンプリングし (F, sh, sw, 4) uint8 を返す。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or F
    idxs = np.linspace(0, max(0, total - 1), F).astype(int)
    vol = np.zeros((F, sh, sw, 4), np.uint8)
    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        if (fr.shape[1], fr.shape[0]) != (sw, sh):
            fr = cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA)
        vol[i, :, :, :3] = fr
        vol[i, :, :, 3] = 255
    cap.release()
    return vol


def load_map01(path, ow, oh, default="gray"):
    """マップ PNG を出力解像度 (ow, oh) の float32 [0,1] にして返す。"""
    m = None
    if path and os.path.exists(path):
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        if default == "h":       # 左→右 ramp (space normal)
            return np.tile(np.linspace(0, 1, ow, dtype=np.float32), (oh, 1))
        if default == "v":       # 上→下 ramp (time normal)
            return np.tile(np.linspace(0, 1, oh, dtype=np.float32)[:, None], (1, ow))
        return np.full((oh, ow), 0.5, np.float32)   # gray (rate normal)
    if m.ndim == 3:
        m = m[..., 0]
    m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_LINEAR)
    mx = 65535.0 if m.dtype == np.uint16 else 255.0
    return np.ascontiguousarray(m.astype(np.float32) / mx)


# ---- uniform 構造 (16byte 整列) ----
_PARAMS_DTYPE = np.dtype({
    "names": ["F", "OW", "OH", "srcW", "mode", "playhead", "span",
              "baseline", "maxdev", "_p0", "_p1", "_p2"],
    "formats": ["<u4", "<u4", "<u4", "<u4", "<u4", "<f4", "<f4",
                "<f4", "<f4", "<f4", "<f4", "<f4"],
})

_WGSL = """
struct P { F:u32, OW:u32, OH:u32, srcW:u32, mode:u32,
           playhead:f32, span:f32, baseline:f32, maxdev:f32,
           _p0:f32, _p1:f32, _p2:f32 };
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
  let s = textureLoad(smap, xy, 0).r;
  let srcX = i32(clamp(s, 0.0, 1.0) * f32(p.srcW - 1u));
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
  let c0 = textureLoad(vol, vec2<i32>(srcX, xy.y), f0, 0);
  let c1 = textureLoad(vol, vec2<i32>(srcX, xy.y), f1, 0);
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

    def set_volume(self, vol):
        F, H, W, _ = vol.shape
        dev = self.device
        self._vol_tex = dev.create_texture(
            size=(W, H, F), format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d")
        dev.queue.write_texture(
            {"texture": self._vol_tex, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(vol),
            {"offset": 0, "bytes_per_row": W * 4, "rows_per_image": H}, (W, H, F))

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

    def set_volume(self, vol):
        self._vol = vol

    def set_map(self, name, arr):
        self._maps[name] = arr

    def finalize(self, ow, oh):
        self._dims = (ow, oh)
        self._yy = np.arange(oh)[:, None]

    def render(self, params):
        vol = self._vol
        F, H, W, _ = vol.shape
        ow, oh = self._dims
        s = self._maps["space"]
        srcX = np.clip(s * (W - 1), 0, W - 1).astype(np.int32)
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
        yy = np.broadcast_to(self._yy, (oh, ow))
        c0 = vol[f0, yy, srcX, :3].astype(np.float32)
        c1 = vol[f1, yy, srcX, :3].astype(np.float32)
        return (c0 * (1 - fr) + c1 * fr).astype(np.uint8)


class _DecodeWorker(threading.Thread):
    def __init__(self, video, sw, sh, F, done_cb):
        super().__init__(daemon=True)
        self.video, self.sw, self.sh, self.F = video, sw, sh, F
        self.done_cb = done_cb

    def run(self):
        try:
            vol = decode_volume(self.video, self.sw, self.sh, self.F)
            self.done_cb(vol, None)
        except Exception as e:  # noqa
            self.done_cb(None, str(e))


class RealtimePreviewWidget(QWidget):
    """Tab3 に埋め込む GPU リアルタイム軸間変換プレビュー。"""

    _volume_ready = pyqtSignal(object, object)  # (vol, error)

    def __init__(self, lang="ja"):
        super().__init__()
        self.lang = lang if lang in ("ja", "en") else "ja"
        self.video_path = None
        self.space_path = None
        self.time_path = None
        self.rate_path = None
        self.mode = "time"
        self.space_set = None
        self.vmin = 0
        self.vmax = 100
        self.baseline = 1.0
        self.maxdev = 0.5

        self._backend = None
        self._vol = None
        self._dims = None      # (ow, oh)
        self._playhead = 0.0
        self._qimg_buf = None

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(33)   # ~30fps
        self._timer.timeout.connect(self._tick)
        self._volume_ready.connect(self._on_volume_ready)

    def _t(self, key, **kw):
        d = _T.get(key, {})
        s = d.get(self.lang) or d.get("ja") or key
        return s.format(**kw) if kw else s

    def set_lang(self, lang):
        if lang in ("ja", "en"):
            self.lang = lang
            self._speed_label.setText(self._t("speed"))
            self.rebuild_btn.setText(self._t("rebuild"))
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

        ctl = QHBoxLayout()
        self.play_btn = QPushButton(self._t("play"))
        self.play_btn.clicked.connect(self._toggle_play)
        ctl.addWidget(self.play_btn)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["time", "rate"])
        self.mode_combo.currentTextChanged.connect(self._on_mode)
        ctl.addWidget(self.mode_combo)

        self._speed_label = QLabel(self._t("speed"))
        ctl.addWidget(self._speed_label)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.0, 8.0)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        ctl.addWidget(self.speed_spin)

        self.rebuild_btn = QPushButton(self._t("rebuild"))
        self.rebuild_btn.clicked.connect(self.rebuild)
        ctl.addWidget(self.rebuild_btn)
        ctl.addStretch()
        v.addLayout(ctl)

        self.status = QLabel("")
        self.status.setStyleSheet("color: gray; font-size: 11px;")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

    # ---- 外部 API ----
    def set_video(self, path):
        self.video_path = path

    def set_maps(self, space_path=None, time_path=None, rate_path=None):
        self.space_path = space_path
        self.time_path = time_path
        self.rate_path = rate_path

    def set_params(self, mode=None, space_set=None, vmin=None, vmax=None,
                   baseline=None, maxdev=None):
        if mode in ("time", "rate"):
            self.mode = mode
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(mode)
            self.mode_combo.blockSignals(False)
        if space_set is not None: self.space_set = space_set
        if vmin is not None: self.vmin = vmin
        if vmax is not None: self.vmax = vmax
        if baseline is not None: self.baseline = baseline
        if maxdev is not None: self.maxdev = maxdev

    def rebuild(self):
        """ボリュームをバックグラウンドでデコードし直し、マップを再アップロード。"""
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
        sw, sh, F = plan_volume(in_w, in_h, total)
        self._pending_dims = (sw, sh, F)
        mb = sw * sh * 4 * F / 1e6
        self.status.setText(self._t("decoding", sw=sw, sh=sh, F=F, mb=mb,
                                    be="GPU/wgpu" if _HAS_WGPU else "CPU/numpy"))
        self.rebuild_btn.setEnabled(False)
        _DecodeWorker(self.video_path, sw, sh, F,
                      lambda vol, err: self._volume_ready.emit(vol, err)).start()

    def _on_volume_ready(self, vol, err):
        self.rebuild_btn.setEnabled(True)
        if err or vol is None:
            self.status.setText(self._t("decode_fail", err=err))
            return
        sw, sh, F = self._pending_dims
        ow, oh = sw, sh
        self._vol = vol
        self._dims = (ow, oh)
        # backend 準備
        be = _WgpuBackend() if _HAS_WGPU else None
        if be is None or not be.ok:
            be = _NumpyBackend()
        self._backend = be
        be.set_volume(vol)
        be.set_map("space", load_map01(self.space_path, ow, oh, default="h"))
        be.set_map("time", load_map01(self.time_path, ow, oh, default="v"))
        be.set_map("rate", load_map01(self.rate_path, ow, oh, default="gray"))
        be.finalize(ow, oh)
        self._playhead = 0.0
        kind = "GPU/wgpu" if isinstance(be, _WgpuBackend) else "CPU/numpy"
        mb = sw * sh * 4 * F / 1e6
        self.status.setText(self._t("ready", sw=sw, sh=sh, F=F, mb=mb, kind=kind))
        self._render_once()
        self.start()

    def refresh_maps(self):
        """マップだけ差し替え (ボリューム再デコードなし)。"""
        if not self._backend or not self._dims:
            return
        ow, oh = self._dims
        self._backend.set_map("space", load_map01(self.space_path, ow, oh, default="h"))
        self._backend.set_map("time", load_map01(self.time_path, ow, oh, default="v"))
        self._backend.set_map("rate", load_map01(self.rate_path, ow, oh, default="gray"))
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

    def _on_mode(self, txt):
        self.mode = txt
        self._render_once()

    def _params(self):
        F = self._vol.shape[0]
        p = np.zeros((), _PARAMS_DTYPE)
        p["F"] = F
        p["OW"], p["OH"] = self._dims
        p["srcW"] = self._vol.shape[2]
        p["mode"] = 0 if self.mode == "time" else 1
        p["playhead"] = self._playhead
        p["span"] = float(F)
        p["baseline"] = float(self.baseline)
        p["maxdev"] = float(self.maxdev)
        return p

    def _tick(self):
        if not self._backend:
            return
        F = self._vol.shape[0]
        # speed(≈1.0 で resident window を ~6 秒で一周)
        self._playhead = (self._playhead + self.speed_spin.value() * F / (6 * 30)) % F
        self._render_once()

    def _render_once(self):
        if not self._backend or self._vol is None:
            return
        img = self._backend.render(self._params())
        ow, oh = self._dims
        self._qimg_buf = np.ascontiguousarray(img)
        qimg = QImage(self._qimg_buf.data, ow, oh, ow * 3, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg).scaled(
            self.view.width(), self.view.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(pm)
