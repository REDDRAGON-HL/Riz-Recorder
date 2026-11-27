"""PySide6 preview window with an embedded moderngl playfield."""

from __future__ import annotations

import json
import os
import time
from bisect import bisect_right

import moderngl
import numpy as np
from PySide6.QtCore import QEvent, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.audio import AudioPlayer
from ..core.chart import load_chart
from ..core.chart.chart import NOTE_DRAG
from ..core.render.assets import AssetStore
from ..core.render import watch
from ..core.render.renderer import RenderSettings, Renderer
from ..paths import data_root, resource_root
from .export_dialog import ExportDialog, ExportWorker
from .library import ChartEntry, import_paths, match_cover

ROOT = data_root()
ASSETS = str(resource_root() / "assets")
# Hit sounds are queued this far ahead of the play position (seconds of chart
# time) so the audio callback can place them sample-accurately.
SFX_LOOKAHEAD = 0.5
# Preview mix headroom: the export normalises its mix, the realtime callback
# only clips, so keep BGM and hits below full scale.
BGM_GAIN = 0.8
SFX_GAIN = 0.55

QSS = """
QMainWindow, QWidget { background: #14161c; color: #e8e8ef;
    font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 12px; }
QGroupBox { border: 1px solid #2a2f3a; border-radius: 7px; margin-top: 9px;
    padding: 8px 10px 6px 10px; font-weight: 600; color: #b9c0cf; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLabel { color: #c8cfdd; }
QLabel#muted { color: #7d8597; }
QLabel#time { color: #aab3c5; font-family: Consolas, monospace; }
QPushButton { background: #262b36; border: 1px solid #343b49; border-radius: 6px;
    padding: 6px 10px; color: #e8e8ef; }
QPushButton:hover { background: #303747; border-color: #465064; }
QPushButton:pressed { background: #1d222b; }
QPushButton#accent { background: #4262c8; border-color: #5a78e0; font-weight: 600; }
QPushButton#accent:hover { background: #4f6fdc; }
QLineEdit { background: #1b1f27; border: 1px solid #2f3643; border-radius: 5px;
    padding: 5px 7px; color: #e8e8ef; }
QSlider::groove:horizontal { height: 4px; background: #2b3240; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px;
    background: #8ba4f5; }
QSlider::sub-page:horizontal { background: #4262c8; border-radius: 2px; }
QListWidget { background: #1b1f27; border: 1px solid #2f3643; border-radius: 6px;
    outline: 0; }
QListWidget::item:selected { background: #4262c8; }
QListWidget#tasks { outline: 0; }
QListWidget#tasks::item { border: none; outline: 0; background: transparent; }
QListWidget#tasks::item:selected { background: transparent; }
QListWidget#nav { background: #101218; border: none; border-right: 1px solid #2a2f3a;
    border-radius: 0; padding: 10px 8px; outline: 0; }
QListWidget#nav::item { padding: 9px 12px; margin: 2px 0; border-radius: 6px; color: #c8cfdd; }
QListWidget#nav::item:hover { background: #262b36; }
QListWidget#nav::item:selected { background: #4262c8; color: #ffffff; }
QComboBox { background: #1b1f27; border: 1px solid #2f3643; border-radius: 5px; padding: 4px; }
QSpinBox, QDoubleSpinBox { background: #1b1f27; border: 1px solid #2f3643;
    border-radius: 5px; padding: 3px 4px; color: #e8e8ef; }
QCheckBox { color: #c8cfdd; spacing: 6px; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: #14161c; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #3a4250; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #4d586c; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QFrame#drawer, QFrame#panel, QFrame#previewcard { background: transparent; border: none; }
QFrame#topbar { background: #101218; border-bottom: 1px solid #2a2f3a; }
QLabel#apptitle { font-size: 18px; font-weight: 700; letter-spacing: 2px; color: #a2b3ef; }
QLabel#appsub { color: #7d8597; font-size: 11px; }
"""


def _fmt_time(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    total_ms = int(round(seconds * 1000))
    minutes, rem = divmod(total_ms, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{sign}{minutes:02d}:{secs:02d}.{ms:03d}"


class _Presenter:
    """Copy the scene FBO onto the widget's framebuffer with a trivial shader.

    Qt rewrites GL state around paintGL, so doing the heavy render in our own
    FBO and then blitting with an opaque quad keeps the result deterministic.
    """

    VS = """#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}"""

    FS = """#version 330 core
uniform sampler2D tex;
in vec2 v_uv;
out vec4 frag;
void main() { frag = vec4(texture(tex, v_uv).rgb, 1.0); }"""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=self.VS, fragment_shader=self.FS)
        self.prog["tex"].value = 0
        quad = np.array(
            [
                -1.0, -1.0, 0.0, 0.0,
                1.0, -1.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                -1.0, -1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                -1.0, 1.0, 0.0, 1.0,
            ],
            dtype="f4",
        )
        self.vbo = ctx.buffer(quad.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")])

    def draw(self, texture, target, x: int, y: int, w: int, h: int) -> None:
        target.use()
        self.ctx.viewport = (x, y, w, h)
        self.ctx.disable(moderngl.BLEND)
        texture.use(0)
        self.vao.render()


class PreviewGL(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 520)
        self.ctx: moderngl.Context | None = None
        self.assets: AssetStore | None = None
        self.renderer: Renderer | None = None
        self.chart = None
        self.settings = RenderSettings()
        self.current_time = 0.0
        self.last_frame_ms = 0.0
        self._presenter: _Presenter | None = None
        self._scene_fbo = None
        self._scene_tex = None
        self._scene_ms_fbo = None
        self._scene_ms_tex = None
        self._scene_size = (0, 0)
        self.aspect = 9.0 / 16.0

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context(require=330)
        self._presenter = _Presenter(self.ctx)
        self.assets = AssetStore(self.ctx, ASSETS)
        if self.chart is not None:
            self._create_renderer()

    def _ensure_scene(self, w: int, h: int) -> None:
        if self._scene_size == (w, h) and self._scene_fbo is not None:
            return
        for obj in (self._scene_fbo, self._scene_tex, self._scene_ms_fbo, self._scene_ms_tex):
            if obj is not None:
                obj.release()
        max_samples = int(self.ctx.info.get("GL_MAX_SAMPLES", 0) or 0)
        self._scene_tex = self.ctx.texture((w, h), 4)
        self._scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._scene_fbo = self.ctx.framebuffer([self._scene_tex])
        if max_samples >= 4:
            ms_samples = 8 if max_samples >= 8 else 4
            self._scene_ms_tex = self.ctx.texture((w, h), 4, samples=ms_samples)
            self._scene_ms_fbo = self.ctx.framebuffer([self._scene_ms_tex])
        else:
            self._scene_ms_tex = None
            self._scene_ms_fbo = None
        self._scene_size = (w, h)

    def _create_renderer(self) -> None:
        if self.ctx is None or self.chart is None:
            return
        # Renderer creates a program/VAO/buffers, so the GL context must be
        # current even when this is triggered by a UI action rather than paint.
        self.makeCurrent()
        try:
            self.renderer = Renderer(
                self.ctx, self.chart, self.assets, 1080, 1920, self.settings
            )
        finally:
            self.doneCurrent()

    def set_chart(self, chart) -> None:
        self.chart = chart
        self.settings.song_name = chart.song_name or "Untitled"
        self._create_renderer()
        self.update()

    def settings_changed(self, rebuild_text: bool = False) -> None:
        if self.renderer is not None:
            self.renderer.settings = self.settings
            if rebuild_text:
                self.renderer.update_meta()
        self.update()

    def set_time(self, t: float) -> None:
        self.current_time = t

    def paintGL(self) -> None:
        if self.ctx is None:
            return
        qt = self.ctx.detect_framebuffer()
        qt.use()
        fw, fh = qt.size
        self.ctx.viewport = (0, 0, fw, fh)
        self.ctx.clear(0.02, 0.02, 0.03, 1.0)
        if os.environ.get("RIZREC_TEST_RED"):
            qt.clear(1.0, 0.0, 0.0, 1.0)
            return
        if self.renderer is None:
            return
        aspect = self.aspect
        if fw / fh > aspect:
            h = fh
            w = max(1, int(h * aspect))
        else:
            w = fw
            h = max(1, int(w / aspect))
        x = (fw - w) // 2
        y = (fh - h) // 2

        self._ensure_scene(w, h)
        scene_target = self._scene_ms_fbo or self._scene_fbo
        scene_target.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.02, 0.02, 0.03, 1.0)
        self.ctx.enable(moderngl.BLEND)
        # Qt's compositor rewrites GL blend state behind moderngl's cache.
        # Toggle through another pair so the straight-alpha blend is really set.
        self.ctx.blend_func = (moderngl.ONE, moderngl.ZERO)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.renderer.resize(w, h)
        start = time.perf_counter()
        self.renderer.render(self.current_time)
        self.ctx.finish()
        if self._scene_ms_fbo is not None:
            self.ctx.copy_framebuffer(self._scene_fbo, self._scene_ms_fbo)
        self.last_frame_ms = (time.perf_counter() - start) * 1000.0
        if os.environ.get("RIZREC_DEBUG_SCENE"):
            data = self._scene_fbo.read(components=3)
            ww, hh = w, h
            yy = hh - 600
            samples = {}
            for xx in (0, 20, 50, 100, 350):
                i = (yy * ww + xx) * 3
                samples[xx] = tuple(data[i : i + 3])
            print("scene pixels", samples, flush=True)
        self._presenter.draw(self._scene_tex, qt, x, y, w, h)
        if os.environ.get("RIZREC_DEBUG_SCENE"):
            qdata = qt.read(components=3)
            qw, qh = fw, fh
            yy = qh - 600
            samples = {}
            for xx in (x, x + 20, x + 50, x + 100, x + 350):
                i = (yy * qw + xx) * 3
                samples[xx] = tuple(qdata[i : i + 3])
            print("qt pixels", samples, flush=True)
        if os.environ.get("RIZREC_DEBUG"):
            print(
                "paint",
                "fbo", (fw, fh),
                "scene", (w, h),
                "vp", (x, y, w, h),
                "err", self.ctx.error,
                flush=True,
            )


class TaskRow(QWidget):
    """Task list row: progress bar next to the buttons, same height as them."""

    TRACK = QColor(255, 255, 255, 16)
    FILL = QColor(70, 140, 255, 155)
    BAR_LEFT = 2

    def __init__(self, text: str):
        super().__init__()
        self._value = 0.0
        self.setMinimumHeight(34)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        self.label = QLabel(text)
        # The global `QWidget { background: #14161c }` rule would make the
        # label an opaque block covering the bar (it used to look like the bar
        # was only an outline around the text).
        self.label.setStyleSheet("background: transparent;")
        lay.addWidget(self.label, 1)

    def set_progress(self, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        if abs(value - self._value) < 1e-4:
            return
        self._value = value
        self.update()

    def _bar_rect(self) -> QRect:
        """Bar spans from the row's left edge up to (not under) the buttons."""
        buttons = self.findChildren(QPushButton)
        if buttons:
            right = min(b.x() for b in buttons) - 6
            height = max(12, min(b.height() for b in buttons))
        else:
            right = self.width() - 8
            height = max(12, self.height() - 8)
        top = (self.height() - height) // 2
        return QRect(self.BAR_LEFT, top, max(4, right - self.BAR_LEFT), height)

    def paintEvent(self, _event) -> None:
        bar = self._bar_rect()
        if bar.width() <= 4 or bar.height() <= 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.NoPen))
        painter.setBrush(self.TRACK)
        painter.drawRoundedRect(bar, 4, 4)
        if self._value <= 0.0:
            return
        width = max(4, int(round(bar.width() * self._value)))
        painter.setBrush(self.FILL)
        painter.drawRoundedRect(
            QRect(bar.left(), bar.top(), min(width, bar.width()), bar.height()), 4, 4
        )


class PreviewWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Riz Recorder")
        self.resize(1120, 732)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(QSS)
        self.setStyleSheet(QSS)

        self.audio = AudioPlayer()
        self.audio.load_sfx(os.path.join(ASSETS, "audio"))
        self.sfx = self.audio.sfx  # BGM + hit sounds share one device stream
        self.chart = None
        self.current_chart_path = ""
        self.current_audio_path = ""
        self.duration = 0.0
        self.intro_start = 0.0
        self.outro_start = None
        self._intro_audio_pending = False
        self.current = 0.0
        self.playing = False
        self.rate = 1.0
        self._anchor_t = 0.0
        self._anchor_wall = 0.0
        self._updating_slider = False
        self._sfx_cursor = 0.0
        self._dirty = True
        self.entries: list[ChartEntry] = []
        self._tasks: list[dict] = []
        self.worker = None
        self._syncing = False
        self.defaults = {
            "song": "", "artist": "", "illustrator": "", "designer": "", "use_meta": True,
            "bgm": 1.0, "hit": 1.0, "speed": 6.0, "revelation": 1.0,
            "offset": 0.0, "start": 0.0, "duration": 0.0,
            "width": 1080, "height": 1920, "fps": 60, "aspect": "9:16",
            "bitrate": 0, "crf": 16, "hud": "auto",
            "color_mode": "full",
            "out_dir": os.path.join(os.getcwd(), "out"),
            "ffmpeg_dir": "",
            "preset": "veryfast", "encoder": "libx264",
            "intro": True, "outro": True,
            "copyright": "",
            "last_dir": "",
        }
        self._settings_file = ROOT / "settings.json"
        self._factory_defaults = dict(self.defaults)
        self._read_settings()
        self.audio_dir = r"D:\Music\Rizline" if os.path.isdir(r"D:\Music\Rizline") else ""
        self.audio.set_ffmpeg_dir(self.defaults["ffmpeg_dir"])

        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        self.pages = QStackedWidget()
        self.pages.addWidget(content)
        body = QWidget()
        body_l = QHBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(156)
        self.nav.addItems(["渲染", "批量", "任务", "设置"])
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        body_l.addWidget(self.nav)
        body_l.addWidget(self.pages, 1)
        outer.addWidget(body, 1)

        self.queue = QListWidget()
        self.queue.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.queue.currentRowChanged.connect(self._on_queue_row)
        self.queue.itemDoubleClicked.connect(self._edit_entry)

        # Independent preview window (modeless; never blocks the main window).
        self.gl = PreviewGL()
        self.preview_win = QDialog(None)
        self.preview_win.setWindowFlags(Qt.Window)
        self.preview_win.setWindowTitle("预览")
        self.preview_win.setModal(False)
        self.preview_win.resize(560, 960)
        pv = QVBoxLayout(self.preview_win)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(self.gl)
        # Space toggles playback while the detached preview has focus.
        self.preview_win.installEventFilter(self)
        self.gl.installEventFilter(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(scroll, 1)
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 6, 0)
        host_layout.setSpacing(10)
        cols = QHBoxLayout()
        cols.setSpacing(12)
        left = QVBoxLayout()
        left.setSpacing(10)
        right = QVBoxLayout()
        right.setSpacing(10)
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        host_layout.addLayout(cols)
        host_layout.addStretch(1)
        scroll.setWidget(host)

        # left column -------------------------------------------------
        info = QGroupBox("曲目信息")
        il = QFormLayout(info)
        self.song_edit = QLineEdit()
        self.song_edit.setPlaceholderText("曲名")
        self.artist_edit = QLineEdit()
        self.artist_edit.setPlaceholderText("曲师")
        self.illustrator_edit = QLineEdit()
        self.illustrator_edit.setPlaceholderText("插画设计")
        self.designer_edit = QLineEdit()
        self.designer_edit.setPlaceholderText("关卡设计")
        self.cover_edit = QLineEdit()
        self.cover_edit.setPlaceholderText("曲绘文件")
        cover_row = QHBoxLayout()
        cover_pick = QPushButton("浏览…")
        cover_pick.clicked.connect(self._pick_cover)
        cover_row.addWidget(self.cover_edit, 1)
        cover_row.addWidget(cover_pick)
        self.chart_edit = QLineEdit()
        self.chart_edit.setPlaceholderText("谱面 JSON")
        chart_row = QHBoxLayout()
        chart_pick = QPushButton("浏览…")
        chart_pick.clicked.connect(self.pick_chart)
        chart_row.addWidget(self.chart_edit, 1)
        chart_row.addWidget(chart_pick)
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("音频文件")
        audio_row = QHBoxLayout()
        audio_pick = QPushButton("浏览…")
        audio_pick.clicked.connect(self.pick_audio)
        audio_row.addWidget(self.audio_edit, 1)
        audio_row.addWidget(audio_pick)
        il.addRow("曲名", self.song_edit)
        il.addRow("曲师", self.artist_edit)
        il.addRow("插画设计", self.illustrator_edit)
        il.addRow("关卡设计", self.designer_edit)
        il.addRow("谱面文件", chart_row)
        il.addRow("音频文件", audio_row)
        il.addRow("插画", cover_row)
        left.addWidget(info)

        audio_group = QGroupBox("音量 / 流速")
        af = QFormLayout(audio_group)
        self.r_bgm = self._volume(af, "BGM", 100)
        self.r_hit = self._volume(af, "打击音", 100)
        self.r_speed = QDoubleSpinBox()
        self.r_speed.setRange(1.0, 10.0)
        self.r_speed.setSingleStep(0.1)
        self.r_speed.setDecimals(1)
        self.r_speed.setValue(6.0)
        self.r_speed.setSuffix(" ×")
        self.r_rev = QSpinBox()
        self.r_rev.setRange(50, 150)
        self.r_rev.setValue(100)
        self.r_rev.setSuffix(" %")
        self.r_offset = QDoubleSpinBox()
        self.r_offset.setRange(-5000.0, 5000.0)
        self.r_offset.setSingleStep(5.0)
        self.r_offset.setDecimals(0)
        self.r_offset.setValue(0.0)
        self.r_offset.setSuffix(" ms")
        af.addRow("流速", self.r_speed)
        af.addRow("缩放", self.r_rev)
        af.addRow("偏移", self.r_offset)
        left.addWidget(audio_group)

        # right column ------------------------------------------------
        video = QGroupBox("画面")
        vf = QFormLayout(video)
        self.r_ratio = QComboBox()
        self.r_ratio.addItems(["9:16", "16:9", "3:4", "1:1"])
        self.r_w = QSpinBox()
        self.r_w.setRange(240, 4320)
        self.r_w.setSingleStep(2)
        self.r_w.setValue(1080)
        self.r_h = QSpinBox()
        self.r_h.setRange(240, 4320)
        self.r_h.setSingleStep(2)
        self.r_h.setValue(1920)
        self.r_fps = QSpinBox()
        self.r_fps.setRange(30, 120)
        self.r_fps.setValue(60)
        self.r_color = QComboBox()
        self.r_color.addItem("full（4:2:0 全范围）", "full")
        self.r_color.addItem("master（4:4:4 精确颜色）", "master")
        self.r_color.addItem("compat（4:2:0 TV）", "compat")
        vf.addRow("比例", self.r_ratio)
        vf.addRow("宽", self.r_w)
        vf.addRow("高", self.r_h)
        vf.addRow("帧率", self.r_fps)
        vf.addRow("颜色模式", self.r_color)
        right.addWidget(video)

        codec = QGroupBox("编码")
        cf = QFormLayout(codec)
        self.r_crf = QSpinBox()
        self.r_crf.setRange(0, 30)
        self.r_crf.setValue(16)
        self.r_bitrate = QSpinBox()
        self.r_bitrate.setRange(0, 60000)
        self.r_bitrate.setSingleStep(1000)
        self.r_bitrate.setValue(0)
        self.r_bitrate.setSpecialValueText("按 CRF")
        self.r_bitrate.setSuffix(" kbps")
        self.r_encoder = QComboBox()
        self.r_encoder.setEditable(False)
        self.r_encoder.addItems(
            ["libx264", "libx265", "h264_qsv", "hevc_qsv",
             "h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf"]
        )
        self.r_preset = QComboBox()
        self.r_preset.setEditable(False)
        self.r_preset.addItems(
            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"]
        )
        cf.addRow("CRF", self.r_crf)
        cf.addRow("码率", self.r_bitrate)
        cf.addRow("编码器", self.r_encoder)
        cf.addRow("preset", self.r_preset)
        right.addWidget(codec)

        # bottom left: HUD / animation / range ------------------------
        extras = QGroupBox("HUD / 动画 / 范围")
        ef = QFormLayout(extras)
        self.r_hud = QComboBox()
        self.r_hud.addItem("auto", "auto")
        self.r_hud.addItem("play", "full")
        self.r_hud.addItem("关闭", "off")
        self.r_intro = QCheckBox("进入动画")
        self.r_intro.setChecked(True)
        self.r_outro = QCheckBox("退出动画")
        self.r_outro.setChecked(True)
        self.r_start = QDoubleSpinBox()
        self.r_start.setRange(0.0, 9999.0)
        self.r_start.setSingleStep(0.1)
        self.r_start.setDecimals(2)
        self.r_start.setSuffix(" s")
        self.r_duration = QDoubleSpinBox()
        self.r_duration.setRange(0.0, 9999.0)
        self.r_duration.setSingleStep(0.1)
        self.r_duration.setDecimals(2)
        self.r_duration.setSuffix(" s")
        self.r_duration.setSpecialValueText("到结尾")
        ef.addRow("HUD", self.r_hud)
        ef.addRow(self.r_intro)
        ef.addRow(self.r_outro)
        self.r_intro.toggled.connect(self._on_anim_toggle)
        self.r_outro.toggled.connect(self._on_anim_toggle)
        ef.addRow("开始", self.r_start)
        ef.addRow("时长", self.r_duration)
        self.status = QLabel("")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        left.addWidget(extras)
        left.addStretch(1)

        # bottom right: output + playback ----------------------------
        out = QGroupBox("输出")
        of = QFormLayout(out)
        row = QHBoxLayout()
        self.r_outdir = QLineEdit(str(self.defaults["out_dir"]))
        pick = QPushButton("浏览…")
        pick.clicked.connect(lambda: self._pick_outdir(self.r_outdir))
        row.addWidget(self.r_outdir, 1)
        row.addWidget(pick)
        of.addRow("目录", row)
        add_task = QPushButton("加入导出任务")
        add_task.setObjectName("accent")
        add_task.clicked.connect(self.export_current)
        of.addRow(add_task)
        right.addWidget(out)

        transport = QGroupBox("播放")
        tl = QVBoxLayout(transport)
        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.play_btn.setObjectName("accent")
        self.play_btn.clicked.connect(self.toggle_play)
        preview_btn = QPushButton("预览窗口")
        preview_btn.clicked.connect(self.open_preview)
        btn_row.addWidget(self.play_btn, 2)
        btn_row.addWidget(preview_btn, 1)
        tl.addLayout(btn_row)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 1000)
        self.seek.valueChanged.connect(self._on_seek_changed)
        tl.addWidget(self.seek)
        info_row = QHBoxLayout()
        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setObjectName("time")
        info_row.addWidget(self.time_label)
        info_row.addWidget(QLabel("倍速"))
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.25, 2.0)
        self.rate_spin.setSingleStep(0.25)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setValue(1.0)
        self.rate_spin.setSuffix(" ×")
        self.rate_spin.valueChanged.connect(self._on_rate)
        info_row.addWidget(self.rate_spin)
        tl.addLayout(info_row)
        tl.addWidget(self.status)
        right.addWidget(transport)
        right.addStretch(1)

        self.song_edit.textChanged.connect(self._on_meta)
        self.artist_edit.textChanged.connect(self._on_meta)
        self.illustrator_edit.textChanged.connect(self._on_meta)
        self.designer_edit.textChanged.connect(self._on_meta)
        self.cover_edit.textChanged.connect(self._on_meta)
        self.r_speed.valueChanged.connect(self._on_speed)
        self.r_rev.valueChanged.connect(self._on_rev)
        self.r_offset.valueChanged.connect(self._on_offset)
        self.r_bgm.valueChanged.connect(self._on_volume)
        self.r_hit.valueChanged.connect(self._on_volume)
        self.r_ratio.currentTextChanged.connect(self._on_ratio)
        self.r_w.valueChanged.connect(self._on_size)
        self.r_h.valueChanged.connect(self._on_size)
        self.r_hud.currentIndexChanged.connect(self._on_hud)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

        self._build_pages()
        self._apply_defaults_to_ui()
        QTimer.singleShot(0, self._load_defaults)

    # ---------------------------------------------------------------- UI

    def _slider(self, layout, label, lo, hi, step, value, callback, scale):
        row = QHBoxLayout()
        name = QLabel(label)
        name.setFixedWidth(58)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setSingleStep(step)
        slider.setPageStep(step * 5)
        slider.setValue(value)
        val = QLabel(f"{value / scale:.2f}")
        val.setFixedWidth(44)
        val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(name)
        row.addWidget(slider, 1)
        row.addWidget(val)
        layout.addLayout(row)

        def changed(v):
            val.setText(f"{v / scale:.2f}")
            callback(v / scale)

        slider.valueChanged.connect(changed)
        return slider, val

    def _volume(self, form: QFormLayout, label: str, percent: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(percent)
        spin.setSuffix("%")
        form.addRow(label, spin)
        return spin

    def _pick_outdir(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", edit.text())
        if path:
            edit.setText(path)

    def _pick_cover(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择曲绘", self._dialog_dir(self.cover_edit.text()),
            "图片 (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self._remember_dir(path)
            self.cover_edit.setText(path)

    def _pick_ffmpeg_dir(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 ffmpeg 目录", edit.text())
        if path:
            edit.setText(path)

    def open_preview(self) -> None:
        self.preview_win.show()
        self.preview_win.raise_()
        self.preview_win.activateWindow()
        self.gl.set_time(self.current)
        self.gl.update()

    @staticmethod
    def _combo_data(combo: QComboBox) -> str:
        data = combo.currentData()
        return str(data) if data is not None else combo.currentText()

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentText(str(value))

    # ------------------------------------------------------------ settings

    def _read_settings(self) -> None:
        try:
            data = json.loads(self._settings_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            if key not in self.defaults:
                continue
            cur = self.defaults[key]
            try:
                if isinstance(cur, bool):
                    self.defaults[key] = bool(value)
                elif isinstance(cur, int):
                    self.defaults[key] = int(value)
                elif isinstance(cur, float):
                    self.defaults[key] = float(value)
                elif isinstance(cur, str):
                    self.defaults[key] = str(value)
            except (TypeError, ValueError):
                pass

    def _write_settings(self) -> bool:
        try:
            self._settings_file.write_text(
                json.dumps(self.defaults, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except OSError as exc:
            self.status.setText(f"写入设置失败: {exc}")
            return False

    def _apply_defaults_to_ui(self) -> None:
        d = self.defaults
        self._syncing = True
        try:
            # settings page
            self.set_bgm.setValue(int(round(d["bgm"] * 100)))
            self.set_hit.setValue(int(round(d["hit"] * 100)))
            self.set_speed.setValue(d["speed"])
            self.set_rev.setValue(int(round(d["revelation"] * 100)))
            self.set_offset.setValue(round(d["offset"] * 1000))
            self.set_intro.setChecked(d["intro"])
            self.set_outro.setChecked(d["outro"])
            self.set_aspect.setCurrentText(d["aspect"])
            self.set_w.setValue(d["width"])
            self.set_h.setValue(d["height"])
            self.set_fps.setValue(d["fps"])
            self._set_combo_data(self.set_hud, d["hud"])
            self._set_combo_data(self.set_color, d["color_mode"])
            self.set_crf.setValue(d["crf"])
            self.set_bitrate.setValue(int(d.get("bitrate") or 0))
            self.set_encoder.setCurrentText(d["encoder"])
            self.set_preset.setCurrentText(d["preset"])
            self.set_start.setValue(d["start"])
            self.set_duration.setValue(d["duration"])
            self.set_outdir.setText(d["out_dir"])
            self.set_ffmpeg_dir.setText(d.get("ffmpeg_dir", ""))
            self.set_copyright.setText(str(d.get("copyright", "")))
            self.audio.set_ffmpeg_dir(d.get("ffmpeg_dir", ""))

            # render page mirrors the same defaults
            self.r_bgm.setValue(int(round(d["bgm"] * 100)))
            self.r_hit.setValue(int(round(d["hit"] * 100)))
            self.r_speed.setValue(d["speed"])
            self.r_rev.setValue(int(round(d["revelation"] * 100)))
            self.r_offset.setValue(round(d["offset"] * 1000))
            self.r_intro.setChecked(d["intro"])
            self.r_outro.setChecked(d["outro"])
            self.r_ratio.setCurrentText(d["aspect"])
            self.r_w.setValue(d["width"])
            self.r_h.setValue(d["height"])
            self.r_fps.setValue(d["fps"])
            self._set_combo_data(self.r_hud, d["hud"])
            self._set_combo_data(self.r_color, d["color_mode"])
            self.r_crf.setValue(d["crf"])
            self.r_bitrate.setValue(int(d.get("bitrate") or 0))
            self.r_encoder.setCurrentText(d["encoder"])
            self.r_preset.setCurrentText(d["preset"])
            self.r_start.setValue(d["start"])
            self.r_duration.setValue(d["duration"])
            self.r_outdir.setText(d["out_dir"])
        finally:
            self._syncing = False
        self._apply_live_settings()

    def _apply_live_settings(self) -> None:
        s = self.gl.settings
        s.speed = self.r_speed.value()
        s.revelation = self.r_rev.value() / 100.0
        s.offset = self.r_offset.value() / 1000.0
        s.copyright_text = self.set_copyright.text()
        s.show_watermark = self._combo_data(self.r_hud) != "off"
        s.hud_mode = self._combo_data(self.r_hud)
        s.song_name = self.song_edit.text() or "Untitled"
        s.artist = self.artist_edit.text()
        s.illustrator = self.illustrator_edit.text()
        s.designer = self.designer_edit.text()
        s.cover_path = self.cover_edit.text().strip()
        s.intro = self.r_intro.isChecked()
        s.outro = self.r_outro.isChecked()
        self.audio.bgm_volume = self.r_bgm.value() / 100.0 * BGM_GAIN
        self.audio.sfx_volume = self.r_hit.value() / 100.0 * SFX_GAIN
        self.gl.aspect = max(0.05, self.r_w.value() / max(1, self.r_h.value()))
        self._recalc_duration()
        s.outro_start = self.outro_start
        self.gl.settings_changed(rebuild_text=True)

    def _render_cfg(self) -> dict:
        return {
            "song": self.song_edit.text().strip(),
            "artist": self.artist_edit.text().strip(),
            "illustrator": self.illustrator_edit.text().strip(),
            "designer": self.designer_edit.text().strip(),
            "use_meta": True,
            "bgm": self.r_bgm.value() / 100.0,
            "hit": self.r_hit.value() / 100.0,
            "speed": self.r_speed.value(),
            "revelation": self.r_rev.value() / 100.0,
            "offset": self.r_offset.value() / 1000.0,
            "copyright": self.set_copyright.text(),
            "lead_in": 0.0,
            "start": self.r_start.value(),
            "duration": self.r_duration.value(),
            "width": self.r_w.value(),
            "height": self.r_h.value(),
            "fps": self.r_fps.value(),
            "bitrate": self.r_bitrate.value(),
            "crf": self.r_crf.value(),
            "hud": self._combo_data(self.r_hud),
            "color_mode": self._combo_data(self.r_color),
            "watermark": self._combo_data(self.r_hud) != "off",
            "intro": self.r_intro.isChecked(),
            "outro": self.r_outro.isChecked(),
            "preset": self.r_preset.currentText().strip() or "veryfast",
            "encoder": self.r_encoder.currentText().strip() or "libx264",
            "out_dir": self.r_outdir.text().strip() or os.path.join(os.getcwd(), "out"),
        }

    def _build_pages(self) -> None:
        batch = QWidget()
        bl = QVBoxLayout(batch)
        bl.addWidget(QLabel("批量导出"))
        row = QHBoxLayout()
        for text, slot in (
            ("文件", self.add_files),
            ("文件夹", self.add_folder),
            ("压缩包", self.add_archive),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        bl.addLayout(row)
        export_all = QPushButton("批量导出…")
        export_all.setObjectName("accent")
        export_all.clicked.connect(self.export_batch)
        bl.addWidget(export_all)
        self.batch_info = QLabel("队列: 0 个谱面")
        self.batch_info.setObjectName("muted")
        bl.addWidget(self.batch_info)
        bl.addWidget(self.queue, 1)
        row2 = QHBoxLayout()
        b_remove = QPushButton("移除")
        b_remove.clicked.connect(self.remove_selected)
        b_clear = QPushButton("清空")
        b_clear.clicked.connect(self.clear_queue)
        row2.addWidget(b_remove)
        row2.addWidget(b_clear)
        bl.addLayout(row2)
        self.pages.addWidget(batch)

        tasks = QWidget()
        tl = QVBoxLayout(tasks)
        tl.addWidget(QLabel("任务"))
        self.task_log = QListWidget()
        self.task_log.setObjectName("tasks")
        # The row draws its own background + inset bar; the selection
        # highlight and focus frame must not paint over it.
        self.task_log.setSelectionMode(QAbstractItemView.NoSelection)
        self.task_log.setFocusPolicy(Qt.NoFocus)
        self.task_log.addItem("导出任务会显示在这里")
        self._task_placeholder = True
        tl.addWidget(self.task_log, 1)
        self.pages.addWidget(tasks)

        settings = QWidget()
        sl = QVBoxLayout(settings)
        sl.setContentsMargins(14, 14, 14, 14)
        sl.setSpacing(10)

        sscroll = QScrollArea()
        sscroll.setWidgetResizable(True)
        sscroll.setFrameShape(QFrame.NoFrame)
        sscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sl.addWidget(sscroll, 1)
        shost = QWidget()
        shost.setObjectName("panel")
        shost_layout = QVBoxLayout(shost)
        shost_layout.setContentsMargins(0, 0, 4, 0)
        shost_layout.setSpacing(10)
        scols = QHBoxLayout()
        scols.setSpacing(12)
        sleft = QVBoxLayout()
        sleft.setSpacing(10)
        sright = QVBoxLayout()
        sright.setSpacing(10)
        scols.addLayout(sleft, 1)
        scols.addLayout(sright, 1)
        shost_layout.addLayout(scols)
        shost_layout.addStretch(1)
        sscroll.setWidget(shost)

        s_audio = QGroupBox("音量 / 流速")
        saf = QFormLayout(s_audio)
        self.set_bgm = self._volume(saf, "BGM", 100)
        self.set_hit = self._volume(saf, "打击音", 100)
        self.set_speed = QDoubleSpinBox()
        self.set_speed.setRange(1.0, 10.0)
        self.set_speed.setSingleStep(0.1)
        self.set_speed.setDecimals(1)
        self.set_speed.setValue(6.0)
        self.set_speed.setSuffix(" ×")
        self.set_rev = QSpinBox()
        self.set_rev.setRange(50, 150)
        self.set_rev.setValue(100)
        self.set_rev.setSuffix(" %")
        self.set_offset = QDoubleSpinBox()
        self.set_offset.setRange(-5000.0, 5000.0)
        self.set_offset.setSingleStep(5.0)
        self.set_offset.setDecimals(0)
        self.set_offset.setSuffix(" ms")
        saf.addRow("流速", self.set_speed)
        saf.addRow("缩放", self.set_rev)
        saf.addRow("偏移", self.set_offset)
        sleft.addWidget(s_audio)

        s_video = QGroupBox("画面")
        svf = QFormLayout(s_video)
        self.set_aspect = QComboBox()
        self.set_aspect.addItems(["9:16", "16:9", "3:4", "1:1"])
        self.set_w = QSpinBox()
        self.set_w.setRange(240, 4320)
        self.set_w.setSingleStep(2)
        self.set_w.setValue(1080)
        self.set_h = QSpinBox()
        self.set_h.setRange(240, 4320)
        self.set_h.setSingleStep(2)
        self.set_h.setValue(1920)
        self.set_fps = QSpinBox()
        self.set_fps.setRange(30, 120)
        self.set_fps.setValue(60)
        self.set_color = QComboBox()
        self.set_color.addItem("full（4:2:0 全范围）", "full")
        self.set_color.addItem("master（4:4:4 精确颜色）", "master")
        self.set_color.addItem("compat（4:2:0 TV）", "compat")
        svf.addRow("比例", self.set_aspect)
        svf.addRow("宽", self.set_w)
        svf.addRow("高", self.set_h)
        svf.addRow("帧率", self.set_fps)
        svf.addRow("颜色模式", self.set_color)
        sright.addWidget(s_video)

        s_codec = QGroupBox("编码")
        scf = QFormLayout(s_codec)
        self.set_crf = QSpinBox()
        self.set_crf.setRange(0, 30)
        self.set_crf.setValue(16)
        self.set_bitrate = QSpinBox()
        self.set_bitrate.setRange(0, 60000)
        self.set_bitrate.setSingleStep(1000)
        self.set_bitrate.setValue(0)
        self.set_bitrate.setSpecialValueText("按 CRF")
        self.set_bitrate.setSuffix(" kbps")
        self.set_encoder = QComboBox()
        self.set_encoder.setEditable(False)
        self.set_encoder.addItems(
            ["libx264", "libx265", "h264_qsv", "hevc_qsv",
             "h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf"]
        )
        self.set_preset = QComboBox()
        self.set_preset.setEditable(False)
        self.set_preset.addItems(
            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"]
        )
        scf.addRow("CRF", self.set_crf)
        scf.addRow("码率", self.set_bitrate)
        scf.addRow("编码器", self.set_encoder)
        scf.addRow("preset", self.set_preset)
        sright.addWidget(s_codec)

        s_extras = QGroupBox("HUD / 动画 / 范围")
        sef = QFormLayout(s_extras)
        self.set_hud = QComboBox()
        self.set_hud.addItem("auto", "auto")
        self.set_hud.addItem("play", "full")
        self.set_hud.addItem("关闭", "off")
        self.set_intro = QCheckBox("进入动画")
        self.set_intro.setChecked(True)
        self.set_outro = QCheckBox("退出动画")
        self.set_outro.setChecked(True)
        self.set_start = QDoubleSpinBox()
        self.set_start.setRange(0.0, 9999.0)
        self.set_start.setSingleStep(0.1)
        self.set_start.setDecimals(2)
        self.set_start.setSuffix(" s")
        self.set_duration = QDoubleSpinBox()
        self.set_duration.setRange(0.0, 9999.0)
        self.set_duration.setSingleStep(0.1)
        self.set_duration.setDecimals(2)
        self.set_duration.setSuffix(" s")
        self.set_duration.setSpecialValueText("到结尾")
        sef.addRow("HUD", self.set_hud)
        sef.addRow(self.set_intro)
        sef.addRow(self.set_outro)
        sef.addRow("开始", self.set_start)
        sef.addRow("时长", self.set_duration)
        sleft.addWidget(s_extras)
        sleft.addStretch(1)

        s_out = QGroupBox("输出")
        sof = QFormLayout(s_out)
        orow = QHBoxLayout()
        self.set_outdir = QLineEdit(str(self.defaults["out_dir"]))
        opick = QPushButton("浏览…")
        opick.clicked.connect(lambda: self._pick_outdir(self.set_outdir))
        orow.addWidget(self.set_outdir, 1)
        orow.addWidget(opick)
        sof.addRow("目录", orow)
        sright.addWidget(s_out)
        sright.addStretch(1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2a2f3a; background: #2a2f3a; max-height: 1px;")
        shost_layout.addWidget(sep)
        other = QGroupBox("其他设置")
        oth = QFormLayout(other)
        self.set_ffmpeg_dir = QLineEdit(str(self.defaults["ffmpeg_dir"]))
        self.set_ffmpeg_dir.setPlaceholderText("留空则使用系统 PATH 中的 ffmpeg / ffplay / ffprobe")
        frow = QHBoxLayout()
        fpick = QPushButton("浏览…")
        fpick.clicked.connect(lambda: self._pick_ffmpeg_dir(self.set_ffmpeg_dir))
        frow.addWidget(self.set_ffmpeg_dir, 1)
        frow.addWidget(fpick)
        oth.addRow("ffmpeg 目录", frow)
        fhint = QLabel("目录中需包含 ffmpeg.exe、ffplay.exe、ffprobe.exe；预览与导出共用。")
        fhint.setObjectName("muted")
        fhint.setWordWrap(True)
        oth.addRow(fhint)
        # Copyright mark shown at the bottom-right of the playfield.  No
        # placeholder/hint on purpose: only the person who knows fills it in.
        self.set_copyright = QLineEdit(str(self.defaults.get("copyright", "")))
        oth.addRow("不懂别填", self.set_copyright)
        self.set_copyright.textChanged.connect(self._on_copyright)
        shost_layout.addWidget(other)

        buttons = QHBoxLayout()
        save = QPushButton("保存默认设置")
        save.setObjectName("accent")
        save.clicked.connect(self._save_defaults)
        reset = QPushButton("恢复默认")
        reset.clicked.connect(self._reset_defaults)
        buttons.addWidget(save)
        buttons.addWidget(reset)
        shost_layout.addLayout(buttons)
        self.set_aspect.currentTextChanged.connect(self._on_set_aspect)
        self.pages.addWidget(settings)

    def _load_defaults(self) -> None:
        # Start empty: no sample chart/audio is auto-loaded.
        self._apply_live_settings()
        self._dirty = True

    def _dialog_dir(self, current: str = "") -> str:
        """Start directory for a file dialog: current field, then last used."""
        if current:
            folder = current if os.path.isdir(current) else os.path.dirname(current)
            if folder and os.path.isdir(folder):
                return folder
        last = self.defaults.get("last_dir", "")
        if last and os.path.isdir(last):
            return last
        return os.path.expanduser("~")

    def _remember_dir(self, path: str) -> None:
        if not path:
            return
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if folder and os.path.isdir(folder):
            self.defaults["last_dir"] = folder

    def pick_chart(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择谱面", self._dialog_dir(self.chart_edit.text()),
            "Rizline 谱面 (*.json);;所有文件 (*)"
        )
        if path:
            self._remember_dir(path)
            self.load_chart(path)

    def pick_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频", self._dialog_dir(self.audio_edit.text()),
            "音频 (*.mp3 *.ogg *.wav *.flac);;所有文件 (*)"
        )
        if path:
            self._remember_dir(path)
            self.load_audio(path)
            row = self.queue.currentRow()
            if 0 <= row < len(self.entries):
                self.entries[row].audio_path = path

    # -------------------------------------------------------- library

    def pick_audio_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择音频目录", self.audio_dir or str(ROOT))
        if path:
            self.audio_dir = path

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择谱面/音频/曲绘", self._dialog_dir(self.chart_edit.text()),
            "谱面/音频/曲绘 (*.json *.wav *.flac *.mp3 *.png *.jpg *.jpeg);;所有文件 (*)"
        )
        if paths:
            self._remember_dir(paths[0])
        self._add_paths(paths)

    def add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择谱面文件夹", self._dialog_dir(self.chart_edit.text())
        )
        if path:
            self._remember_dir(path)
            self._add_paths([path])

    def add_archive(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择压缩包", self._dialog_dir(self.chart_edit.text()), "ZIP (*.zip)"
        )
        if path:
            self._remember_dir(path)
            self._add_paths([path])

    def _add_paths(self, paths) -> None:
        if not paths:
            return
        audio_exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
        image_exts = (".png", ".jpg", ".jpeg", ".webp")
        row = self.queue.currentRow()
        charts = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in audio_exts:
                if 0 <= row < len(self.entries):
                    self.entries[row].audio_path = path
                    self.load_audio(path)
            elif ext in image_exts:
                if 0 <= row < len(self.entries):
                    self.entries[row].cover_path = path
                    self.status.setText("已设置曲绘")
            else:
                charts.append(path)
        if not charts:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            entries = import_paths(list(charts), self.audio_dir)
        finally:
            QApplication.restoreOverrideCursor()
        if not entries:
            self.status.setText("没有找到可用谱面（需要包含 bPM/lines 的 JSON）")
            return
        for entry in entries:
            self.entries.append(entry)
            self.queue.addItem(entry.label)
        self.queue.setCurrentRow(len(self.entries) - 1)
        missing = sum(1 for entry in entries if not entry.audio_path)
        msg = f"已导入 {len(entries)} 个谱面"
        if missing:
            msg += f"，{missing} 个未匹配到音频"
        self.status.setText(msg)
        self.batch_info.setText(f"队列: {len(self.entries)} 个谱面")

    def _on_queue_row(self, row: int) -> None:
        if not (0 <= row < len(self.entries)):
            return
        entry = self.entries[row]
        self.load_chart(entry.chart_path)
        if entry.audio_path and os.path.isfile(entry.audio_path):
            self.load_audio(entry.audio_path)
        if entry.song:
            self.song_edit.setText(entry.song)
        if entry.artist:
            self.artist_edit.setText(entry.artist)
        self.illustrator_edit.setText(entry.illustrator)
        self.designer_edit.setText(entry.designer)
        self.cover_edit.setText(entry.cover_path or match_cover(entry.chart_path))

    def remove_selected(self) -> None:
        rows = sorted({self.queue.row(item) for item in self.queue.selectedItems()}, reverse=True)
        for row in rows:
            if 0 <= row < len(self.entries):
                self.queue.takeItem(row)
                self.entries.pop(row)
        if self.entries and self.queue.currentRow() < 0:
            self.queue.setCurrentRow(0)

    def clear_queue(self) -> None:
        self.queue.clear()
        self.entries.clear()
        self._pause()
        self.chart = None
        self.gl.chart = None
        self.gl.renderer = None
        self.gl.update()

    def _save_defaults(self) -> None:
        self.defaults.update({
            "bgm": self.set_bgm.value() / 100.0,
            "hit": self.set_hit.value() / 100.0,
            "color_mode": self._combo_data(self.set_color),
            "width": self.set_w.value(),
            "height": self.set_h.value(),
            "aspect": self.set_aspect.currentText(),
            "fps": self.set_fps.value(),
            "hud": self._combo_data(self.set_hud),
            "speed": self.set_speed.value(),
            "revelation": self.set_rev.value() / 100.0,
            "offset": self.set_offset.value() / 1000.0,
            "intro": self.set_intro.isChecked(),
            "outro": self.set_outro.isChecked(),
            "start": self.set_start.value(),
            "duration": self.set_duration.value(),
            "bitrate": self.set_bitrate.value(),
            "crf": self.set_crf.value(),
            "preset": self.set_preset.currentText().strip() or "veryfast",
            "encoder": self.set_encoder.currentText().strip() or "libx264",
            "out_dir": self.set_outdir.text().strip() or os.path.join(os.getcwd(), "out"),
            "ffmpeg_dir": self.set_ffmpeg_dir.text().strip(),
            "copyright": self.set_copyright.text(),
        })
        if self._write_settings():
            self._apply_defaults_to_ui()
            self.status.setText("默认导出设置已保存到 settings.json")

    def _reset_defaults(self) -> None:
        self.defaults = dict(self._factory_defaults)
        if self._write_settings():
            self._apply_defaults_to_ui()
            self.status.setText("已恢复默认设置")

    def _edit_entry(self, item) -> None:
        row = self.queue.row(item)
        if not (0 <= row < len(self.entries)):
            return
        entry = self.entries[row]
        if getattr(self, "_entry_dialog", None) is not None:
            self._entry_dialog.close()
            self._entry_dialog = None
        dlg = ExportDialog([entry], None, self.defaults)
        dlg.setModal(False)
        dlg.accepted.connect(lambda d=dlg, e=entry: self._save_entry_settings(d, e))
        dlg.show()
        self._entry_dialog = dlg

    def _save_entry_settings(self, dlg, entry) -> None:
        entry.settings = dlg._config()
        self.status.setText("已保存该谱面的独立设置")
        if getattr(self, "_entry_dialog", None) is dlg:
            self._entry_dialog = None

    def export_current(self) -> None:
        """Queue the chart shown on the render page, without touching batch."""
        if self.chart is None or not self.current_chart_path:
            self.status.setText("先选择谱面再导出")
            return
        entry = ChartEntry(
            chart_path=self.current_chart_path,
            audio_path=self.current_audio_path,
            song=self.song_edit.text().strip(),
            artist=self.artist_edit.text().strip(),
            illustrator=self.illustrator_edit.text().strip(),
            designer=self.designer_edit.text().strip(),
            cover_path=self.cover_edit.text().strip(),
        )
        self._start_export([entry])

    def export_batch(self) -> None:
        selected = [self.entries[self.queue.row(item)] for item in self.queue.selectedItems()]
        entries = selected or self.entries
        if not entries:
            self.status.setText("先导入谱面再导出")
            return
        self._start_export(entries)

    def _start_export(self, entries) -> None:
        if getattr(self, "_task_placeholder", False):
            self.task_log.clear()
            self._task_placeholder = False
        cfg = dict(self.defaults)
        cfg.update(self._render_cfg())
        os.makedirs(cfg["out_dir"], exist_ok=True)
        self.nav.setCurrentRow(2)
        running = self.worker is not None and self.worker.isRunning()
        if running:
            # Append to the export already in flight.  Starting a second
            # worker used to leave the first row's progress frozen (the new
            # worker replaced `_tasks`) and could hang (two GLFW/ffmpeg
            # contexts in two threads).
            self._tasks.extend(self._add_task(entry) for entry in entries)
            self.worker.add(entries, cfg)
            self.status.setText("已加入当前导出队列")
            return
        self._tasks = [self._add_task(entry) for entry in entries]
        self.worker = ExportWorker(entries, cfg)
        self.worker.file_done.connect(self._task_file_done)
        self.worker.done.connect(self._tasks_done)
        self.worker.progress.connect(self._task_progress)
        self.worker.start()

    def _add_task(self, entry) -> dict:
        item = QListWidgetItem()
        self.task_log.addItem(item)
        row = TaskRow(entry.label)
        h = row.layout()
        label = row.label
        task = {"label": label, "row": row, "item": item, "deleted": False}
        delete = QPushButton("删除")
        delete.setFixedWidth(64)
        delete.clicked.connect(lambda _=False, t=task: self._remove_task(t))
        play = QPushButton("播放")
        folder = QPushButton("文件夹")
        for btn in (play, folder):
            btn.setEnabled(False)
            btn.setFixedWidth(64)
        task["play"] = play
        task["folder"] = folder
        h.addWidget(delete)
        h.addWidget(play)
        h.addWidget(folder)
        self.task_log.setItemWidget(item, row)
        return task

    def _remove_task(self, task: dict) -> None:
        task["deleted"] = True
        row = self.task_log.row(task["item"])
        if row >= 0:
            self.task_log.takeItem(row)

    def _task_progress(self, index: int, msg: str, frac: float = -1.0) -> None:
        if not (0 <= index < len(self._tasks)):
            return
        task = self._tasks[index]
        if task.get("deleted"):
            return
        total = max(1, len(self._tasks))
        task["label"].setText(f"[{index + 1}/{total}] {msg}")
        if frac >= 0.0:
            # Keep the highest value seen for this file: messages without a
            # fraction (mixing, ffmpeg setup) must not pull the bar backwards.
            task["frac"] = max(float(task.get("frac", 0.0)), min(1.0, frac))
        row = task.get("row")
        if row is not None:
            done = index + float(task.get("frac", 0.0))
            row.set_progress(done / total)

    def _task_file_done(self, index: int, path: str) -> None:
        if not (0 <= index < len(self._tasks)):
            return
        task = self._tasks[index]
        if task.get("deleted"):
            return
        task["label"].setText("完成 · " + os.path.basename(path))
        task["frac"] = 1.0
        if task.get("row") is not None:
            task["row"].set_progress(1.0)
        task["play"].setEnabled(True)
        task["folder"].setEnabled(True)
        task["play"].clicked.connect(lambda _=False, p=path: os.startfile(p))
        task["folder"].clicked.connect(lambda _=False, p=path: os.startfile(os.path.dirname(p)))

    def _tasks_done(self, ok: bool, msg: str) -> None:
        self.status.setText(msg)
        # Let the next "导出" start a fresh worker instead of appending to a
        # thread that already finished.
        self.worker = None

    def load_chart(self, path: str) -> None:
        try:
            chart = load_chart(path)
        except Exception as exc:
            self.status.setText(f"谱面加载失败: {exc}")
            return
        self.chart = chart
        self.current_chart_path = path
        self.gl.set_chart(chart)
        self.chart_edit.setText(path)
        if not self.song_edit.text():
            self.song_edit.setText(chart.song_name or "")
        if not self.cover_edit.text().strip():
            self.cover_edit.setText(match_cover(path))
        self._recalc_duration()
        self.seek.setValue(0)
        self.current = 0.0
        self._sfx_cursor = self._chart_time(0.0)
        self.gl.set_time(0.0)
        self.status.setText(f"谱面 {chart.note_count} 音符 / {chart.duration:.1f}s")
        self._dirty = True

    def load_audio(self, path: str) -> None:
        was_playing = self.playing
        if was_playing:
            self._pause()
        # Decoding a whole track takes ~1 s; do not look frozen meanwhile.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.status.setText("正在读取音频…")
            ok = self.audio.load(path)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            self.status.setText(self.audio.error)
            return
        self.audio_edit.setText(path)
        self.current_audio_path = path
        self._recalc_duration()
        self.current = 0.0
        self.audio.seek(0.0)
        self.gl.set_time(0.0)
        self.seek.setValue(0)
        self._dirty = True
        if was_playing:
            self._play()

    def _recalc_duration(self) -> None:
        chart_len = 0.0
        if self.chart is not None:
            chart_len = self.chart.duration + max(0.0, self.gl.settings.offset)
        intro_on = bool(self.r_intro.isChecked()) if hasattr(self, "r_intro") else False
        outro_on = bool(self.r_outro.isChecked()) if hasattr(self, "r_outro") else False
        self.intro_start = -watch.INTRO_PREROLL if intro_on else 0.0
        end = max(chart_len, self.audio.duration)
        self.outro_start = None
        if outro_on:
            base = self.audio.duration if self.audio.duration > 0 else chart_len
            if base > 0:
                # Same rule as the exporter / RizPlayer: 0.1 s before the audio
                # ends, but never before the last note (hold tail) is over.
                notes_end = self.chart.notes_end if self.chart is not None else 0.0
                self.outro_start = max(0.0, base - watch.OUTRO_LEAD, notes_end)
                # Video/preview ends when the card lands (same as the export).
                end = self.outro_start + watch.OUTRO_TOTAL
        self.duration = end
        lo = int(round(self.intro_start * 1000))
        self.seek.setRange(lo, max(lo + 1, int(self.duration * 1000)))
        if getattr(self, "gl", None) is not None:
            # Keep the renderer in sync: load_chart/load_audio call this method
            # without going through _apply_live_settings.
            self.gl.settings.outro_start = self.outro_start

    # ------------------------------------------------------------ actions

    def _on_speed(self, value: float) -> None:
        if self._syncing:
            return
        self.gl.settings.speed = value
        self.gl.settings_changed()
        self._dirty = True

    def _on_rev(self, value: int) -> None:
        if self._syncing:
            return
        self.gl.settings.revelation = value / 100.0
        self.gl.settings_changed()
        self._dirty = True

    def _on_ratio(self, text: str) -> None:
        if self._syncing:
            return
        sizes = {"9:16": (1080, 1920), "16:9": (1920, 1080), "3:4": (1080, 1440), "1:1": (1080, 1080)}
        size = sizes.get(text)
        if size:
            self._syncing = True
            try:
                self.r_w.setValue(size[0])
                self.r_h.setValue(size[1])
            finally:
                self._syncing = False
        self._on_size()

    def _on_set_aspect(self, text: str) -> None:
        if self._syncing:
            return
        sizes = {"9:16": (1080, 1920), "16:9": (1920, 1080), "3:4": (1080, 1440), "1:1": (1080, 1080)}
        size = sizes.get(text)
        if size:
            self._syncing = True
            try:
                self.set_w.setValue(size[0])
                self.set_h.setValue(size[1])
            finally:
                self._syncing = False

    def _on_hud(self, *_args) -> None:
        if self._syncing:
            return
        self.gl.settings.hud_mode = self._combo_data(self.r_hud)
        self.gl.settings.show_watermark = self._combo_data(self.r_hud) != "off"
        self.gl.update()
        self._dirty = True

    def _on_size(self, *_args) -> None:
        if self._syncing:
            return
        self.gl.aspect = max(0.05, self.r_w.value() / max(1, self.r_h.value()))
        self.gl.update()
        self._dirty = True

    def _on_offset(self, value: float) -> None:
        if self._syncing:
            return
        self.gl.settings.offset = value / 1000.0
        self._recalc_duration()
        self.gl.settings_changed()
        self._dirty = True

    def _on_volume(self, *_args) -> None:
        if self._syncing:
            return
        self.audio.bgm_volume = self.r_bgm.value() / 100.0 * BGM_GAIN
        self.audio.sfx_volume = self.r_hit.value() / 100.0 * SFX_GAIN

    def _on_rate(self, value: float) -> None:
        if self.playing and self._anchor_wall:
            self._anchor_t = self._current_wall()
            self._anchor_wall = time.monotonic()
        self.rate = max(0.25, min(2.0, value))
        self.audio.set_rate(self.rate)
        self.sfx.set_rate(self.rate)
        self._dirty = True

    def _on_anim_toggle(self, _checked: bool = False) -> None:
        if self._syncing:
            return
        self._apply_live_settings()
        self._dirty = True

    def _on_copyright(self, text: str) -> None:
        self.defaults["copyright"] = text
        if self._syncing:
            return
        self.gl.settings.copyright_text = text
        self.gl.settings_changed(rebuild_text=True)
        self._dirty = True

    def _on_meta(self) -> None:
        self.gl.settings.song_name = self.song_edit.text() or "Untitled"
        self.gl.settings.artist = self.artist_edit.text()
        self.gl.settings.illustrator = self.illustrator_edit.text()
        self.gl.settings.designer = self.designer_edit.text()
        self.gl.settings.cover_path = self.cover_edit.text().strip()
        self.gl.settings_changed(rebuild_text=True)
        self._dirty = True

    def _on_seek_changed(self, value: int) -> None:
        if self._updating_slider:
            return
        self._seek_to(value / 1000.0)
        self._dirty = True

    def _seek_to(self, t: float) -> None:
        t = max(self.intro_start, min(self.duration, t))
        self.current = t
        if t < 0.0:
            # Still inside the watch card: keep the music parked at 0.
            self.audio.pause()
            self._intro_audio_pending = self.playing
        else:
            self.audio.seek(t)
            self._intro_audio_pending = False
        self.sfx.stop_all()
        self._sfx_cursor = self._chart_time(t)
        self._anchor_t = t
        self._anchor_wall = time.monotonic()
        self.gl.set_time(t)
        self.time_label.setText(f"{_fmt_time(t)} / {_fmt_time(self.duration)}")

    def toggle_play(self) -> None:
        if self.playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if self.chart is None:
            self.status.setText("先加载谱面")
            return
        self.open_preview()
        if self.current >= self.duration - 0.01:
            self._seek_to(self.intro_start)
        elif self.intro_start < 0.0 and abs(self.current) < 1e-6:
            # A fresh play-through from 0 includes the watch card.
            self._seek_to(self.intro_start)
        self.audio.set_rate(self.rate)
        self.sfx.set_rate(self.rate)
        if self.current >= 0.0:
            self.audio.play(self.current)
            self._intro_audio_pending = False
        else:
            # The watch card runs before chart time 0; the music starts when
            # the clock crosses zero (same timeline as the video export).
            self._intro_audio_pending = True
        self._anchor_t = self.current
        self._anchor_wall = time.monotonic()
        self._sfx_cursor = self._chart_time(self.current)
        self.playing = True
        self.play_btn.setText("暂停")
        self._dirty = True

    def _pause(self) -> None:
        if self.playing:
            if self.audio.playing:
                self.current = self.audio.position
            else:
                self.current = self._current_wall()
        self.audio.pause()
        self._intro_audio_pending = False
        self.sfx.stop_all()
        self.playing = False
        self.play_btn.setText("播放")
        self._dirty = True

    def _chart_time(self, t: float) -> float:
        if self.chart is None:
            return 0.0
        return max(0.0, t - self.chart.chart_delay - self.gl.settings.offset)

    def _trigger_sfx(self, t: float) -> None:
        """Queue hit sounds ahead of time on the BGM timeline.

        ``_sfx_cursor`` is the chart time already *queued*, i.e. it runs up to
        ``SFX_LOOKAHEAD`` ahead of the play position.  Comparing it with the
        position needs that offset (a plain ``now - cursor`` check sees the
        cursor "in the future" on every tick and re-queues the same notes
        dozens of times - that produced a blast of hit sounds).
        """
        if self.chart is None or not self.playing or t < 0.0:
            return
        look = SFX_LOOKAHEAD
        now = self._chart_time(t)
        prev = self._sfx_cursor
        if now < prev - look - 0.05 or now > prev + 0.5:
            # Seek/pause jump (or a long stall): drop queued sounds instead of
            # firing a burst of missed notes.
            self._sfx_cursor = now
            self.audio.clear_schedule_before(now + self.chart.chart_delay + self.gl.settings.offset)
            return
        start = max(prev, now)
        horizon = now + look
        if horizon <= start + 1e-6:
            return
        starts = self.chart.note_starts
        lo = bisect_right(starts, start)
        hi = bisect_right(starts, horizon)
        if hi > lo:
            shift = self.chart.chart_delay + self.gl.settings.offset
            for note in self.chart.notes[lo:hi]:
                self.audio.schedule_sfx(
                    "drag" if note.kind == NOTE_DRAG else "hit",
                    note.start_sec + shift,
                )
        self._sfx_cursor = horizon

    def _current_wall(self) -> float:
        if not self._anchor_wall:
            return self.current
        return self._anchor_t + (time.monotonic() - self._anchor_wall) * self.rate

    # --------------------------------------------------------------- tick

    def _tick(self) -> None:
        if self.playing:
            if self._intro_audio_pending:
                # The card runs on the wall clock; the BGM starts at chart 0
                # (AudioPlayer starts `latency` into the track so it is audible
                # immediately and in sync from frame one).
                t = self._current_wall()
                if t >= 0.0:
                    self.audio.play(0.0)
                    self._intro_audio_pending = False
                    self._anchor_t = 0.0
                    self._anchor_wall = time.monotonic()
                    self._sfx_cursor = self._chart_time(0.0)
                    t = 0.0
            elif self.audio.playing:
                t = self.audio.position
            else:
                t = self._current_wall()
            if t >= self.duration:
                t = self.duration
                self._pause()
            self.current = t
            self.gl.set_time(t)
            self._trigger_sfx(t)
            self._updating_slider = True
            self.seek.setValue(int(t * 1000))
            self._updating_slider = False
            self.time_label.setText(f"{_fmt_time(t)} / {_fmt_time(self.duration)}")
            self._dirty = True
        if self._dirty:
            self.gl.update()
            self._dirty = False
            if self.gl.renderer is not None:
                buf_ms = self.audio.latency * 1000.0 if self.audio.duration > 0 else 0.0
                self.status.setText(
                    f"{self.gl.last_frame_ms:.1f} ms/帧 · {self.gl.width()}×{self.gl.height()} 显示"
                    + (f" · 输出缓冲 {buf_ms:.0f} ms" if buf_ms > 0 else "")
                )

    # ------------------------------------------------------------ events

    def eventFilter(self, obj, event) -> bool:
        if obj is self.preview_win and event.type() == QEvent.Close:
            # Closing the detached preview stops playback instead of leaving
            # the audio running headless.
            if self.playing:
                self._pause()
        if event.type() == QEvent.KeyPress and obj in (self.preview_win, self.gl):
            if event.key() == Qt.Key_Space:
                self.toggle_play()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self.toggle_play()
            event.accept()
            return
        if event.key() == Qt.Key_Left:
            self._seek_to(self.current - 5.0)
            self._dirty = True
            event.accept()
            return
        if event.key() == Qt.Key_Right:
            self._seek_to(self.current + 5.0)
            self._dirty = True
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.preview_win.close()
        if getattr(self, "_entry_dialog", None) is not None:
            self._entry_dialog.close()
            self._entry_dialog = None
        self.timer.stop()
        self._write_settings()
        self.audio.close()
        self.sfx.close()
        super().closeEvent(event)
