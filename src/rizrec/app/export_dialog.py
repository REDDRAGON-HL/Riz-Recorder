"""Export settings dialog and background batch worker."""

from __future__ import annotations

import os
import re
import threading

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.chart import load_chart
from ..core.export import export_video
from .library import ChartEntry, difficulty_of

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "assets")

ASPECTS = {
    "9:16（手机竖屏）": (1080, 1920),
    "16:9（横屏）": (1920, 1080),
    "3:4": (1080, 1440),
    "1:1": (1080, 1080),
    "自定义": None,
}


def _safe_name(text: str, fallback: str = "export") -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text).strip(" .")
    return text or fallback


def _set_combo_data(combo: QComboBox, value: str) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return
    combo.setCurrentText(str(value))


def _combo_value(combo: QComboBox) -> str:
    data = combo.currentData()
    return str(data) if data is not None else combo.currentText()


class ExportWorker(QThread):
    # index, message, per-file fraction (0..1, -1 when the message carries none)
    progress = Signal(int, str, float)
    file_done = Signal(int, str)
    done = Signal(bool, str)

    _FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

    def __init__(self, entries: list[ChartEntry], cfg: dict):
        super().__init__()
        # One job per (entry, config): a second "导出" while this thread is
        # still running is appended here instead of starting a second worker.
        # Two workers used to fight over GLFW/ffmpeg (hang) and overwrite the
        # task list, which froze the first row's progress bar.
        self._jobs: list[tuple[ChartEntry, dict]] = [(e, dict(cfg)) for e in entries]
        self._lock = threading.Lock()
        self._cancelled = False

    def add(self, entries: list[ChartEntry], cfg: dict) -> None:
        with self._lock:
            self._jobs.extend((e, dict(cfg)) for e in entries)

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        index = 0
        try:
            while True:
                with self._lock:
                    if index >= len(self._jobs):
                        break
                    entry, cfg = self._jobs[index]
                if self._cancelled:
                    self.done.emit(False, "已取消")
                    return
                # UI settings are authoritative; stale per-entry overrides used
                # to make the export range appear to work only sometimes.
                chart = load_chart(entry.chart_path)
                use_meta = cfg.get("use_meta", True)
                song = entry.song if use_meta else cfg.get("song", "")
                artist = entry.artist if use_meta else cfg.get("artist", "")
                if use_meta and not artist:
                    artist = cfg.get("artist", "")
                if not song:
                    song = chart.song_name or "Untitled"
                diff = difficulty_of(entry.chart_path)
                base = _safe_name(f"{song}_{diff}" if diff else song)
                out_dir = cfg.get("out_dir") or os.path.join(os.getcwd(), "out")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, base + ".mp4")
                stem, ext = os.path.splitext(out_path)
                n = 2
                while os.path.exists(out_path):
                    out_path = f"{stem}_{n}{ext}"
                    n += 1

                def report(msg: str, i=index) -> None:
                    # The frame log is "  done/total  .. fps  ETA ..": turn it
                    # into a per-file fraction so the task bar advances inside
                    # a file instead of only per entry.
                    frac = -1.0
                    found = self._FRACTION_RE.findall(msg)
                    if found:
                        done, count = found[-1]
                        frac = int(done) / max(1, int(count))
                    self.progress.emit(i, msg, frac)

                report(f"导出 {song} …")
                export_video(
                    chart,
                    out_path,
                    ASSETS,
                    audio_path=entry.audio_path or None,
                    width=cfg["width"],
                    height=cfg["height"],
                    fps=cfg["fps"],
                    speed=cfg["speed"],
                    revelation=cfg["revelation"],
                    chart_offset=float(cfg.get("offset", 0.0)),
                    lead_in=0.0,
                    start=float(cfg.get("start", 0.0)),
                    duration=(float(cfg["duration"]) if cfg.get("duration") else None),
                    song=song,
                    artist=artist,
                    bgm_volume=cfg["bgm"],
                    hit_volume=cfg["hit"],
                    crf=int(cfg.get("crf", 16)),
                    bitrate=(int(cfg["bitrate"]) if cfg.get("bitrate") else None),
                    show_watermark=cfg.get("hud", "auto") != "off",
                    color_mode=cfg["color_mode"],
                    preset=cfg.get("preset", "veryfast"),
                    encoder=cfg.get("encoder", "libx264"),
                    ffmpeg_dir=cfg.get("ffmpeg_dir", ""),
                    intro=bool(cfg.get("intro", True)),
                    outro=bool(cfg.get("outro", True)),
                    hud_mode=cfg.get("hud", "auto"),
                    copyright_text=str(cfg.get("copyright", "")),
                    cover_path=entry.cover_path,
                    illustrator=entry.illustrator or cfg.get("illustrator", ""),
                    designer=entry.designer or cfg.get("designer", ""),
                    progress=report,
                )
                self.file_done.emit(index, out_path)
                report(f"完成 {os.path.basename(out_path)}")
                index += 1
            self.done.emit(True, "全部导出完成")
        except Exception as exc:  # pragma: no cover - depends on files/codecs
            self.done.emit(False, str(exc))


class ExportDialog(QDialog):
    def __init__(self, entries: list[ChartEntry], parent: QWidget | None = None,
                 defaults: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("导出设置")
        self.setMinimumWidth(880)
        self.entries = entries
        self.defaults = defaults or {}
        first = entries[0] if entries else None
        d = self.defaults

        root = QVBoxLayout(self)
        root.setSpacing(10)

        columns = QHBoxLayout()
        columns.setSpacing(10)
        left = QVBoxLayout()
        left.setSpacing(10)
        right = QVBoxLayout()
        right.setSpacing(10)
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        root.addLayout(columns)

        meta = QGroupBox("曲目信息")
        form = QFormLayout(meta)
        self.song = QLineEdit(first.song if first else "")
        self.artist = QLineEdit(first.artist if first else "")
        self.illustrator = QLineEdit(first.illustrator if first else "")
        self.illustrator.setPlaceholderText("画师（插画设计）")
        self.designer = QLineEdit(first.designer if first else "")
        self.designer.setPlaceholderText("谱师（关卡设计）")
        form.addRow("曲名", self.song)
        form.addRow("曲师", self.artist)
        form.addRow("画师", self.illustrator)
        form.addRow("关卡设计", self.designer)
        self.chart_path = QLineEdit(first.chart_path if first else "")
        self.audio_path = QLineEdit(first.audio_path if first else "")
        self.cover_path = QLineEdit(first.cover_path if first else "")
        for label, edit, title, flt in (
            ("谱面文件", self.chart_path, "选择谱面", "谱面 (*.json);;所有文件 (*)"),
            ("音频文件", self.audio_path, "选择音频", "音频 (*.wav *.flac *.mp3 *.ogg)"),
            ("曲绘文件", self.cover_path, "选择曲绘", "图片 (*.png *.jpg *.jpeg *.webp)"),
        ):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            btn = QPushButton("浏览…")
            btn.clicked.connect(lambda _=False, e=edit, t=title, f=flt: self._pick_file(e, t, f))
            row.addWidget(btn)
            form.addRow(label, row)
        left.addWidget(meta)

        audio = QGroupBox("音量 / 流速")
        af = QFormLayout(audio)
        self._volumes = {}
        self._volume_input(af, "BGM", int(round(float(d.get("bgm", 1.0)) * 100)))
        self._volume_input(af, "打击音", int(round(float(d.get("hit", 1.0)) * 100)))
        self.speed = QDoubleSpinBox()
        self.speed.setRange(1.0, 10.0)
        self.speed.setSingleStep(0.1)
        self.speed.setDecimals(1)
        self.speed.setValue(float(d.get("speed", 6.0)))
        self.speed.setSuffix(" ×")
        self.revelation = QSpinBox()
        self.revelation.setRange(50, 150)
        self.revelation.setValue(int(round(float(d.get("revelation", 1.0)) * 100)))
        self.revelation.setSuffix(" %")
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-5000.0, 5000.0)
        self.offset.setSingleStep(5.0)
        self.offset.setDecimals(0)
        self.offset.setValue(round(float(d.get("offset", 0.0)) * 1000))
        self.offset.setSuffix(" ms")
        af.addRow("流速", self.speed)
        af.addRow("缩放", self.revelation)
        af.addRow("偏移", self.offset)
        left.addWidget(audio)
        left.addStretch(1)

        video = QGroupBox("画面")
        vf = QFormLayout(video)
        self.aspect = QComboBox()
        self.aspect.addItems(ASPECTS.keys())
        self.width = QSpinBox()
        self.height = QSpinBox()
        for box in (self.width, self.height):
            box.setRange(240, 4320)
            box.setSingleStep(2)
        self.width.setValue(int(d.get("width", 1080)))
        self.height.setValue(int(d.get("height", 1920)))
        self.fps = QSpinBox()
        self.fps.setRange(30, 120)
        self.fps.setValue(int(d.get("fps", 60)))
        vf.addRow("比例", self.aspect)
        vf.addRow("宽", self.width)
        vf.addRow("高", self.height)
        vf.addRow("帧率", self.fps)
        self.color_mode = QComboBox()
        self.color_mode.addItem("full（4:2:0 全范围）", "full")
        self.color_mode.addItem("master（4:4:4 精确颜色）", "master")
        self.color_mode.addItem("compat（4:2:0 TV）", "compat")
        _set_combo_data(self.color_mode, d.get("color_mode", "full"))
        vf.addRow("颜色", self.color_mode)
        matched = None
        for label, size in ASPECTS.items():
            if size == (self.width.value(), self.height.value()):
                matched = label
                break
        self.aspect.setCurrentText(matched or "自定义")
        self.aspect.currentTextChanged.connect(self._apply_aspect)
        right.addWidget(video)

        codec = QGroupBox("编码")
        cf = QFormLayout(codec)
        self.crf = QSpinBox()
        self.crf.setRange(0, 30)
        self.crf.setValue(int(d.get("crf", 16)))
        self.bitrate = QSpinBox()
        self.bitrate.setRange(0, 60000)
        self.bitrate.setSingleStep(1000)
        self.bitrate.setValue(int(d.get("bitrate") or 0))
        self.bitrate.setSpecialValueText("按 CRF")
        self.bitrate.setSuffix(" kbps")
        self.encoder = QComboBox()
        self.encoder.setEditable(False)
        self.encoder.addItems(
            ["libx264", "libx265", "h264_qsv", "hevc_qsv",
             "h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf"]
        )
        self.encoder.setCurrentText(str(d.get("encoder", "libx264")))
        self.preset = QComboBox()
        self.preset.setEditable(False)
        self.preset.addItems(
            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"]
        )
        self.preset.setCurrentText(str(d.get("preset", "veryfast")))
        cf.addRow("CRF", self.crf)
        cf.addRow("码率", self.bitrate)
        cf.addRow("编码器", self.encoder)
        cf.addRow("preset", self.preset)
        right.addWidget(codec)
        right.addStretch(1)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        extras = QGroupBox("HUD / 动画 / 范围")
        ef = QFormLayout(extras)
        self.hud = QComboBox()
        self.hud.addItem("auto", "auto")
        self.hud.addItem("play", "full")
        self.hud.addItem("关闭", "off")
        _set_combo_data(self.hud, d.get("hud", "auto"))
        self.intro = QCheckBox("进入动画")
        self.outro = QCheckBox("退出动画")
        self.intro.setChecked(bool(d.get("intro", True)))
        self.outro.setChecked(bool(d.get("outro", True)))
        self.start = QDoubleSpinBox()
        self.start.setRange(0.0, 9999.0)
        self.start.setSingleStep(0.1)
        self.start.setDecimals(2)
        self.start.setValue(float(d.get("start", 0.0)))
        self.start.setSuffix(" s")
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.0, 9999.0)
        self.duration.setSingleStep(0.1)
        self.duration.setDecimals(2)
        self.duration.setValue(float(d.get("duration", 0.0)))
        self.duration.setSpecialValueText("到结尾")
        self.duration.setSuffix(" s")
        ef.addRow("HUD", self.hud)
        ef.addRow(self.intro)
        ef.addRow(self.outro)
        ef.addRow("开始", self.start)
        ef.addRow("时长", self.duration)
        bottom.addWidget(extras, 1)

        out = QGroupBox("输出")
        of = QFormLayout(out)
        row = QHBoxLayout()
        self.out_dir = QLineEdit(str(d.get("out_dir") or os.path.join(os.getcwd(), "out")))
        pick = QPushButton("选择…")
        pick.clicked.connect(self._pick_dir)
        row.addWidget(self.out_dir, 1)
        row.addWidget(pick)
        of.addRow("目录", row)
        bottom.addWidget(out, 1)
        root.addLayout(bottom)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(
            f"导出 {len(entries)} 个" if len(entries) > 1 else "导出"
        )
        buttons.accepted.connect(self.start_export)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.worker: ExportWorker | None = None

    def _volume_input(self, form: QFormLayout, label: str, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(value)
        spin.setSuffix("%")
        form.addRow(label, spin)
        self._volumes[label] = spin
        return spin

    def _apply_aspect(self, text: str) -> None:
        size = ASPECTS.get(text)
        if size:
            self.width.setValue(size[0])
            self.height.setValue(size[1])

    def _pick_file(self, edit: QLineEdit, title: str, flt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, edit.text(), flt)
        if path:
            edit.setText(path)

    def _pick_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择导出目录", self.out_dir.text())
        if path:
            self.out_dir.setText(path)

    def _config(self) -> dict:
        return {
            "song": self.song.text().strip(),
            "artist": self.artist.text().strip(),
            "illustrator": self.illustrator.text().strip(),
            "designer": self.designer.text().strip(),
            "use_meta": len(self.entries) > 1,
            "bgm": self._volumes["BGM"].value() / 100.0,
            "hit": self._volumes["打击音"].value() / 100.0,
            "speed": self.speed.value(),
            "revelation": self.revelation.value() / 100.0,
            "offset": self.offset.value() / 1000.0,
            "copyright": str(self.defaults.get("copyright", "")),
            "lead_in": 0.0,
            "width": self.width.value(),
            "height": self.height.value(),
            "fps": self.fps.value(),
            "bitrate": self.bitrate.value(),
            "crf": self.crf.value(),
            "hud": _combo_value(self.hud),
            "color_mode": _combo_value(self.color_mode),
            "watermark": True,
            "intro": self.intro.isChecked(),
            "outro": self.outro.isChecked(),
            "start": self.start.value(),
            "duration": self.duration.value(),
            "preset": self.preset.currentText().strip() or "veryfast",
            "encoder": self.encoder.currentText().strip() or "libx264",
            "out_dir": self.out_dir.text().strip() or os.path.join(os.getcwd(), "out"),
            "ffmpeg_dir": self.defaults.get("ffmpeg_dir", ""),
        }

    def start_export(self) -> None:
        if len(self.entries) == 1:
            entry = self.entries[0]
            if self.chart_path.text().strip():
                entry.chart_path = self.chart_path.text().strip()
            entry.audio_path = self.audio_path.text().strip()
            entry.cover_path = self.cover_path.text().strip()
            entry.illustrator = self.illustrator.text().strip()
            entry.designer = self.designer.text().strip()
        os.makedirs(self._config()["out_dir"], exist_ok=True)
        self.setEnabled(False)
        self.worker = ExportWorker(self.entries, self._config())
        progress = QProgressDialog("准备导出…", "取消", 0, len(self.entries), self)
        progress.setWindowTitle("批量导出")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)

        def on_progress(index: int, msg: str, frac: float) -> None:
            progress.setValue(index)
            pct = f" ({frac * 100:.0f}%)" if frac >= 0.0 else ""
            progress.setLabelText(f"[{index + 1}/{len(self.entries)}] {msg}{pct}")

        self.worker.progress.connect(on_progress)
        self.worker.done.connect(lambda ok, msg: self._finish(progress, ok, msg))
        progress.canceled.connect(self.worker.cancel)
        self.worker.start()

    def _finish(self, progress: QProgressDialog, ok: bool, msg: str) -> None:
        progress.setValue(progress.maximum())
        progress.close()
        self.setEnabled(True)
        if ok:
            QMessageBox.information(self, "导出完成", msg)
            self.accept()
        else:
            QMessageBox.critical(self, "导出失败", msg)
