"""Playback check: the preview must run the watch card, then start the BGM."""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QSurfaceFormat  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rizrec.app.preview import PreviewWindow  # noqa: E402


def main() -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    win = PreviewWindow()
    win.show()
    win.load_chart(os.path.join(ROOT, "assets", "LAIGN_IN.json"))
    win.load_audio(os.path.join(ROOT, "assets", "LAIGN.mp3"))
    win._apply_live_settings()
    print("intro_start", win.intro_start, "outro_start", win.outro_start,
          "duration", win.duration, flush=True)

    state = {"t0": 0.0, "started": False}

    def tick() -> None:
        now = time.monotonic()
        if not state["started"]:
            state["started"] = True
            state["t0"] = now
            win._seek_to(win.intro_start)
            win._play()
        print(f"{now - state['t0']:5.2f}s current={win.current:7.3f} "
              f"audio_playing={win.audio.playing} audio_pos={win.audio.position:7.3f} "
              f"slider={win.seek.value()} label={win.time_label.text()}", flush=True)
        if now - state["t0"] > 7.0:
            win._pause()
            app.quit()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(250)
    app.exec()
    win.audio.close()
    win.sfx.close()


if __name__ == "__main__":
    main()
