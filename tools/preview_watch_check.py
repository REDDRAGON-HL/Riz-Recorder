"""Offscreen check of the watch card through the Qt preview widget."""
from __future__ import annotations

import os
import sys

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
    win.open_preview()
    win.song_edit.setText("LAIGN")
    win.artist_edit.setText("Kiu feat. Nikki Simmons")
    win.cover_edit.setText(os.path.join(ROOT, "assets", "LAIGN_cover.png"))
    win.load_chart(os.path.join(ROOT, "assets", "LAIGN_IN.json"))

    shots = [-4.15, -3.15, -2.6, 141.95, 142.5, 143.2]

    def run() -> None:
        for t in shots:
            win.gl.set_time(t)
            win.gl.update()
            win.gl.repaint()
            img = win.gl.grabFramebuffer()
            path = os.path.join(ROOT, ".tmp_frames", f"qt_watch_{int(round(t * 1000)):06d}.png")
            img.save(path)
            print("saved", path, img.size(), f"{win.gl.last_frame_ms:.1f}ms")
        app.quit()

    QTimer.singleShot(3500, run)
    app.exec()
    win.audio.close()
    win.sfx.close()


if __name__ == "__main__":
    main()
