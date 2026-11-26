"""Grab the main window as PNG for each page (UI check)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from rizrec.app.preview import PreviewWindow


def main() -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = PreviewWindow()
    window.resize(1360, 940)
    window.show()
    window.open_preview()
    out = os.path.join(ROOT, ".ref_frames")
    os.makedirs(out, exist_ok=True)

    def grab(name: str) -> None:
        window.repaint()
        window.grab().save(os.path.join(out, name))
        print("saved", name, window.size().width(), window.size().height())

    def step1() -> None:
        window.gl.set_time(39.76)
        window.gl.repaint()
        grab("ui_render_page.png")
        window.preview_win.grab().save(os.path.join(out, "ui_preview_window.png"))
        window.nav.setCurrentRow(3)
        QTimer.singleShot(300, step2)

    def step2() -> None:
        grab("ui_settings_page.png")
        window.nav.setCurrentRow(1)
        QTimer.singleShot(300, step3)

    def step3() -> None:
        grab("ui_batch_page.png")
        app.quit()

    QTimer.singleShot(4000, step1)
    app.exec()
    window.audio.close()
    window.sfx.close()


if __name__ == "__main__":
    main()
