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
    window.show()
    window.open_preview()

    def shot() -> None:
        window.gl.set_time(39.76)
        window.gl.update()
        window.gl.repaint()
        img = window.gl.grabFramebuffer()
        path = os.path.join(ROOT, ".ref_frames", "preview_qt.png")
        img.save(path)
        print("saved", path, img.size(), "render_ms", window.gl.last_frame_ms)
        app.quit()

    QTimer.singleShot(4000, shot)
    app.exec()
    window.audio.close()
    window.sfx.close()


if __name__ == "__main__":
    main()
