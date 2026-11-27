import faulthandler
import os
import sys
import threading
import traceback

from PySide6.QtGui import QIcon, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from ..paths import data_root, resource_root
from .preview import PreviewWindow

ROOT = data_root()


def _install_crash_log():
    """Keep a log next to settings.json so pythonw crashes are diagnosable."""
    try:
        log = open(ROOT / "rizrec_crash.log", "a", encoding="utf-8", buffering=1)
    except Exception:
        return None
    faulthandler.enable(log, all_threads=True)
    log.write(f"\n--- session {os.getpid()} ---\n")

    def hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb, file=log)
        log.flush()
        sys.__excepthook__(exc_type, exc, tb)

    def thread_hook(args):
        traceback.print_exception(
            args.exc_type, args.exc_value, args.exc_traceback, file=log
        )
        log.flush()

    sys.excepthook = hook
    threading.excepthook = thread_hook
    return log


def main() -> None:
    crash_log = _install_crash_log()

    # The audio callback lives in a PortAudio thread and needs the GIL every
    # few ms; the render thread builds vertices in Python and can hold it for
    # tens of ms.  A 1 ms switch interval keeps the callback responsive (with
    # the default 5 ms the preview's audio stalls once rendering gets heavy).
    sys.setswitchinterval(0.001)

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    icon_file = resource_root() / "assets" / "app.ico"
    if icon_file.is_file():
        app.setWindowIcon(QIcon(str(icon_file)))
    # Give the window its own taskbar group so Windows uses the app icon
    # instead of the python.exe default.
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "REDDRAGON.RizRecorder"
        )
    except Exception:
        pass
    window = PreviewWindow()
    window.show()
    try:
        sys.exit(app.exec())
    finally:
        if crash_log is not None:
            try:
                crash_log.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
