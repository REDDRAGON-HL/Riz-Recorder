"""Filesystem roots that work both from source and from a PyInstaller build.

``resource_root()`` holds the bundled read-only data (``assets/``); ``data_root()``
is the writable place for ``settings.json`` and the crash log.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    if _frozen():
        # PyInstaller onedir/onefile unpack data under _MEIPASS; --add-data
        # keeps the ``assets/`` directory shape there.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    if _frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]
