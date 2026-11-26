"""Locate ffmpeg-family executables, optionally from a user directory."""

from __future__ import annotations

import os
import shutil

# Keep ffmpeg/ffprobe from flashing a console window under pythonw.exe.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def find_tool(name: str, directory: str | None = None) -> str | None:
    """Return the executable path, preferring ``directory`` when given."""
    exe = name + (".exe" if os.name == "nt" else "")
    if directory:
        directory = os.path.expandvars(os.path.expanduser(directory.strip()))
        for candidate in (os.path.join(directory, exe), os.path.join(directory, name)):
            if os.path.isfile(candidate):
                return candidate
    return shutil.which(name) or shutil.which(exe)
