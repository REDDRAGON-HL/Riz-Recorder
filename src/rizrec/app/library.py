"""Chart discovery, metadata and audio matching for the app."""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
CHART_PREFIX = "chart_"


@dataclass
class ChartEntry:
    chart_path: str
    audio_path: str = ""
    song: str = ""
    artist: str = ""
    illustrator: str = ""
    designer: str = ""
    cover_path: str = ""
    source: str = ""
    settings: dict = field(default_factory=dict)
    _temp_dir: str | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        name = self.song or os.path.basename(os.path.dirname(self.chart_path)) or "Chart"
        diff = difficulty_of(self.chart_path)
        return f"{name}  [{diff}]" if diff else name


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+", "", text.casefold())


def difficulty_of(path: str) -> str:
    name = os.path.basename(path)
    m = re.search(r"(EZ|HD|IN|AT|SP|MR|CN)$", os.path.splitext(name)[0], re.I)
    if m:
        return m.group(1).upper()
    return ""


def is_chart_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    if os.path.splitext(name)[1].lower() not in ("", ".json"):
        return False
    try:
        if os.path.getsize(path) < 2048:
            return False
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except Exception:
        return False
    if not isinstance(data, dict) or "m_GameObject" in data:
        return False
    return "bPM" in data and "lines" in data


def find_charts(root: str) -> list[str]:
    if os.path.isfile(root):
        return [root] if is_chart_file(root) else []
    found = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            if is_chart_file(path):
                found.append(path)
    found.sort()
    return found


def _metadata_for(chart_path: str) -> tuple[str, str]:
    """Read sibling *song.artist.index.json* (Unity MonoBehaviour) if present."""
    folder = os.path.dirname(chart_path)
    for name in os.listdir(folder):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(folder, name)
        if path == chart_path or os.path.getsize(path) > 8192:
            continue
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if not isinstance(data, dict) or "m_Name" not in data:
            continue
        parts = str(data["m_Name"]).split(".")
        artist = parts[-2] if len(parts) >= 3 else ""
        return artist, os.path.basename(folder)
    return "", os.path.basename(folder)


def _audio_candidates(chart_path: str, audio_dir: str) -> list[str]:
    folders = [os.path.dirname(chart_path)]
    if audio_dir and os.path.isdir(audio_dir):
        folders.append(audio_dir)
    out = []
    for folder in folders:
        try:
            for name in os.listdir(folder):
                if name.lower().endswith(AUDIO_EXTS):
                    out.append(os.path.join(folder, name))
        except Exception:
            pass
    return out


def match_audio(chart_path: str, audio_dir: str) -> str:
    artist, folder_song = _metadata_for(chart_path)
    stem = os.path.splitext(os.path.basename(chart_path))[0]
    stem = re.sub(r"_(EZ|HD|IN|AT|SP|MR|CN)$", "", stem, flags=re.I)
    keys = [k for k in {_norm(folder_song), _norm(stem)} if k]
    best = ("", 0)
    for cand in _audio_candidates(chart_path, audio_dir):
        cname = _norm(os.path.splitext(os.path.basename(cand))[0])
        for key in keys:
            if cname.startswith(key) or key.startswith(cname):
                score = len(key)
                if score > best[1]:
                    best = (cand, score)
    return best[0]


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
COVER_SUFFIXES = ("cover", "art", "illustration", "jacket", "曲绘", "插画")


def match_cover(chart_path: str) -> str:
    """Find a cover art file next to the chart (``<song>_cover.png`` etc.)."""
    folder = os.path.dirname(os.path.abspath(chart_path))
    try:
        names = os.listdir(folder)
    except OSError:
        return ""
    stem = re.sub(
        r"_(EZ|HD|IN|AT|SP|MR|CN)$",
        "",
        os.path.splitext(os.path.basename(chart_path))[0],
        flags=re.I,
    )
    keys = {_norm(stem)}
    if "_" in stem:
        keys.add(_norm(stem.split("_")[0]))
    keys = {k for k in keys if len(k) >= 3}
    best = ("", 0)
    for name in names:
        if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
            continue
        norm = _norm(os.path.splitext(name)[0])
        for suffix in COVER_SUFFIXES:
            tail = _norm(suffix)
            if not tail or not norm.endswith(tail):
                continue
            base = norm[: -len(tail)]
            if not base:
                continue
            for key in keys:
                if base == key or (len(base) >= 3 and key.startswith(base)):
                    score = len(base)
                    if score > best[1]:
                        best = (os.path.join(folder, name), score)
    return best[0]


def make_entry(chart_path: str, audio_dir: str, temp_dir: str | None = None) -> ChartEntry:
    artist, folder_song = _metadata_for(chart_path)
    song = folder_song
    if song.lower() in ("assets", "test", ""):
        song = re.sub(r"_(EZ|HD|IN|AT|SP|MR|CN)$", "", os.path.splitext(os.path.basename(chart_path))[0], flags=re.I)
    return ChartEntry(
        chart_path=chart_path,
        audio_path=match_audio(chart_path, audio_dir),
        song=song,
        artist=artist,
        illustrator="",
        cover_path=match_cover(chart_path),
        source=os.path.basename(chart_path),
        _temp_dir=temp_dir,
    )


def import_paths(paths: list[str], audio_dir: str) -> list[ChartEntry]:
    entries: list[ChartEntry] = []
    for path in paths:
        low = path.lower()
        if low.endswith((".zip", ".7z", ".rar")):
            entries.extend(_import_archive(path, audio_dir))
        elif os.path.isdir(path):
            for chart in find_charts(path):
                entries.append(make_entry(chart, audio_dir))
        elif os.path.isfile(path):
            if is_chart_file(path):
                entries.append(make_entry(path, audio_dir))
        else:
            for chart in find_charts(path):
                entries.append(make_entry(chart, audio_dir))
    # De-duplicate by resolved path
    seen = set()
    unique = []
    for e in entries:
        key = os.path.abspath(e.chart_path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    return unique


def _import_archive(path: str, audio_dir: str) -> list[ChartEntry]:
    if not zipfile.is_zipfile(path):
        return []
    temp = tempfile.mkdtemp(prefix="rizrec-chart-")
    try:
        with zipfile.ZipFile(path) as zf:
            base = os.path.realpath(temp)
            for member in zf.infolist():
                target = os.path.realpath(os.path.join(base, member.filename))
                if target != base and not target.startswith(base + os.sep):
                    continue  # refuse zip-slip entries
                zf.extract(member, temp)
    except Exception:
        return []
    return [make_entry(chart, audio_dir, temp) for chart in find_charts(temp)]
