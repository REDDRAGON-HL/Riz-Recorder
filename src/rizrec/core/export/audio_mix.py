"""Offline audio mixing: BGM + hit/drag sounds on the video timeline."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import numpy as np

from ..chart.chart import NOTE_DRAG, Chart
from ..toolchain import CREATE_NO_WINDOW, find_tool

SAMPLE_RATE = 48000
CHANNELS = 2


def _asset_path(root: str, name: str) -> str:
    """Join a fixed asset name under root and reject escapes."""
    base = Path(root).expanduser().resolve()
    candidate = (base / name).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"asset path escapes its root: {name}")
    return str(candidate)


def decode_float(path: str, sample_rate: int = SAMPLE_RATE, ffmpeg_dir: str = "") -> np.ndarray:
    ffmpeg = find_tool("ffmpeg", ffmpeg_dir)
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg")
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            path,
            "-f",
            "f32le",
            "-ac",
            str(CHANNELS),
            "-ar",
            str(sample_rate),
            "-",
        ],
        capture_output=True,
        check=True,
        creationflags=CREATE_NO_WINDOW,
    )
    raw = np.frombuffer(proc.stdout, dtype="<f4")
    if raw.size % CHANNELS:
        raw = raw[: raw.size - raw.size % CHANNELS]
    return raw.reshape(-1, CHANNELS).copy()


def _add_clip(dst: np.ndarray, clip: np.ndarray, start: int, gain: float) -> None:
    if start >= len(dst):
        return
    if start < 0:
        clip = clip[-start:]
        start = 0
    n = min(len(clip), len(dst) - start)
    if n <= 0:
        return
    dst[start : start + n] += clip[:n] * gain


def mix_audio(
    chart: Chart,
    bgm_path: str | None,
    sfx_dir: str,
    duration: float,
    start: float = 0.0,
    lead_in: float = 0.0,
    bgm_volume: float = 1.0,
    hit_volume: float = 0.55,
    cull: bool = True,
    chart_offset: float = 0.0,
    ffmpeg_dir: str = "",
) -> np.ndarray:
    total = max(1, int(math.ceil((lead_in + duration) * SAMPLE_RATE)))
    mix = np.zeros((total, CHANNELS), dtype=np.float32)

    if bgm_path and os.path.isfile(bgm_path):
        bgm = decode_float(bgm_path, ffmpeg_dir=ffmpeg_dir)
        src0 = max(0, int(round(start * SAMPLE_RATE)))
        dst0 = max(0, int(round(lead_in * SAMPLE_RATE)))
        n = min(len(bgm) - src0, total - dst0)
        if n > 0:
            mix[dst0 : dst0 + n] += bgm[src0 : src0 + n] * bgm_volume

    hit_path = _asset_path(sfx_dir, "hit.wav")
    drag_path = _asset_path(sfx_dir, "drag.wav")
    hit_buf = decode_float(hit_path, ffmpeg_dir=ffmpeg_dir) if os.path.isfile(hit_path) else None
    drag_buf = decode_float(drag_path, ffmpeg_dir=ffmpeg_dir) if os.path.isfile(drag_path) else None

    # Quantize to 5 ms buckets and cap identical sounds per bucket to avoid
    # summing dozens of copies into a hard clip on dense charts.
    buckets: dict[tuple[int, int], int] = {}
    for note in chart.notes:
        # The chart offset shifts the visual chart relative to the music, and
        # the hit sounds must follow the visuals (same as the live preview).
        when = note.start_sec + chart_offset
        if when < start - 1e-6:
            continue
        if when > start + duration + 1e-6:
            continue
        clip = drag_buf if note.kind == NOTE_DRAG else hit_buf
        if clip is None:
            continue
        key = (note.kind, int(round(when / 0.005)))
        if cull:
            count = buckets.get(key, 0)
            if count >= 3:
                continue
            buckets[key] = count + 1
        t = lead_in + when - start
        _add_clip(mix, clip, int(round(t * SAMPLE_RATE)), hit_volume)

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    return mix
