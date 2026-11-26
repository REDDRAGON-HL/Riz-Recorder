"""Real-time hit/drag sound effects for the preview.

Deprecated: the preview now mixes BGM and hit sounds in one stream
(``audio/player.py``) so the sounds are scheduled on the music timeline.
This module is kept only for the standalone scripts that still use it.
"""

from __future__ import annotations

import os
import threading

import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 2


class SfxPlayer:
    def __init__(self, sfx_dir: str, master: float = 0.55) -> None:
        self.sr = SAMPLE_RATE
        self.master = master
        self.rate = 1.0
        self.enabled = True
        self.error = ""
        self._sd = None
        try:
            import sounddevice as sd

            self._sd = sd
        except Exception as exc:  # pragma: no cover - audio device dependent
            self.error = f"sounddevice 不可用: {exc}"

        self._buffers: dict[str, np.ndarray] = {}
        self._load(sfx_dir)
        self._lock = threading.Lock()
        self._voices: list[dict] = []
        self._thread: threading.Thread | None = None
        self._stream = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------

    def _load(self, sfx_dir: str) -> None:
        from ..export.audio_mix import decode_float

        for name in ("hit", "drag"):
            path = os.path.join(sfx_dir, name + ".wav")
            if not os.path.isfile(path):
                continue
            try:
                buf = decode_float(path, self.sr)
            except Exception as exc:  # pragma: no cover - depends on ffmpeg
                self.error = f"{name}.wav 读取失败: {exc}"
                continue
            # Short fade-out avoids clicks at the end of the clip.
            n = min(len(buf), 256)
            if n:
                buf[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
            self._buffers[name] = buf

    @property
    def available(self) -> bool:
        return self._sd is not None and bool(self._buffers)

    def _start(self) -> None:
        if self._thread is not None or self._sd is None:
            return
        try:
            self._stream = self._sd.OutputStream(
                samplerate=self.sr,
                channels=CHANNELS,
                dtype="float32",
                blocksize=1024,
                latency="low",
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover - audio device dependent
            self.error = f"音频输出打开失败: {exc}"
            self._sd = None
            self._stream = None
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rizrec-sfx", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        block = 1024
        stream = self._stream
        if stream is None:
            return
        while not self._stop.is_set():
            with self._lock:
                voices = list(self._voices)
            mix = np.zeros((block, CHANNELS), dtype=np.float32)
            alive = []
            for v in voices:
                pos = v["pos"]
                step = v["step"]
                buf = v["buf"]
                if pos >= len(buf):
                    continue
                count = block
                idx = pos + step * np.arange(count, dtype=np.float64)
                valid = idx < len(buf) - 1
                if not valid.any():
                    continue
                base = idx.astype(np.int64)
                frac = (idx - base).astype(np.float32)[:, None]
                safe = np.minimum(base, len(buf) - 1)
                nxt = np.minimum(safe + 1, len(buf) - 1)
                samples = buf[safe] * (1.0 - frac) + buf[nxt] * frac
                samples[~valid] = 0.0
                mix += samples * v["gain"]
                v["pos"] = pos + step * count
                if v["pos"] < len(buf):
                    alive.append(v)
            with self._lock:
                self._voices = alive
            np.clip(mix, -1.0, 1.0, out=mix)
            try:
                stream.write(mix)
            except Exception:  # pragma: no cover - device removed
                self._sd = None
                with self._lock:
                    self._voices.clear()
                break

    # ------------------------------------------------------------------

    def trigger(self, name: str, gain: float = 1.0) -> None:
        if not self.enabled or self._sd is None:
            return
        buf = self._buffers.get(name)
        if buf is None:
            return
        self._start()
        if self._stream is None:
            return
        with self._lock:
            self._voices.append(
                {
                    "buf": buf,
                    "pos": 0.0,
                    "step": max(0.25, min(2.0, self.rate)),
                    "gain": self.master * gain,
                }
            )

    def stop_all(self) -> None:
        with self._lock:
            self._voices.clear()

    def set_rate(self, rate: float) -> None:
        self.rate = max(0.25, min(2.0, rate))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
