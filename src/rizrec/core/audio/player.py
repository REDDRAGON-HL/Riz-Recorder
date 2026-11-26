"""Preview audio: one sounddevice stream for BGM + hit sounds, device clock.

Why not ffplay (the previous approach): ffplay reports its own clock only a
few times per second and that clock runs on the *buffer* position, so the
visual time leads the audible sound by an unknown device latency; the value
also jitters between refreshes, which made the preview stutter once audio was
loaded.  There was no way to tune that away because the real latency is
unknown.

This module keeps everything in one output stream:

* the BGM is decoded once to a float32 array and the callback advances a read
  position through it (``rate`` is a resampling factor, so speed changes keep
  the same stream);
* the preview clock is the device clock: ``submitted_frames / sample_rate``
  minus the stream's own reported output latency.  No guessed constant;
* hit sounds are **scheduled on the BGM timeline** instead of being fired when
  the UI notices a note, so they land sample-accurately relative to the music
  (the game also triggers sounds on the chart timeline, not on frame ticks).

The callback only copies/slices preallocated numpy arrays; all decoding and
mixing setup happens on the caller's thread.
"""

from __future__ import annotations

import heapq
import os
import threading

import numpy as np

from ..toolchain import find_tool

SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK = 2048              # ~43 ms per callback; WASAPI reports ~85 ms buffer
SCHEDULE_LEAD = 0.20      # keep hit sounds queued this far ahead


def _wasapi_default_output(sd) -> int | None:
    """Default Windows WASAPI output device (falls back to PortAudio default)."""
    try:
        for api in sd.query_hostapis():
            if "WASAPI" in str(api.get("name", "")):
                dev = int(api.get("default_output_device", -1))
                if dev >= 0:
                    return dev
    except Exception:
        pass
    return None


def _resampled(buf: np.ndarray, pos: float, step: float, count: int) -> np.ndarray:
    """Linear-interpolated slice of ``count`` frames starting at ``pos``."""
    if abs(step - 1.0) < 1e-6:
        # Fast path (normal 1x playback): a plain slice, no index arrays.  The
        # callback holds the GIL, so keep the work minimal.
        start = int(pos)
        end = min(start + count, len(buf))
        if end <= start:
            return None
        out = buf[start:end]
        if end - start < count:
            pad = np.zeros((count - (end - start), buf.shape[1]), dtype=buf.dtype)
            out = np.concatenate((out, pad))
        return out
    idx = pos + step * np.arange(count, dtype=np.float64)
    valid = idx < len(buf) - 1
    if not valid.any():
        return None
    base = idx.astype(np.int64)
    frac = (idx - base).astype(np.float32)[:, None]
    safe = np.minimum(base, len(buf) - 1)
    nxt = np.minimum(safe + 1, len(buf) - 1)
    out = buf[safe] * (1.0 - frac) + buf[nxt] * frac
    out[~valid] = 0.0
    return out


class SfxFacade:
    """``SfxPlayer``-compatible view onto the engine's hit-sound mixer."""

    __slots__ = ("_engine",)

    def __init__(self, engine: "AudioPlayer") -> None:
        self._engine = engine

    @property
    def available(self) -> bool:
        return self._engine.sfx_available

    @property
    def enabled(self) -> bool:
        return self._engine.sfx_enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._engine.sfx_enabled = bool(value)

    @property
    def error(self) -> str:
        return self._engine.sfx_error

    def trigger(self, name: str, gain: float = 1.0) -> None:
        self._engine.schedule_sfx(name, None, gain)

    def stop_all(self) -> None:
        self._engine.stop_sfx()

    def set_rate(self, rate: float) -> None:
        self._engine.set_rate(rate)

    def close(self) -> None:
        self._engine.close()


class AudioPlayer:
    def __init__(self, ffmpeg_dir: str = "") -> None:
        self.path = ""
        self.duration = 0.0
        self.rate = 1.0
        self.playing = False
        self.error = ""
        self.bgm_volume = 1.0
        self.sfx_volume = 0.55
        self.sfx_enabled = True
        self.sfx_error = ""
        self.ffmpeg_dir = ffmpeg_dir
        self._ffmpeg = find_tool("ffmpeg", ffmpeg_dir)

        self._bgm: np.ndarray | None = None
        self._sfx: dict[str, np.ndarray] = {}

        self._lock = threading.Lock()
        self._stream = None
        self._sd = None
        self._rate = 1.0
        self._bgm_pos = 0.0          # read position in frames
        self._base_frames = 0        # submitted frames when playback (re)started
        self._base_time = 0.0        # BGM seconds at that moment
        self._frames = 0             # frames submitted since the stream opened
        self._voices: list[dict] = []
        self._pending: list[tuple[float, str, float]] = []
        self._status_count = 0
        self._pause_pos = 0.0
        self._open_stream()

    # ------------------------------------------------------------------
    # device

    def _open_stream(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - audio device dependent
            self.error = f"sounddevice 不可用: {exc}"
            return
        self._sd = sd
        # PortAudio's default host API on Windows is MME, whose output buffer
        # is ~100 ms: that was a large, invisible A/V offset.  WASAPI reports
        # and uses a few milliseconds, so the device clock is usable.
        device = _wasapi_default_output(sd)
        try:
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=BLOCK,
                latency="low",
                device=device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover - audio device dependent
            self.error = f"音频输出打开失败: {exc}"
            self._stream = None

    @property
    def latency(self) -> float:
        stream = self._stream
        if stream is None:
            return 0.0
        try:
            return float(stream.latency or 0.0)
        except Exception:
            return 0.0

    @property
    def device_error(self) -> str:
        return self.error

    # ------------------------------------------------------------------
    # loading

    def set_ffmpeg_dir(self, directory: str) -> None:
        self.ffmpeg_dir = directory
        self._ffmpeg = find_tool("ffmpeg", directory)

    def load(self, path: str) -> bool:
        """Decode the whole track to float32 (fast: ffmpeg does one pass)."""
        self.stop()
        self.error = ""
        if not self._ffmpeg:
            self.error = "找不到 ffmpeg"
            return False
        from ..export.audio_mix import decode_float

        try:
            buf = decode_float(path, SAMPLE_RATE, self.ffmpeg_dir)
        except Exception as exc:  # pragma: no cover - depends on ffmpeg
            self.error = f"音频读取失败: {exc}"
            return False
        if buf.size == 0:
            self.error = "音频时长无效"
            return False
        self._bgm = buf
        self.path = path
        self.duration = len(buf) / SAMPLE_RATE
        self._bgm_pos = 0.0
        self._base_time = 0.0
        self._base_frames = self._frames
        self._pause_pos = 0.0
        return True

    def load_sfx(self, sfx_dir: str) -> None:
        from ..export.audio_mix import decode_float

        for name in ("hit", "drag"):
            path = os.path.join(sfx_dir, name + ".wav")
            if not os.path.isfile(path):
                continue
            try:
                buf = decode_float(path, SAMPLE_RATE, self.ffmpeg_dir)
            except Exception as exc:  # pragma: no cover - depends on ffmpeg
                self.sfx_error = f"{name}.wav 读取失败: {exc}"
                continue
            n = min(len(buf), 256)
            if n:
                buf[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
            self._sfx[name] = buf

    @property
    def sfx_available(self) -> bool:
        return self._stream is not None and bool(self._sfx)

    def _limit(self) -> float:
        """Upper bound for play/seek positions.

        With no BGM loaded ``duration`` is 0 and clamping to it forced every
        seek/play back to 0 (the preview then could not scrub without audio).
        """
        return self.duration if self.duration > 0.0 else 1.0e9

    @property
    def sfx(self) -> SfxFacade:
        return SfxFacade(self)

    # ------------------------------------------------------------------
    # transport

    def play(self, start: float | None = None) -> None:
        if self._stream is None:
            self._open_stream()
        if self._stream is None:
            self.playing = False
            return
        if start is not None:
            self._pause_pos = max(0.0, min(self._limit(), start))
        pos = self._pause_pos
        with self._lock:
            self._bgm_pos = pos * SAMPLE_RATE
            self._base_time = pos
            self._base_frames = self._frames
            self._voices = []
            self._pending.clear()
            self.playing = True

    def pause(self) -> None:
        if self.playing:
            self._pause_pos = self.position
        with self._lock:
            self.playing = False
            self._voices = []
            self._pending.clear()
            # The read position must survive so a resume without an explicit
            # start continues where the audible position was.
            self._bgm_pos = self._pause_pos * SAMPLE_RATE
            self._base_time = self._pause_pos
            self._base_frames = self._frames

    def stop(self) -> None:
        self.pause()
        self._pause_pos = 0.0
        with self._lock:
            self._bgm_pos = 0.0
            self._base_time = 0.0
            self._base_frames = self._frames

    def seek(self, seconds: float) -> None:
        seconds = max(0.0, min(self._limit(), seconds))
        with self._lock:
            self._bgm_pos = seconds * SAMPLE_RATE
            self._base_time = seconds
            self._base_frames = self._frames
            self._pause_pos = seconds
            self._voices = []
            self._pending.clear()

    def set_rate(self, rate: float) -> None:
        """Change the resampling rate without a visual/clock jump.

        The clock is ``base + (frames - base_frames) * rate``: rebasing on the
        current audible position keeps the preview continuous when the user
        changes speed mid-play.
        """
        rate = max(0.25, min(2.0, rate))
        with self._lock:
            if abs(rate - self._rate) < 1e-6:
                return
            submitted = self._bgm_pos / SAMPLE_RATE
            audible = submitted - self.latency * self._rate
            self._rate = rate
            if self.playing:
                self._base_time = max(self._base_time, audible)
                self._base_frames = self._frames
        self.rate = self._rate

    @property
    def position(self) -> float:
        """Audible BGM position in seconds (device clock minus its latency).

        The read pointer advances ``rate`` frames per output frame, so the
        track time advances ``rate`` times faster than the wall clock; the
        visual clock has to do the same or the preview desyncs at any speed
        other than 1x.
        """
        if not self.playing:
            return self._pause_pos
        with self._lock:
            base = self._base_time
            base_frames = self._base_frames
            step = self._rate
        elapsed = (self._frames - base_frames) / SAMPLE_RATE
        pos = base + elapsed * step - self.latency * step
        pos = max(base, pos)
        duration = self.duration
        if duration > 0.0 and pos >= duration:
            self._pause_pos = duration
            self.playing = False
            return duration
        return pos

    # ------------------------------------------------------------------
    # hit sounds

    def schedule_sfx(self, name: str, when: float | None, gain: float = 1.0) -> None:
        """Queue a hit sound at BGM time ``when`` (None = as soon as possible)."""
        if not self.sfx_enabled or self._stream is None:
            return
        if name not in self._sfx:
            return
        if when is None:
            with self._lock:
                when = self._bgm_pos / SAMPLE_RATE
        with self._lock:
            heapq.heappush(self._pending, (float(when), name, float(gain)))

    def stop_sfx(self) -> None:
        with self._lock:
            self._voices = []
            self._pending.clear()

    def clear_schedule_before(self, when: float) -> None:
        with self._lock:
            self._pending = [e for e in self._pending if e[0] >= when]
            heapq.heapify(self._pending)

    # ------------------------------------------------------------------
    # callback

    def _callback(self, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            self._status_count += 1
        outdata.fill(0.0)
        with self._lock:
            playing = self.playing
            bgm = self._bgm if playing else None
            pos = self._bgm_pos
            step = self._rate
            gain = self.bgm_volume
            block_start = pos / SAMPLE_RATE
            block_end = (pos + step * frames) / SAMPLE_RATE
            due = []
            while self._pending and self._pending[0][0] < block_end:
                due.append(heapq.heappop(self._pending))
            voices = self._voices
        # --- BGM -----------------------------------------------------
        if bgm is not None and pos < len(bgm) - 1:
            chunk = _resampled(bgm, pos, step, frames)
            if chunk is not None:
                if gain != 1.0:
                    outdata += chunk * gain
                else:
                    outdata += chunk
        if playing:
            pos += step * frames
        # --- scheduled hit sounds (sample accurate) ------------------
        for when, name, voice_gain in due:
            buf = self._sfx.get(name)
            if buf is None or not self.sfx_enabled:
                continue
            offset = 0
            if when > block_start and step > 1e-6:
                offset = int(round((when - block_start) * SAMPLE_RATE / step))
                offset = max(0, min(frames - 1, offset))
            voices.append({"buf": buf, "pos": 0.0, "step": step,
                           "gain": self.sfx_volume * voice_gain, "start": offset})
        alive = []
        for voice in voices:
            start = voice.pop("start", 0)
            n = frames - start
            if n <= 0:
                voice["start"] = start - frames
                alive.append(voice)
                continue
            chunk = _resampled(voice["buf"], voice["pos"], voice["step"], n)
            voice["pos"] += voice["step"] * n
            if chunk is not None:
                outdata[start:start + n] += chunk * voice["gain"]
            if voice["pos"] < len(voice["buf"]) - 1:
                alive.append(voice)
        np.clip(outdata, -1.0, 1.0, out=outdata)
        with self._lock:
            if playing:
                self._bgm_pos = pos
            self._voices = alive
            self._frames += frames

    # ------------------------------------------------------------------

    def toggle_mute(self, muted: bool) -> None:
        self.bgm_volume = 0.0 if muted else 1.0

    def close(self) -> None:
        self.pause()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._sd = None
