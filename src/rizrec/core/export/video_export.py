"""Deterministic offline video export (raw frames -> ffmpeg)."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
import time
from typing import Callable

import glfw
import moderngl
import numpy as np

from ..chart.chart import Chart
from ..render import watch
from ..render.assets import AssetStore
from ..render.renderer import RenderSettings, Renderer
from ..toolchain import CREATE_NO_WINDOW, find_tool
from .audio_mix import SAMPLE_RATE, mix_audio


def _make_context(width: int, height: int):
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(64, 64, "rizrec-export", None, None)
    if not win:
        raise RuntimeError("glfw.create_window failed")
    glfw.make_context_current(win)
    ctx = moderngl.create_context(require=330)
    return win, ctx


def _probe_duration(path: str, ffmpeg_dir: str = "") -> float:
    """Length of an audio/video file in seconds (0.0 when unknown)."""
    ffprobe = find_tool("ffprobe", ffmpeg_dir)
    if not ffprobe or not path or not os.path.isfile(path):
        return 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True,
            creationflags=CREATE_NO_WINDOW,
        ).stdout
        return max(0.0, float(out.strip()))
    except Exception:
        return 0.0


def _encoder_args(encoder: str, crf: int, preset: str) -> list[str]:
    if encoder in ("libx264", "libx265"):
        return ["-preset", preset, "-crf", str(crf)]
    if encoder.endswith("_qsv"):        return ["-preset", "veryfast", "-global_quality", str(crf)]
    if encoder.endswith("_nvenc"):
        return ["-preset", "p5", "-cq", str(crf)]
    if encoder.endswith("_amf"):
        return ["-quality", "speed", "-qp_p", str(crf)]
    return ["-q", str(crf)]


# GLFW and the ffmpeg child are process-global: two exports running in two
# threads (task page + export dialog, or a second click while the first runs)
# used to hang or kill each other's context.  Serialise them instead.
_EXPORT_LOCK = threading.Lock()


def export_video(*args, **kwargs) -> str:
    """Serialised wrapper around :func:`_export_impl`."""
    with _EXPORT_LOCK:
        return _export_impl(*args, **kwargs)


def _export_impl(
    chart: Chart,
    out_path: str,
    assets_root: str,
    audio_path: str | None = None,
    width: int = 1080,
    height: int = 1920,
    fps: int = 60,
    speed: float = 6.0,
    revelation: float = 1.0,
    chart_offset: float = 0.0,
    lead_in: float = 0.0,
    start: float = 0.0,
    duration: float | None = None,
    song: str = "",
    artist: str = "",
    bgm_volume: float = 1.0,
    hit_volume: float = 0.55,
    crf: int = 16,
    preset: str = "veryfast",
    encoder: str = "libx264",
    show_watermark: bool = True,
    color_mode: str = "full",
    bitrate: int | None = None,
    ffmpeg_dir: str = "",
    intro: bool = False,
    outro: bool = False,
    intro_preroll: float = watch.INTRO_PREROLL,
    cover_path: str = "",
    hud_mode: str = "auto",
    copyright_text: str = "",
    illustrator: str = "",
    designer: str = "",
    progress: Callable[[str], None] | None = None,
) -> str:
    log = progress or (lambda _msg: None)
    use_intro = intro and abs(start) < 1e-9
    if use_intro:
        lead_in = intro_preroll
    audio_length = _probe_duration(audio_path, ffmpeg_dir) if audio_path else 0.0
    outro_start = None
    if outro:
        # RizPlayer: the card fires at the first frame where
        # ``t > audio - 0.1`` **and** ``t >= notesEnd`` (last note / hold tail).
        base = audio_length if audio_length > 0.0 else chart.duration
        outro_start = max(0.0, base - watch.OUTRO_LEAD, chart.notes_end)
    segment = chart.duration - start if duration is None else max(0.05, duration)
    # An explicit 时长 must win: previously the outro tail always extended the
    # segment to the end of the song, so the range appeared to be ignored.
    # With the outro the video ends exactly when the card lands (RizPlayer
    # appends a WATCH_OUTRO = 1.7 s tail to the frame where the card fires).
    if outro_start is not None and duration is None:
        segment = outro_start + watch.OUTRO_TOTAL - start
    total_time = lead_in + segment
    total_frames = max(1, int(round(total_time * fps)))
    out_abs = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    log(f"混音: {segment:.2f}s 片段 + {lead_in:.2f}s 前导")
    if use_intro:
        log(f"进场白卡: 前导 {intro_preroll:.2f}s（chart 0 在 {lead_in:.2f}s）")
    if outro_start is not None:
        log(f"退场白卡: chart {outro_start:.2f}s 起，共 {watch.OUTRO_TOTAL:.2f}s")
    sfx_dir = os.path.join(assets_root, "audio")
    if audio_path:
        audio = mix_audio(
            chart,
            audio_path,
            sfx_dir,
            segment,
            start=start,
            lead_in=lead_in,
            bgm_volume=bgm_volume,
            hit_volume=hit_volume,
            chart_offset=chart_offset,
            ffmpeg_dir=ffmpeg_dir,
        )
    else:
        audio = mix_audio(
            chart,
            None,
            sfx_dir,
            segment,
            start=start,
            lead_in=lead_in,
            bgm_volume=0.0,
            hit_volume=hit_volume,
            chart_offset=chart_offset,
            ffmpeg_dir=ffmpeg_dir,
        )

    audio_tmp = tempfile.NamedTemporaryFile(
        prefix="rizrec_audio_", suffix=".f32", delete=False, dir=tempfile.gettempdir()
    )
    try:
        audio_tmp.write(np.ascontiguousarray(audio, dtype="<f4").tobytes())
        audio_tmp.close()
        audio_path_tmp = audio_tmp.name
    except Exception:
        try:
            audio_tmp.close()
        finally:
            pass
        raise

    ffmpeg = find_tool("ffmpeg", ffmpeg_dir)
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg")

    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "2",
        "-i",
        audio_path_tmp,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        encoder,
        *(["-preset", preset, "-b:v", f"{bitrate}k"] if bitrate else _encoder_args(encoder, 0 if color_mode == "master" else crf, preset)),
        "-vf",
        "vflip,scale=in_range=full:out_range="
        + ("tv" if color_mode == "compat" else "pc")
        + ":in_color_matrix=bt709:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int",
        "-pix_fmt",
        "yuv444p" if color_mode == "master" else "yuv420p",
        "-color_range",
        "tv" if color_mode == "compat" else "pc",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        *(
            ["-c:a", "flac"]
            if color_mode == "master"
            else ["-c:a", "aac", "-b:a", "320k"]
        ),
        "-shortest",
        "-movflags",
        "+faststart",
        out_abs,
    ]

    win, ctx = _make_context(width, height)
    proc = None
    try:
        assets = AssetStore(ctx, assets_root)
        settings = RenderSettings(
            speed=speed,
            revelation=revelation,
            offset=chart_offset,
            song_name=song or chart.song_name,
            artist=artist,
            show_watermark=show_watermark,
            illustrator=illustrator,
            designer=designer,
            cover_path=cover_path,
            hud_mode=hud_mode,
            copyright_text=copyright_text,
            intro=use_intro,
            outro=outro_start is not None,
            intro_preroll=intro_preroll,
            outro_start=outro_start,
        )
        renderer = Renderer(ctx, chart, assets, width, height, settings)
        # 4x MSAA keeps diagonal lines, hold edges and note circles smooth.
        max_samples = int(ctx.info.get("GL_MAX_SAMPLES", 0) or 0)
        samples = 8 if max_samples >= 8 else (4 if max_samples >= 4 else 0)
        if samples:
            target_tex = ctx.texture((width, height), 4, samples=samples)
            target = ctx.framebuffer([target_tex])
            resolve_tex = ctx.texture((width, height), 4)
            resolve = ctx.framebuffer([resolve_tex])
        else:
            target_tex = ctx.texture((width, height), 4)
            target = ctx.framebuffer([target_tex])
            resolve_tex = None
            resolve = None
        target.use()
        ctx.viewport = (0, 0, width, height)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
        err_chunks: list[bytes] = []

        def _drain() -> None:
            assert proc is not None and proc.stderr is not None
            for chunk in iter(lambda: proc.stderr.read(4096), b""):
                err_chunks.append(chunk)

        threading.Thread(target=_drain, daemon=True).start()
        log(f"编码: {total_frames} 帧 @ {fps}fps, {width}x{height}")
        started = time.perf_counter()
        last_report = started
        for frame in range(total_frames):
            video_t = frame / fps
            chart_t = video_t - lead_in + start
            if chart_t < 0.0 and not use_intro:
                # Lead-in before the chart: black, unless the watch card runs.
                data = bytes(width * height * 3)
            else:
                target.clear(0.0, 0.0, 0.0, 1.0)
                renderer.render(chart_t)
                if samples:
                    ctx.copy_framebuffer(resolve, target)
                    data = resolve.read(components=3)
                else:
                    data = target.read(components=3)
            proc.stdin.write(data)

            now = time.perf_counter()
            if now - last_report >= 0.5 or frame + 1 == total_frames:
                done = frame + 1
                rate = done / max(1e-6, now - started)
                eta = (total_frames - done) / max(1e-6, rate)
                log(f"  {done}/{total_frames}  {rate:5.1f} fps  ETA {eta:5.1f}s")
                last_report = now
        proc.stdin.close()
        code = proc.wait()
        err = b"".join(err_chunks).decode("utf-8", "replace")
        if code != 0:
            raise RuntimeError(f"ffmpeg 失败 ({code}): {err[-2000:]}")
        log(f"完成: {os.path.basename(out_abs)}")
        return out_abs
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            if os.path.exists(audio_path_tmp):
                os.remove(audio_path_tmp)
        except Exception:
            pass
        glfw.terminate()
