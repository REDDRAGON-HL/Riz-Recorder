"""Render individual frames to PNG for visual comparison.

Usage:
    python tools/render_frame.py 12 45 90 --out .ref_frames/out
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import glfw
import moderngl
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rizrec.core.chart import load_chart  # noqa: E402
from rizrec.core.render.assets import AssetStore  # noqa: E402
from rizrec.core.render.renderer import Renderer, RenderSettings  # noqa: E402


def make_context():
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(64, 64, "rizrec", None, None)
    if not win:
        raise RuntimeError("glfw.create_window failed")
    glfw.make_context_current(win)
    ctx = moderngl.create_context(require=330)
    return win, ctx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("--chart", default=os.path.join(ROOT, "assets", "LAIGN_IN.json"))
    ap.add_argument("--assets", default=os.path.join(ROOT, "assets"))
    ap.add_argument("--out", default=os.path.join(ROOT, ".ref_frames", "out"))
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--speed", type=float, default=6.5)
    ap.add_argument("--revelation", type=float, default=0.93)
    ap.add_argument("--song", default="LAIGN")
    ap.add_argument("--artist", default="Kiu feat. Nikki Simmons")
    ap.add_argument("--offset", type=float, default=0.0, help="video lead-in seconds")
    ap.add_argument("--no-watermark", action="store_true")
    ap.add_argument("--intro", action="store_true", help="draw the watch intro card")
    ap.add_argument("--outro", action="store_true", help="draw the watch outro card")
    ap.add_argument("--outro-start", type=float, default=None)
    ap.add_argument("--cover", default="")
    ap.add_argument("--hud", choices=["auto", "full", "off"], default="auto")
    ap.add_argument("--illustrator", default="")
    ap.add_argument("--designer", default="")
    ap.add_argument("--copyright", default="")
    args = ap.parse_args()

    win, ctx = make_context()
    chart = load_chart(args.chart)
    assets = AssetStore(ctx, args.assets)
    settings = RenderSettings(
        speed=args.speed,
        revelation=args.revelation,
        song_name=args.song,
        artist=args.artist,
        show_watermark=(args.hud != "off") and not args.no_watermark,
        offset=args.offset,
        intro=args.intro,
        outro=args.outro,
        outro_start=args.outro_start,
        cover_path=args.cover,
        hud_mode=args.hud,
        illustrator=args.illustrator,
        designer=args.designer,
        copyright_text=args.copyright,
    )
    w, h = args.width, args.height
    samples = 8 if int(ctx.info.get("GL_MAX_SAMPLES", 0) or 0) >= 8 else (4 if int(ctx.info.get("GL_MAX_SAMPLES", 0) or 0) >= 4 else 0)
    if samples:
        ms = ctx.framebuffer([ctx.texture((w, h), 4, samples=samples)])
        fbo = ctx.framebuffer([ctx.texture((w, h), 4)])
        target = ms
    else:
        ms = None
        fbo = ctx.framebuffer([ctx.texture((w, h), 4)])
        target = fbo
    renderer = Renderer(ctx, chart, assets, w, h, settings)
    target.use()
    ctx.viewport = (0, 0, w, h)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for t in args.times:
        target.clear(0.0, 0.0, 0.0, 1.0)
        t0 = time.perf_counter()
        renderer.render(t)
        ctx.finish()
        dt = time.perf_counter() - t0
        if ms is not None:
            ctx.copy_framebuffer(fbo, ms)
            data = fbo.read(components=3)
        else:
            data = fbo.read(components=3)
        img = Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)
        name = f"{args.out}_{int(round(t * 1000)):06d}.png"
        img.save(name)
        print(f"t={t:7.3f}s  render={dt * 1000:6.1f}ms  -> {name}")
    glfw.terminate()


if __name__ == "__main__":
    main()
