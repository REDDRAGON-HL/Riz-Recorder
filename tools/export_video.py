"""Export a Rizline chart to MP4."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rizrec.core.chart import load_chart  # noqa: E402
from rizrec.core.export import export_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", default=os.path.join(ROOT, "assets", "LAIGN_IN.json"))
    ap.add_argument("--audio", default=os.path.join(ROOT, "assets", "LAIGN.mp3"))
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "LAIGN_IN.mp4"))
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--revelation", type=float, default=1.0)
    ap.add_argument("--offset", type=float, default=0.0, help="chart offset in seconds")
    ap.add_argument("--lead-in", type=float, default=0.0, help="seconds before chart time 0")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--song", default="")
    ap.add_argument("--artist", default="")
    ap.add_argument("--bgm-volume", type=float, default=1.0)
    ap.add_argument("--hit-volume", type=float, default=0.55)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--color-mode", choices=["compat", "full", "master"], default="full")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--encoder", default="libx264")
    ap.add_argument("--ffmpeg-dir", default="", help="directory containing ffmpeg/ffplay/ffprobe")
    ap.add_argument("--intro", action="store_true", help="level intro card")
    ap.add_argument("--outro", action="store_true", help="level outro card")
    ap.add_argument("--cover", default="", help="cover art image for the watch card")
    ap.add_argument("--hud", choices=["auto", "full", "off"], default="auto")
    ap.add_argument("--copyright", default="", help="right-bottom copyright mark (empty hides it)")
    ap.add_argument("--illustrator", default="")
    ap.add_argument("--designer", default="")
    args = ap.parse_args()

    chart = load_chart(args.chart)
    print(
        f"谱面 {os.path.basename(args.chart)}: {chart.note_count} notes, "
        f"duration {chart.duration:.2f}s, speed {args.speed}, rev {args.revelation}"
    )
    export_video(
        chart,
        os.path.abspath(args.out),
        os.path.join(ROOT, "assets"),
        audio_path=None if args.no_audio else args.audio,
        width=args.width,
        height=args.height,
        fps=args.fps,
        speed=args.speed,
        revelation=args.revelation,
        chart_offset=args.offset,
        lead_in=args.lead_in,
        start=args.start,
        duration=args.duration,
        song=args.song,
        artist=args.artist,
        bgm_volume=args.bgm_volume,
        hit_volume=args.hit_volume,
        crf=args.crf,
        preset=args.preset,
        encoder=args.encoder,
        color_mode=args.color_mode,
        ffmpeg_dir=args.ffmpeg_dir,
        intro=args.intro,
        outro=args.outro,
        cover_path=args.cover,
        hud_mode=args.hud,
        copyright_text=args.copyright,
        illustrator=args.illustrator,
        designer=args.designer,
        progress=lambda msg: print(msg, flush=True),
    )


if __name__ == "__main__":
    main()
