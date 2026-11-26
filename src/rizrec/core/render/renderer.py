"""The Rizline playfield renderer (RizPlayer-compatible layout)."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import moderngl

from ..chart.chart import (
    NOTE_DRAG,
    NOTE_HOLD,
    NOTE_TAP,
    Chart,
    Color,
    NoteState,
    color_at,
    find_value,
)
from ..chart.easing import ease
from . import hitfx
from . import hud
from . import watch
from .assets import (
    TEX_CIRCLE,
    TEX_DRAG,
    TEX_HITFX,
    TEX_HOLD,
    TEX_NOTEBG,
    TEX_RING,
    TEX_SIDE,
    TEX_TEXT,
    TEX_WHITE,
    AssetStore,
)
from .mesh import MeshBuilder

LINE_W_BASE = 0.045 * 360.0 / (3.3125 * 2.0)  # ~2.4453 px on a 360px canvas
# The game draws easeZero/easeOne instant-x jumps with a separate thin line
# (not the 0.045 LineRenderer ribbon).  Measured against the device capture:
# the jumper is about one third of the judged-line width.
INSTANT_LINE_RATIO = 1.0 / 3.0
JUDGE_RING_RATIO = 16.0 / 360.0
HITFX_DURATION = hitfx.HITFX_DURATION
DEFAULT_EFFECT = Color(200, 200, 200, 255)


@dataclass
class RenderSettings:
    speed: float = 6.0
    revelation: float = 1.0
    scale: float = 1.0
    offset: float = 0.0
    show_watermark: bool = True
    artist: str = ""
    song_name: str = ""
    copyright_text: str = ""  # user-supplied; empty hides the mark
    judge_ratio: float | None = None  # judge line y / height, None = RizPlayer formula
    illustrator: str = ""
    designer: str = ""
    cover_path: str = ""
    hud_mode: str = "auto"  # "auto" shows the watch-mode pill on the card
    intro: bool = False
    outro: bool = False
    intro_preroll: float = watch.INTRO_PREROLL
    outro_start: float | None = None  # chart time where the outro card begins


def _rgba(c: Color, alpha: float = 1.0):
    return (c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0 * alpha)


def _is_opaque_white(c: Color) -> bool:
    return c.r == 255 and c.g == 255 and c.b == 255 and c.a == 255


def _curve_stride(p, q, window_speed: float) -> int:
    """Sampling step the game uses for one chart segment.

    ``LineControl.SetDynamicLines`` picks the step from the segment's length
    in scene units (constants 20/10/5/2/1; moving canvases use a looser set).
    The step matters: Unity's strip turns each submitted position into a
    cross-section whose miter limit depends on the previous chord, so the
    same curve with a different step has a different seam width.
    """
    length = math.hypot((q.x - p.x) * 5.625, window_speed * (q.fp - p.fp))
    if p.canvas != q.canvas:
        if length < 0.5:
            return 20
        if length < 1.5:
            return 10
        if length < 2.5:
            return 5
        return 2 if length < 3.5 else 1
    if length < 0.25:
        return 20
    if length < 0.5:
        return 10
    if length < 1.0:
        return 5
    return 2 if length < 2.0 else 1


class Renderer:
    def __init__(
        self,
        ctx: moderngl.Context,
        chart: Chart,
        assets: AssetStore,
        width: int,
        height: int,
        settings: RenderSettings | None = None,
    ):
        self.ctx = ctx
        self.chart = chart
        self.assets = assets
        self.width = int(width)
        self.height = int(height)
        self.settings = settings or RenderSettings()
        self.mesh = MeshBuilder(ctx, assets)
        self._text_lines: dict[str, object] = {}
        self.watch_card = watch.WatchCard(assets, self.width, self.height)
        self.card = watch.CardBatch(ctx)
        self.hud = hud.Hud(assets, self.width, self.height)
        self._build_text()
        try:
            self.hud.prewarm(self.chart, self._watch_accent(0.0))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # layout helpers

    @property
    def real_speed(self) -> float:
        return (215.0 / 32.0 + self.settings.speed) * (10.0 / 129.0)

    def metrics(self) -> tuple[float, float, float, float, float]:
        """Mobile scene: (scene_scale, unit_px, origin_y, judge_y, field_width)."""
        w, h = self.width, self.height
        aspect = w / h
        capped = min(aspect, 18.0 / 29.0)
        scene_scale = capped / (5.625 / 10.0)
        unit = h * scene_scale / 10.0
        extra = max(0.0, capped - 5.625 / 10.0)
        origin = h * (0.5 + extra * (64.0 / 9.0) / 10.0)
        judge = origin + 2.4 * unit
        return scene_scale, unit, origin, judge, 5.625 * unit

    def judge_offset(self) -> float:
        _, _, _, judge, _ = self.metrics()
        return judge - self.height / 2.0

    def resize(self, width: int, height: int) -> None:
        if (width, height) == (self.width, self.height):
            return
        self.width = int(width)
        self.height = int(height)
        self.watch_card.resize(self.width, self.height)
        self.hud.resize(self.width, self.height)
        self.watch_card.reset_text()
        self.assets.reset_text()
        self._build_text()
        # Re-bake the HUD textures at the new size now, not on the first HUD
        # frame (that hitch looked like a low frame rate at the entrance).
        try:
            self.hud.prewarm(self.chart, self._watch_accent(0.0))
        except Exception:
            pass

    def update_meta(self) -> None:
        self.watch_card.reset_text()
        self.assets.reset_text()
        self._build_text()

    def _watch_meta(self) -> watch.WatchMeta:
        st = self.settings
        return watch.WatchMeta(
            title=st.song_name or self.chart.song_name or "Untitled",
            artist=st.artist,
            illustrator=st.illustrator or "",
            designer=st.designer or "",
        )

    def _watch_accent(self, tick: float) -> tuple[int, int, int]:
        state = self.chart.theme_state(tick)
        color = self.chart.theme_color(state.active_theme, 2)
        return (color.r, color.g, color.b)

    def _hud_alpha(self, raw_time: float) -> float:
        """Track/artist/copyright alpha after the intro card.

        The watermark lives in the prefab's ``AutoPlay`` panel, whose
        CanvasGroup has a 1.0 s InOutSine GroupFade.  The device capture shows
        it starting ~0.15 s before the white sheet is gone (the card hides the
        first part) and reaching full alpha ~0.85 s later.
        """
        st = self.settings
        if not st.intro:
            return 1.0
        card_end = -st.intro_preroll + watch.INTRO_TOTAL
        u = (raw_time - (card_end - 0.15)) / 1.0
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return 0.5 - 0.5 * math.cos(math.pi * u)

    def _build_text(self) -> None:
        s = self.width / 360.0
        song = self.settings.song_name or self.chart.song_name or "Untitled"
        artist = self.settings.artist or ""
        # The copyright mark is user-supplied; this is not an official
        # renderer, so nothing is baked into the code.
        copy_text = (self.settings.copyright_text or "").strip()
        self._text_lines = {
            # Measured on the device capture (1080x1920): song ink 161x44,
            # artist ink ~22 px, copyright ink ~23 px.  The PC prefab
            # (sharedassets8 AutoPlay/Panel) uses 53 / 24 / 24.
            "song": self.assets.text.add(song, max(1, round(17.8 * s))),
            "artist": self.assets.text.add(artist, max(1, round(8.0 * s))) if artist else None,
            "copy": self.assets.text.add(copy_text, max(1, round(8.0 * s))) if copy_text else None,
        }

    def _draw_text(
        self,
        line,
        x: float,
        ink_bottom: float,
        color: Color,
        align: str = "left",
        alpha: float = 1.0,
    ) -> None:
        if line is None:
            return
        # ``ink_bottom`` is the bottom of the glyph ink (the visual baseline
        # for text without descenders).
        top = ink_bottom - line.height + 1
        if align == "right":
            left = x - line.advance + line.x_off - 1
        else:
            left = x + line.x_off - 1
        self.mesh.quad(
            TEX_TEXT,
            left,
            top,
            left + line.width,
            top + line.height,
            _rgba(color, alpha),
            mode=1,
            uv=(line.u0, line.v0, line.u1, line.v1),
        )

    # ------------------------------------------------------------------
    # main frame

    def render(self, time_seconds: float) -> None:
        chart = self.chart
        st = self.settings
        w, h = self.width, self.height
        raw_time = time_seconds
        chart_time_raw = raw_time - chart.chart_delay - st.offset
        time_seconds = max(0.0, chart_time_raw)

        tick = chart.timing.seconds_to_tick(time_seconds)
        cam_scale = max(0.001, find_value(chart.camera_scale, tick))
        scale = cam_scale * st.revelation * st.scale
        cam_x = find_value(chart.camera_x, tick)
        flow = self.real_speed
        scene_scale, unit, origin_y, judge_y, field_w = self.metrics()
        half_w = w / 2.0

        # canvas state --------------------------------------------------
        cpos: dict[int, tuple[float, float]] = {}
        for canvas in chart.canvases:
            cpos[canvas.index] = (
                (canvas.x_at(tick) - cam_x) * scale * field_w,
                canvas.floor_at(time_seconds),
            )

        # theme / riztime state -----------------------------------------
        theme = chart.theme_state(tick)
        circle = (0.0, 0.0, 0.0, 0.0)
        if theme.transition is None:
            c0 = chart.theme_color(theme.active_theme, 0)
            c1 = chart.theme_color(theme.active_theme, 1)
            c0_in = c0
            c1_in = c1
        else:
            tr = theme.transition
            c0 = chart.theme_color(tr.outside_theme, 0)
            c1 = chart.theme_color(tr.outside_theme, 1)
            c0_in = chart.theme_color(tr.inside_theme, 0)
            c1_in = chart.theme_color(tr.inside_theme, 1)
            entering = tr.kind == "entering"
            cx = half_w
            # Mobile ChallengeTimeControl: enter from the bottom edge, exit
            # from the top edge; the mask sprite is a 512px circle at 30x.
            cy = float(h) if entering else 0.0
            max_r = (512.0 / 200.0) * 30.0 * unit
            circle = (cx, cy, max_r * tr.mask_progress, 1.0)

        effect_color = chart.effect_color(tick, theme)

        import os

        if os.environ.get("RIZREC_DEBUG_RENDER"):
            print(
                "render",
                "time", round(time_seconds, 4),
                "tick", round(tick, 4),
                "scale", round(scale, 4),
                "camx", round(cam_x, 4),
                "c0", c0,
                "c1", c1,
                "theme", theme.active_theme,
                "trans", theme.transition.kind if theme.transition else None,
                "size", (w, h),
                flush=True,
            )

        mesh = self.mesh
        import os as _os

        _split = bool(_os.environ.get("RIZREC_DEBUG_SPLIT"))

        def _flush(label: str) -> None:
            if _split:
                mesh.flush(w, h, circle)
                print("section", label, "err", self.ctx.error, flush=True)

        # 1. background -------------------------------------------------
        mesh.quad(0, 0.0, 0.0, w, h, _rgba(c0), _rgba(c0_in), 0)
        _flush("background")

        # 2. line bodies ------------------------------------------------
        line_w = max(0.5, 0.045 * unit * scale)
        ring_segments: list[tuple] = []
        top_cull = judge_y - h * 1.5
        # The game only draws judged-line geometry inside a floor window
        # around the judge line (~8 units ahead / 3 behind), evaluated per
        # canvas.  This is what keeps cross-canvas lines from leaking onto
        # the playfield.
        window_speed = ((st.speed - 1.0) / 9.0) * 7.0 + 6.0
        if not math.isfinite(window_speed) or window_speed <= 0.0:
            window_speed = self.real_speed * 10.0
        ahead = 8.0 / window_speed
        behind = 3.0 / window_speed
        cwin: dict[int, tuple[float, float]] = {}
        for canvas in chart.canvases:
            base = canvas.native_floor_at(time_seconds)
            cwin[canvas.index] = (base - behind, base + ahead)

        def _window_overlaps(pa, pb) -> bool | None:
            wa = cwin.get(pa.canvas)
            wb = cwin.get(pb.canvas)
            if wa is None or wb is None:
                return None
            start, end = pa.fp, pb.fp
            return (start <= wa[1] or end <= wb[1]) and (start >= wa[0] or end >= wb[0])

        def _flush_run(run: list[tuple], width: float | None = None,
                       cap_start: bool = True, cap_end: bool = True) -> None:
            if not run:
                return
            w = line_w if width is None else width
            if len(run) == 1:
                x, y, col = run[0]
                r = w * 0.5
                mesh.quad(TEX_CIRCLE, x - r, y - r, x + r, y + r, col, mode=1)
                return
            mesh.line_strip(
                [p[0] for p in run],
                [p[1] for p in run],
                [p[2] for p in run],
                w,
                cap_start=cap_start,
                cap_end=cap_end,
            )

        for line in chart.draw_lines:
            if len(line.points) < 2:
                continue
            overlay = color_at(line.line_color, tick)
            line_width = line_w
            pts = line.points
            proj = []
            colors = []
            raw_colors = []
            for pt in pts:
                cp = cpos.get(pt.canvas)
                if cp is None:
                    proj = None
                    break
                proj.append(
                    (
                        half_w + pt.x * scale * field_w + cp[0],
                        judge_y - (pt.fp - cp[1]) * h * scene_scale * flow * scale,
                    )
                )
                colors.append(pt.color.mix(overlay) if overlay is not None else pt.color)
                raw_colors.append(pt.color)
            if proj is None:
                continue
            ys = [p[1] for p in proj]
            if max(ys) < top_cull or min(ys) > h:
                continue

            outer = _window_overlaps(pts[0], pts[-1])
            is_two_dot = len(pts) == 2
            run: list[tuple] = []
            run_w = line_width
            run_cap_start = True
            prev_static = False
            prev_end_col = None
            prev_end_raw = None
            prev_ease = None

            for i in range(len(pts) - 1):
                p, q = pts[i], pts[i + 1]
                x1, y1 = proj[i]
                x2, y2 = proj[i + 1]
                lc1, lc2 = _rgba(colors[i]), _rgba(colors[i + 1])
                screen = not (max(y1, y2) < top_cull or min(y1, y2) > h)
                if screen:
                    # Rings keep the old screen-space culling; the floor
                    # window only gates the line bodies.
                    ring_segments.append((line, p, q, x1, y1, x2, y2))

                if outer is None:
                    visible = screen
                elif outer:
                    win = _window_overlaps(p, q)
                    visible = screen if win is None else (screen and win)
                else:
                    visible = False

                if not visible:
                    _flush_run(run, run_w)
                    run = []
                    prev_static = False
                    prev_end_col = None
                    prev_end_raw = None
                    continue

                # A true chart "dot": the same addressed point twice.
                if (
                    is_two_dot
                    and p.canvas == q.canvas
                    and p.time == q.time
                    and p.x == q.x
                    and p.fp == q.fp
                    and colors[i].r == colors[i + 1].r
                    and colors[i].g == colors[i + 1].g
                    and colors[i].b == colors[i + 1].b
                    and colors[i].a == colors[i + 1].a
                ):
                    _flush_run(run, run_w)
                    run = []
                    prev_static = False
                    prev_end_raw = None
                    r = line_w * 0.5
                    mesh.quad(TEX_CIRCLE, x1 - r, y1 - r, x1 + r, y1 + r, lc1, mode=1)
                    continue

                # Only backdrop walls are truly huge; Quaoar runs on canvases
                # whose normal floors are >1000, so span (not absolute fp) decides.
                huge = abs(q.fp - p.fp) > 100.0
                if huge:
                    _flush_run(run, run_w)
                    run = []
                    prev_static = False
                    # Static backdrop: clip to the screen; the rear cover
                    # provides the real fade-out at the bottom.
                    lo = max(min(y1, y2), top_cull)
                    hi = min(max(y1, y2), float(h))
                    if hi - lo < 0.5 or y1 == y2:
                        continue
                    t_lo = (lo - y1) / (y2 - y1)
                    t_hi = (hi - y1) / (y2 - y1)
                    if t_lo > t_hi:
                        t_lo, t_hi = t_hi, t_lo
                    col_lo = colors[i].lerp(colors[i + 1], max(0.0, min(1.0, t_lo)))
                    col_hi = colors[i].lerp(colors[i + 1], max(0.0, min(1.0, t_hi)))
                    if overlay is not None:
                        col_lo = col_lo.mix(overlay)
                        col_hi = col_hi.mix(overlay)
                    x_lo = x1 + (x2 - x1) * t_lo
                    x_hi = x1 + (x2 - x1) * t_hi
                    mesh.line_quad_grad(
                        x_lo, lo, x_hi, hi, line_width, _rgba(col_lo), _rgba(col_hi)
                    )
                    continue

                static = p.canvas == q.canvas
                seg_w = line_width
                # Native groups positions by the authored 8-bit point colour
                # (LineRenderer only carries start/end colours), not by the
                # theme-tinted colour used for drawing.
                merge = (
                    bool(run)
                    and prev_static
                    and static
                    and seg_w == run_w
                    and prev_end_raw == p.color
                    and p.color == q.color
                )
                affine = p.ease == 0 or x1 == x2 or p.ease in (13, 14)
                frm = ease(p.ease, 0.0)
                to = ease(p.ease, 1.0)
                px = x1 + (x2 - x1) * frm
                endx = x1 + (x2 - x1) * to

                # easeZero/easeOne instant jumps are a separate thin
                # horizontal line in the game, not part of the 0.045 ribbon.
                # Draw that jumper from the previous body's x to this body's
                # x, then start a fresh body at the new x so the ribbon never
                # spans the jump.  This also works when the previous body was
                # culled, which is the first seam of a staircase section.
                src_x = None
                if prev_ease == 13 and i > 0:
                    src_x = proj[i - 1][0]
                elif p.ease == 14:
                    src_x = x1
                if src_x is not None and abs(px - src_x) > 0.5 and 0.0 <= y1 <= h:
                    mesh.line_quad(
                        src_x, y1, px, y1, line_w * INSTANT_LINE_RATIO, lc1
                    )
                    if run:
                        _flush_run(run, run_w, cap_start=run_cap_start, cap_end=False)
                    run_w = seg_w
                    run = [(px, y1, lc1)]
                    run_cap_start = False
                    merge = True

                # The game's SetDynamicLines tessellates each chart segment by
                # length (stride 20/10/5/2/1).  Unity then builds one
                # cross-section per submitted position, so the chord lengths
                # decide the join normal at every seam.  Reproduce the same
                # positions instead of resampling at a fixed 100 steps.
                stride = _curve_stride(p, q, window_speed) if (merge or not affine) else 1
                if affine and not merge:
                    # A one-segment renderer is a straight leg (native
                    # fillStraightJudgedLine): easeZero/One contribute no
                    # interior positions of their own.
                    _flush_run(run, run_w, cap_start=run_cap_start)
                    run_w = seg_w
                    run = [(px, y1, lc1), (endx, y2, lc2)]
                    run_cap_start = True
                elif affine:
                    # Merged run: keep native-length chords at both sides of
                    # every join, so a shifted cross-section cannot taper an
                    # entire straight leg.
                    if abs(run[-1][0] - px) > 1e-9 or abs(run[-1][1] - y1) > 1e-9:
                        run.append((px, y1, lc1))
                    last = ((100 - 1) // stride) * stride
                    ex = (to - frm) * (x2 - x1)
                    step = stride
                    while step <= last:
                        if step > 2 * stride and step < last - stride:
                            step = last - stride
                        t = step / 100.0
                        run.append(
                            (
                                px + ex * t,
                                y1 + (y2 - y1) * t,
                                _rgba(colors[i].lerp(colors[i + 1], t)),
                            )
                        )
                        step += stride
                    run.append((endx, y2, lc2))
                else:
                    samples = []
                    for step in range(stride, 100, stride):
                        t = step / 100.0
                        samples.append(
                            (
                                x1 + (x2 - x1) * ease(p.ease, t),
                                y1 + (y2 - y1) * t,
                                _rgba(colors[i].lerp(colors[i + 1], t)),
                            )
                        )
                    samples.append((endx, y2, lc2))
                    if merge:
                        run.extend(samples)
                    else:
                        _flush_run(run, run_w, cap_start=run_cap_start)
                        run_w = seg_w
                        run = [(px, y1, lc1), *samples]
                        run_cap_start = True
                prev_static = static
                prev_end_col = lc2
                prev_end_raw = q.color
                prev_ease = p.ease

            _flush_run(run, run_w, cap_start=run_cap_start)

        # 3. rear cover (bottom band centred on the judge line) ---------
        _flush("lines")
        self._draw_cover_band(-2.4, False, c0, c0_in)
        _flush("rearcover")

        # 4. hold bodies -------------------------------------------------
        # The prefab stacks the black outline strips (``HoldLineRenderer/
        # BlackLine 0..3``) *below* the theme-coloured body (``Color 1/2``).
        # When two holds sit close together their colours fuse (the solid
        # part is alpha 1, so overlaps replace instead of stacking) and the
        # shared black edges disappear underneath the fill.  Two passes.
        hold_bodies: list[tuple] = []
        for note in chart.notes:
            if note.kind != NOTE_HOLD:
                continue
            if tick > note.end_tick:
                continue
            hit = tick >= note.tick
            if hit:
                x = half_w + self._line_x(note, tick, cpos, scale, field_w)
                y = 0.0
            else:
                x = half_w + self._note_base_x(note, cpos, scale, field_w)
                cp = cpos.get(note.start_canvas)
                if cp is None:
                    continue
                y = -(note.start_floor - cp[1]) * h * scene_scale * flow * scale
            # The top cover (drawn later) fades notes in; do not clip early.
            body_fade = 1.0
            if not hit and y < -h * 1.5:
                continue
            ce = cpos.get(note.end_canvas)
            if ce is None:
                continue
            end_y = -(note.end_floor - ce[1]) * h * scene_scale * flow * scale
            if abs(end_y - y) <= 0.05:
                continue
            body_w = 0.192 * unit * scale
            edge_w = body_w * 0.25
            hold_bodies.append(
                (
                    x,
                    judge_y + y,
                    judge_y + end_y,
                    body_w,
                    edge_w,
                    _rgba(c1, body_fade),
                    _rgba(c1_in, body_fade),
                    body_fade,
                )
            )

        def _hold_fill(rec) -> None:
            x, head, tail, body_w, _edge_w, col, col_in, _fade = rec
            if tail < head:
                y0, v0, y1, v1 = tail, 0.0, head, 0.125
            else:
                y0, v0, y1, v1 = head, 0.125, tail, 0.0
            mesh.quad(
                TEX_HOLD,
                x - body_w * 0.5,
                y0,
                x + body_w * 0.5,
                y1,
                col,
                col_in,
                mode=1,
                uv=(0.0, v0, 1.0, v1),
            )

        def _hold_edges(rec) -> None:
            x, head, tail, body_w, edge_w, _col, _col_in, fade = rec
            black = (0.0, 0.0, 0.0, fade)
            mid = (head + tail) * 0.5
            for edge_x in (x - body_w * 0.5 - edge_w, x + body_w * 0.5):
                # The head half is solid black; the tail half follows the strip.
                lo, hi = (head, mid) if head <= mid else (mid, head)
                if hi - lo > 0.05:
                    mesh.quad(TEX_WHITE, edge_x, lo, edge_x + edge_w, hi, black)
                if abs(tail - mid) > 0.05:
                    if tail < mid:
                        y0, v0, y1, v1 = tail, 0.0, mid, 0.125
                    else:
                        y0, v0, y1, v1 = mid, 0.125, tail, 0.0
                    mesh.quad(
                        TEX_HOLD,
                        edge_x,
                        y0,
                        edge_x + edge_w,
                        y1,
                        black,
                        mode=1,
                        uv=(0.0, v0, 1.0, v1),
                    )

        # 4a. black outline strips for every hold ...
        for rec in hold_bodies:
            _hold_edges(rec)
        # 4b. ... then the coloured fills on top, so an edge crossing a
        # neighbouring body ends up under the colour (device behaviour).
        for rec in hold_bodies:
            _hold_fill(rec)

        # 6. note heads --------------------------------------------------
        # The prefab's visible layers come from a mask pass plus a fill pass:
        # the mask layers (tap/hold NoteBackground, the hold black ring, the
        # drag circle) are written first and the theme-coloured fills on top,
        # so an outline that crosses a neighbouring note ends up underneath
        # its colour — the same rule as the hold body (§19).  Across note
        # types the stacks do not interleave: drag is lowest, then the whole
        # hold stack, then the whole tap stack (device: every tap layer sits
        # above hold).
        _flush("holds")
        drags: list[tuple] = []
        hold_bgs: list[tuple] = []
        hold_blacks: list[tuple] = []
        hold_fills: list[tuple] = []
        tap_bgs: list[tuple] = []
        tap_fills: list[tuple] = []
        for note in chart.notes:
            canvas_obj = chart.canvas_by_index.get(note.start_canvas)
            if canvas_obj is not None and tick < canvas_obj.activation_tick:
                continue
            if note.kind != NOTE_HOLD:
                if tick >= note.tick:
                    continue
            else:
                if time_seconds >= note.end_sec + 0.25:
                    continue
            hit = tick >= note.tick
            if note.kind == NOTE_HOLD and hit:
                x = half_w + self._line_x(note, tick, cpos, scale, field_w)
            else:
                x = half_w + self._note_base_x(note, cpos, scale, field_w)
            if hit:
                y = 0.0
            else:
                cp = cpos.get(note.start_canvas)
                if cp is None:
                    continue
                y = -(note.start_floor - cp[1]) * h * scene_scale * flow * scale
            if y < -h * 1.5:
                continue
            # No manual top fade: the 1-unit top cover gradient does it.
            fade = 1.0
            head_scale = 1.0
            if note.kind == NOTE_HOLD and time_seconds > note.end_sec:
                u = (time_seconds - note.end_sec) / 0.25
                head_scale = 1.0 - min(1.0, u) * min(1.0, u)
                if head_scale <= 0.0:
                    continue
            sy = judge_y + y
            note_unit = unit * scale * head_scale

            if note.kind == NOTE_DRAG:
                drags.append((x, sy, 0.60 * note_unit, fade))
            elif note.kind == NOTE_HOLD:
                hold_bgs.append((x, sy, 0.48 * note_unit, fade))
                hold_blacks.append((x, sy, 0.3264 * note_unit, fade))
                hold_fills.append((x, sy, 0.2304 * note_unit, fade, None))
            else:
                tap_bgs.append((x, sy, 0.48 * note_unit, fade))
                tap_fills.append((x, sy, 0.2664 * note_unit, fade, (c1, c1_in)))

        def _head_quad(tid: int, x, sy, d, color, color2=None, mode: int = 1) -> None:
            mesh.quad(
                tid,
                x - d * 0.5,
                sy - d * 0.5,
                x + d * 0.5,
                sy + d * 0.5,
                color,
                color2,
                mode=mode,
            )

        # 6a. drag rings: lowest note layer (a hold head covers them).
        for x, sy, d, fade in drags:
            _head_quad(TEX_DRAG, x, sy, d, (1.0, 1.0, 1.0, fade), mode=0)

        # 6b. hold stack, bottom to top: white NoteBackground -> black ring ->
        # white core.  The whole stack stays below every tap layer.
        def _draw_hold_stack() -> None:
            for x, sy, d, fade in hold_bgs:
                _head_quad(TEX_NOTEBG, x, sy, d, (1.0, 1.0, 1.0, fade), mode=0)
            for x, sy, d, fade in hold_blacks:
                _head_quad(TEX_CIRCLE, x, sy, d, (0.0, 0.0, 0.0, fade))
            for x, sy, d, fade, _col in hold_fills:
                _head_quad(TEX_CIRCLE, x, sy, d, (1.0, 1.0, 1.0, fade))

        # 6c. tap stack, above the hold stack: white background -> theme core.
        def _draw_tap_stack() -> None:
            for x, sy, d, fade in tap_bgs:
                _head_quad(TEX_NOTEBG, x, sy, d, (1.0, 1.0, 1.0, fade), mode=0)
            for x, sy, d, fade, col in tap_fills:
                _head_quad(TEX_CIRCLE, x, sy, d, _rgba(col[0], fade), _rgba(col[1], fade))

        # Cross-type stacks do not interleave: drag < hold < tap, so a tap
        # always wins over a hold (device: every tap layer is above hold).
        _draw_hold_stack()
        _draw_tap_stack()
        _flush("notes")

        # 7. judge rings (above the notes) --------------------------------
        ring_r = 0.3125 * unit * scale
        for line, p, q, x1, y1, x2, y2 in ring_segments:
            if p.time == q.time:
                if not (p.time <= tick < p.time + 0.4):
                    continue
                prog = (tick - p.time) / 0.4
                rx = x1 + (x2 - x1) * prog
                alpha = max(0.0, 1.0 - prog)
            elif p.time <= tick < q.time:
                if x1 == x2:
                    rx = x1
                elif y1 == y2:
                    prog = (tick - p.time) / (q.time - p.time)
                    rx = x1 + (x2 - x1) * ease(p.ease, prog)
                else:
                    # The ring sits where the line crosses the judge line.
                    # `proj` is in screen space; convert to judge-relative.
                    rx = x1 + (x2 - x1) * ease(p.ease, (judge_y - y1) / (y2 - y1))
                alpha = 1.0
            else:
                continue
            ring_col = color_at(line.ring_color, tick)
            if ring_col is None:
                continue
            line_col = color_at(line.line_color, tick)
            if line_col is not None and not (ring_col.a == 0 and _is_opaque_white(line_col)):
                ring_col = ring_col.mix(line_col)
            if ring_col.a == 0:
                continue
            mesh.quad(
                TEX_RING,
                rx - ring_r,
                judge_y - ring_r,
                rx + ring_r,
                judge_y + ring_r,
                _rgba(ring_col, alpha),
                mode=1,
            )
        _flush("rings")

        # 8. top / bottom covers ----------------------------------------
        self._draw_cover_band(3.6, True, c0, c0_in)
        self._draw_cover_band(-3.4, False, c0, c0_in)
        _flush("covers")

        # 9. side covers -------------------------------------------------
        self._draw_side(c0, c0_in)
        _flush("sides")

        # 10. hit effects ------------------------------------------------
        _flush("covers")
        lo = max(0, self._note_index_at(time_seconds - HITFX_DURATION - 0.01))
        for i in range(lo, len(chart.notes)):
            note = chart.notes[i]
            if note.start_sec > time_seconds + 1e-6:
                break
            if note.start_sec >= time_seconds - HITFX_DURATION:
                ex = self._hit_x(note, note.tick)
                self._draw_hitfx(note, ex, time_seconds - note.start_sec, tick, effect_color)
        # Holds are looked up by end time; their start may be far outside the
        # effect window (long holds).
        end_times = chart.hold_end_secs
        h_lo = bisect.bisect_left(end_times, time_seconds - HITFX_DURATION)
        h_hi = bisect.bisect_right(end_times, time_seconds)
        for _, note in chart.hold_ends[h_lo:h_hi]:
            ex = self._hit_x(note, note.end_tick)
            self._draw_hitfx(note, ex, time_seconds - note.end_sec, tick, effect_color, tail=True)

        # 11. watermark ---------------------------------------------------
        _flush("hitfx")
        hud_alpha = self._hud_alpha(raw_time)
        # The track/artist/copyright block is the prefab's ``AutoPlay`` panel:
        # normal play (``full`` HUD) does not draw it at all (verified on the
        # 2026-11-25 device recordings).
        if self.settings.show_watermark and st.hud_mode != "full" and hud_alpha > 0.002:
            s = w / 360.0
            pad = 22.0 * s
            bottom = h - 36.67 * s
            self._draw_text(self._text_lines.get("song"), pad, bottom, effect_color, "left",
                            0.95 * hud_alpha)
            self._draw_text(
                self._text_lines.get("artist"), pad, h - 57.0 * s, effect_color, "left",
                0.95 * hud_alpha
            )
            self._draw_text(
                self._text_lines.get("copy"),
                w - 20.3 * s,
                h - 34.3 * s,
                Color(0, 0, 0, 255),
                "right",
                0.298 * hud_alpha,
            )

        mesh.flush(w, h, circle)
        if _split:
            print("section final err", self.ctx.error, flush=True)

        # 12. normal-play HUD (only outside auto) ------------------------
        card = self.card
        card.begin()
        if st.hud_mode == "full":
            # t=0 is the moment the intro card's white sheet is gone: the
            # slide starts there (from +190), while the CanvasGroup fade
            # started 0.35 s earlier behind the card (hud_fade handles it).
            # Without intro the HUD is simply static.
            if st.intro:
                # The entrance belongs to the level-start timeline (card and
                # music), not to chart time: a chart offset must not shift it.
                hud_t = raw_time - (-st.intro_preroll + watch.INTRO_TOTAL)
            else:
                hud_t = 10.0
            hud_alpha = hud.hud_fade(hud_t)
            if hud_alpha > 0.002:
                self.hud.prepare(chart)
                y_off = hud.hud_enter_offset(hud_t)
                self.hud.draw(card, time_seconds, self._watch_accent(tick), hud_alpha,
                              y_off=y_off, anim_t=hud_t)
        # Watch card (level intro / outro overlay), on top of the HUD.
        if st.intro:
            elapsed = raw_time + st.intro_preroll
            if 0.0 <= elapsed < watch.INTRO_CONTENT_TOTAL:
                self.watch_card.draw_intro(
                    card, elapsed, self._watch_meta(), self._watch_accent(0.0),
                    st.cover_path, show_pill=st.hud_mode == "auto",
                )
        if st.outro and st.outro_start is not None:
            elapsed = raw_time - st.outro_start
            if 0.0 <= elapsed < watch.OUTRO_TOTAL:
                self.watch_card.draw_outro(
                    card, elapsed, self._watch_accent(
                        chart.timing.seconds_to_tick(max(0.0, st.outro_start))
                    ),
                    st.cover_path,
                )
        card.flush(w, h)

    # ------------------------------------------------------------------
    # drawing helpers

    def _draw_cover_band(self, y_world: float, top: bool, c: Color, c_in: Color) -> None:
        """Mobile vertical cover: 1-unit ``side.png`` band + solid fill beyond it.

        ``start = originY - y * unit - unit / 2``; the band is opaque at the
        outer edge and fades towards the judge line.
        """
        w, h = self.width, self.height
        _, unit, origin, _, _ = self.metrics()
        start = origin - y_world * unit - unit * 0.5
        end = start + unit
        mesh = self.mesh
        if top:
            if start > 0.0:
                mesh.quad(TEX_WHITE, 0.0, 0.0, w, start, _rgba(c), _rgba(c_in))
            mesh.quad(
                TEX_SIDE,
                0.0,
                start,
                w,
                end,
                _rgba(c),
                _rgba(c_in),
                mode=1,
                uv=(0.0, 1.0, 1.0, 0.0),
            )
        else:
            if end < h:
                mesh.quad(TEX_WHITE, 0.0, end, w, h, _rgba(c), _rgba(c_in))
            mesh.quad(
                TEX_SIDE,
                0.0,
                start,
                w,
                end,
                _rgba(c),
                _rgba(c_in),
                mode=1,
                uv=(0.0, 0.0, 1.0, 1.0),
            )

    def _draw_side(self, c: Color, c_in: Color) -> None:
        w, h = self.width, self.height
        _, unit, _, _, _ = self.metrics()
        # Stage 5 side textures sit 2.3125 scene units from the centre.
        left = w * 0.5 - 2.3125 * unit - unit * 0.5
        right = w - left - unit
        mesh = self.mesh
        if left > 0.0:
            mesh.quad(TEX_WHITE, 0.0, 0.0, left, h, _rgba(c), _rgba(c_in))
            mesh.quad(TEX_WHITE, w - left, 0.0, w, h, _rgba(c), _rgba(c_in))
        # left: opaque at the outer edge, transparent towards the centre
        mesh.quad4(
            TEX_SIDE,
            ((left, 0.0), (left + unit, 0.0), (left + unit, h), (left, h)),
            ((0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            _rgba(c),
            _rgba(c_in),
            mode=1,
        )
        # right
        mesh.quad4(
            TEX_SIDE,
            ((right, 0.0), (right + unit, 0.0), (right + unit, h), (right, h)),
            ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
            _rgba(c),
            _rgba(c_in),
            mode=1,
        )

    def _note_index_at(self, seconds: float) -> int:
        import bisect

        return bisect.bisect_left(self.chart.note_starts, seconds)

    def _note_base_x(self, note: NoteState, cpos: dict, scale: float, field_w: float) -> float:
        line = self.chart.lines[note.line_index]
        p0 = line.points[note.p0]
        p1 = line.points[note.p1]
        cp0 = cpos.get(p0.canvas, (0.0, 0.0))
        cp1 = cpos.get(p1.canvas, (0.0, 0.0))
        width = scale * field_w
        height = self.height * self.metrics()[0] * self.real_speed * scale
        x1 = p0.x * width + cp0[0]
        x2 = p1.x * width + cp1[0]
        y1 = -(p0.fp - cp0[1]) * height
        y2 = -(p1.fp - cp1[1]) * height
        y = -(note.start_floor - cp0[1]) * height
        span = y2 - y1
        progress = (y - y1) / span if span != 0.0 else note.x_progress
        return x1 + (x2 - x1) * ease(p0.ease, progress)

    def _line_x(self, note: NoteState, target_tick: float, cpos: dict, scale: float, field_w: float) -> float:
        line = self.chart.lines[note.line_index]
        i = line.point_index(target_tick)
        p0 = line.points[i]
        p1 = line.points[i + 1] if i + 1 < len(line.points) else p0
        cp0 = cpos.get(p0.canvas, (0.0, 0.0))
        cp1 = cpos.get(p1.canvas, (0.0, 0.0))
        width = scale * field_w
        height = self.height * self.metrics()[0] * self.real_speed * scale
        x1 = p0.x * width + cp0[0]
        x2 = p1.x * width + cp1[0]
        if x1 == x2:
            return x1
        y1 = -(p0.fp - cp0[1]) * height
        y2 = -(p1.fp - cp1[1]) * height
        if y1 == y2:
            span = p1.time - p0.time
            progress = (target_tick - p0.time) / span if span > 0 else 0.0
        else:
            # Intersection with the judge line (y = 0), like the game.
            progress = -y1 / (y2 - y1)
        return x1 + (x2 - x1) * ease(p0.ease, progress)

    def _state_at(self, target_tick: float) -> tuple[dict, float]:
        """Canvas screen positions and camera scale frozen at a tick."""
        chart = self.chart
        st = self.settings
        w = self.width
        sec = chart.timing.tick_to_seconds(target_tick)
        cam_scale = max(0.001, find_value(chart.camera_scale, target_tick)) * st.revelation * st.scale
        cam_x = find_value(chart.camera_x, target_tick)
        _, _, _, _, field_w = self.metrics()
        cpos = {}
        for canvas in chart.canvases:
            cpos[canvas.index] = (
                (canvas.x_at(target_tick) - cam_x) * cam_scale * field_w,
                canvas.floor_at(sec),
            )
        return cpos, cam_scale

    def _hit_x(self, note: NoteState, target_tick: float) -> float:
        cpos, scale = self._state_at(target_tick)
        _, _, _, _, field_w = self.metrics()
        return self.width / 2.0 + self._line_x(note, target_tick, cpos, scale, field_w)

    def _draw_hitfx(
        self,
        note: NoteState,
        x: float,
        elapsed: float,
        tick: float,
        color: Color,
        tail: bool = False,
    ) -> None:
        if elapsed < 0.0 or elapsed > HITFX_DURATION:
            return
        w, h = self.width, self.height
        _, _, _, judge_y, _ = self.metrics()
        mesh = self.mesh
        base_ppu = h / 10.0 * self.settings.scale
        unit = base_ppu * hitfx.PREFAB_SCALE
        judge_tick = note.end_tick if tail else note.tick
        emission = note.index * 2 + (1 if tail else 0)
        col = _rgba(color)

        # Particles and the perfect spray sit below the sprite.
        for p in hitfx.create_particles(judge_tick, emission):
            age = elapsed - p["delay"]
            if age < 0.0 or age >= hitfx.PARTICLE_LIFETIME:
                continue
            progress = age / hitfx.PARTICLE_LIFETIME
            distance = (
                p["start_radius"] + hitfx.particle_travel(age, p["launch_speed"])
            ) * unit
            px = x + distance * p["dx"]
            py = judge_y + distance * p["dy"]
            d = p["diameter"] * unit * hitfx.particle_size(progress)
            alpha = col[3] * hitfx.particle_alpha(progress)
            if d <= 0.05 or alpha <= 0.0:
                continue
            mesh.quad(
                TEX_CIRCLE,
                px - d * 0.5,
                py - d * 0.5,
                px + d * 0.5,
                py + d * 0.5,
                (col[0], col[1], col[2], alpha),
                mode=1,
            )

        if any(ch.start < judge_tick <= ch.end for ch in self.chart.challenges):
            spray_ppu = base_ppu * hitfx.PERFECT_SPRAY_SCALE
            for p in hitfx.create_spray(judge_tick):
                progress = (elapsed - p["delay"]) / hitfx.PERFECT_LIFETIME
                if progress < 0.0 or progress >= 1.0:
                    continue
                travel = hitfx.perfect_sample(progress)
                distance = (
                    p["start_radius"]
                    + p["speed"] * hitfx.PERFECT_LIFETIME * travel
                ) * spray_ppu
                size = hitfx.spray_size(progress) * p["diameter"] * spray_ppu
                alpha = min(1.0, 2.0 * (1.0 - progress)) * col[3]
                radius = max(0.5, size * 0.5)
                if alpha <= 0.0:
                    continue
                px = x + p["x_dir"] * distance
                py = judge_y + p["y_dir"] * distance
                mesh.quad(
                    TEX_CIRCLE,
                    px - radius,
                    py - radius,
                    px + radius,
                    py + radius,
                    (col[0], col[1], col[2], alpha),
                    mode=1,
                )

        # HitFX atlas: frames 1..40, 7 columns, 256 px cells (1792x1536).
        frame = min(hitfx.FRAME_LAST, hitfx.FRAME_FIRST + int(elapsed * hitfx.FPS))
        c = frame % hitfx.ATLAS_COLUMNS
        r = frame // hitfx.ATLAS_COLUMNS
        cell_u = hitfx.ATLAS_FRAME_SIZE / 1792.0
        cell_v = hitfx.ATLAS_FRAME_SIZE / 1536.0
        u0 = c * cell_u
        v0 = r * cell_v
        size = hitfx.SPRITE_WORLD_SIZE * hitfx.PREFAB_SCALE * base_ppu
        mesh.quad(
            TEX_HITFX,
            x - size * 0.5,
            judge_y - size * 0.5,
            x + size * 0.5,
            judge_y + size * 0.5,
            _rgba(color),
            mode=1,
            uv=(u0, v0, u0 + cell_u, v0 + cell_v),
        )
