"""In-game (non-auto) HUD: life bar, star row and combo.

The layout comes from the PC prefab (``sharedassets8``) and was checked
against the desktop/device recordings added on 2026-11-25:

* a capsule life bar across the top (white body + theme fill + pale empty part);
* five hollow dots at the top-left (the performance-rating display);
* ``combo N`` at the top-right, right aligned at the combo canvas edge —
  lowercase label plus the **weighted** combo (C / 2C-5 / 3C-13 / 4C-24).

A recording renders a perfect play, so the run simply counts the judgements
that have already happened.  Every note is one judgement at ``start_sec``;
a hold is **two** — one at the head and one when the body ends
(``end_sec``), matching the game.  Accuracy follows the game's own formula
(``NormalModeInLevelController.GetJudge``):
``comboScore(passed)/comboScore(total) + 0.2 * passed/total`` in 0..1.2.
"""

from __future__ import annotations

import bisect
import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..chart.chart import Chart, NOTE_HOLD
from .assets import AssetStore, Texture

# design-space layout (1080x1920), from the PC prefab (sharedassets8) and the
# device recording: LifeBar rect 968x38 at y 82; white body inset 2.5 and
# theme fill / pale cover inset 6.0 (white border 3.5px, shadow ring 2.5px);
# the bar is a capsule.  Star rings are LeanCircle size 45.504 / thickness 5.
BAR_X = 56.0
BAR_Y = 82.0
BAR_W = 968.0
BAR_H = 38.0
BAR_RADIUS = 19.0
BAR_BODY_INSET = 2.5      # white body; leaves a ~2.5px shadow ring
BAR_ACCENT_INSET = 6.0    # theme fill / pale cover; white border 3.5px
BAR_COVER_ALPHA = 0.82

STAR_FIRST_X = 86.0      # first rating dot centre (prefab + recording)
STAR_DX = 58.46
STAR_CY = 174.0
STAR_D = 45.504          # LeanCircle rect size = outer diameter
STAR_STROKE = 5.0        # prefab LeanCircle thickness
STAR_ALPHA = 0.239       # empty ring: #242424 at 23.9%
STAR_INK = (36, 36, 36)  # 0.14 grey
# A dot fades in when it lights up (user: "判定评级出来时是有渐显的").
DOT_FADE = 0.20
# User: "渐显启动晚一点" then "启动再慢点，我说的是启动后延不是整体后延" - the
# ramp is not shifted later as a whole; its *onset* is soft (ease-in), so the
# dot stays almost hollow for the first half of the fade and fills sharply.
DOT_EASE = 2.0

COMBO_RIGHT = 1019.0
COMBO_BASELINE = 203.0
COMBO_NUMBER = 65.0
COMBO_LABEL = 44.0
COMBO_GAP = 13.0
COMBO_LABEL_TEXT = "combo"

TEXT_FACE = (73, 73, 73)  # TMP m_fontColor 0.286
TEXT_STROKE = (255, 255, 255)
STROKE_RATIO = 0.067      # white outline (TMP material _OutlineWidth 0.3)
DILATE_RATIO = 0.008      # face +0.5px (outline stays 6.7%)
GLYPHS = "0123456789.%abcdefghijklmnopqrstuvwxyz"

# top-centre "pull down to pause" handle: HintBar 45x6 at y=130 (black 20%)
HINT_W = 45.0
HINT_H = 6.0
HINT_Y = 130.0
HINT_ALPHA = 0.20
HINT_TEXT = "向下滑动以暂停"
HINT_DELAY = 2.0 / 60.0   # the unfold starts ~2 frames later (user feedback)

# The handle "unfolds" into the hint text and folds back into the handle.
# (t, text centre y in design px, scale, alpha) measured from the recording.
HINT_MORPH = (
    (0.65, 150.0, 0.35, 0.30),
    (0.72, 169.0, 0.40, 0.55),
    (0.80, 193.0, 0.62, 0.60),
    (0.88, 200.0, 0.89, 1.00),
    (0.97, 187.0, 0.95, 1.00),
    (1.15, 165.0, 0.99, 1.00),
    (1.32, 158.0, 1.00, 1.00),
    (2.13, 158.0, 1.00, 1.00),
    (2.17, 150.0, 0.80, 0.80),
    (2.20, 142.0, 0.50, 0.50),
    (2.25, 132.0, 0.25, 0.10),
    (2.30, 130.0, 0.20, 0.00),
)


def hint_text_state(t: float) -> tuple[float, float, float]:
    """Text (alpha, scale, centre y) of the unfolding pause hint."""
    if t <= HINT_MORPH[0][0] or t >= HINT_MORPH[-1][0]:
        return (0.0, 0.2, HINT_Y)
    for (t0, y0, s0, a0), (t1, y1, s1, a1) in zip(HINT_MORPH, HINT_MORPH[1:]):
        if t <= t1:
            k = (t - t0) / max(1e-6, t1 - t0)
            k = k * k * (3.0 - 2.0 * k)
            return (a0 + (a1 - a0) * k, s0 + (s1 - s0) * k, y0 + (y1 - y0) * k)
    return (0.0, 0.2, HINT_Y)


def hint_bar_alpha(t: float) -> float:
    if t < HINT_MORPH[0][0] - 0.13 or t > 2.27:
        return 1.0
    if t < HINT_MORPH[0][0]:
        return 1.0 - (t - (HINT_MORPH[0][0] - 0.13)) / 0.13
    return 0.0

# HUD entrance (prefab `NormalModeController/Canvas`):
#   * RectTransform LocalMove 0.75 s, AnimationCurve (weighted tangents
#     470.3*0.0010638 ~ 0.5 -> 0);
#   * CanvasGroup GroupFade 1.0 s InOutSine.
# The slide itself starts when the sheet reaches alpha 0.  Measured offsets
# (design px, t from that moment) are 99 / 56 / 27 / 10 / 1 at 0.10 / 0.20 /
# 0.30 / 0.40 / 0.50 s; a least-squares fit of ``A*(1-t/T)^p`` over them gives
# T=0.795 s, p=4.4 (rmse ~3.5 px).  A cubic (p=3) is noticeably too slow in the
# middle and too fast at the end.
# User iterations on the slide: "有点慢" -> "还是慢了" -> "应该是前快后慢".
# The capture fit (T=0.795, p=4.4) is a shot-then-crawl: the first frame moves
# 22 px, half the distance is gone in 0.1 s, and the last 10 px take 0.3 s, so
# it reads as a snap plus a stall instead of a decelerating motion.  Now it is
# a real ease-out that also arrives sooner:
#   offsets/t  -  190 @0 / 105 @0.1 / 48 @0.2 / 15 @0.3 / 1 @0.4 / 0 @0.46
#   speed/t    -  850 / 570 / 330 / 137 px/s (monotone, real braking in the tail)
# p: 2.2 -> 1.8 ("后半段快一点点") -> T shortened four times ("运动再快点" x2) ->
# p back up to 2.4 ("后半段的前半部分快了，后半部分需要减速") -> T back to 0.46
# ("运动还是快了": p=2.4 over 0.39 s made the first frames ~1058 px/s).  The
# tail now brakes instead of cruising in.  HUD_DIP is a separate action.
HUD_SLIDE_T = 0.46
HUD_SLIDE_P = 2.4
HUD_SLIDE_A = 190.0
HUD_FADE_LEAD = -0.35     # group fade starts this much before the slide
HUD_FADE_T = 1.0          # GroupFade duration (InOutSine)
HUD_DIP = (
    (0.60, 0.0), (0.70, 5.5), (0.80, 8.3), (0.95, 5.5), (1.15, 2.0),
    (1.35, 0.0),
)
# The life bar gets its own, longer fade: the user wants it fully opaque only
# when the dip has swung back up to the top ("渐显是到回弹最顶端时才完全显示"),
# i.e. at HUD_DIP[-1][0], instead of at the end of the 1.0 s group fade.
# The bar's own fade ends when the slide lands ("改成到位后就满，不等到回弹最顶端"),
# i.e. at HUD_SLIDE_T instead of the dip's top.  Still starts with the group
# (HUD_FADE_LEAD).  Exponent knob: bigger = dimmer early / steeper finish.
HUD_BAR_FADE_T = HUD_SLIDE_T - HUD_FADE_LEAD       # 0.7433 s
HUD_BAR_FADE_POW = 1.9   # re-tuned when the slide went back to 0.46 s (longer window)


def _sample(points, t: float) -> float:
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t <= t1:
            k = (t - t0) / max(1e-6, t1 - t0)
            k = k * k * (3.0 - 2.0 * k)
            return v0 + (v1 - v0) * k
    return points[-1][1]


def hud_slide_offset(t: float) -> float:
    """The entrance slide alone (no dip): 190 design px -> 0.

    The slide is already running when the HUD becomes visible: the game starts
    it under the fading watch card, so at ``t=0`` (white sheet gone) the HUD is
    at ``HUD_SLIDE_A`` and still moving.  The curve is the measured fit, just
    extended back to ``HUD_FADE_LEAD`` with an initial offset that keeps
    ``offset(0) == HUD_SLIDE_A`` (so it never sits still and then jumps).
    """
    lead = HUD_FADE_LEAD                      # negative
    if t >= HUD_SLIDE_T:
        return 0.0
    span = HUD_SLIDE_T - lead
    u = (max(t, lead) - lead) / span
    # Scale so that the offset at t=0 is exactly HUD_SLIDE_A.
    a0 = HUD_SLIDE_A * (HUD_SLIDE_T / span) ** (-HUD_SLIDE_P)
    return a0 * (1.0 - u) ** HUD_SLIDE_P


def hud_enter_offset(t: float) -> float:
    """Vertical HUD offset (design px, positive = down) during entrance.

    Slide + the later life-bar dip/bounce; the two are independent motions in
    the game (the dip is the pause-hint unfolding), so they add.
    """
    return hud_slide_offset(t) + _sample(HUD_DIP, t)


def hud_fade(t: float) -> float:
    """CanvasGroup GroupFade: 1.0 s InOutSine, starting slightly before the slide.

    ``t`` is seconds since the white sheet finished fading (the slide origin).
    The group fade starts ``HUD_FADE_LEAD`` earlier while the card still hides
    it, which is what the capture shows (alpha ~0.27 at t=0).
    """
    u = (t - HUD_FADE_LEAD) / HUD_FADE_T
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * u)


def hud_bar_fade(t: float) -> float:
    """Life-bar alpha: ease-out over the whole entrance, full at the dip top.

    The bar used to stay translucent until the dip's top (t=1.35); the user then
    asked for it to be fully opaque as soon as the slide lands.  So the window
    is [-0.35, HUD_SLIDE_T] and the shape is u ** POW (ease-in, dim early and
    full exactly at arrival).  POW is re-tuned whenever HUD_SLIDE_T changes so
    the mid-entrance brightness stays put (1.8 "亮了点" -> 2.2 -> 1.9 after the
    slide went back to 0.46 s): 0.20@0 / 0.33@0.10 / 0.48@0.20 / 0.66@0.30 /
    1.00@0.46.
    The result is *not* clamped by the group fade any more, otherwise it would
    still top out at the group's 0.85 at arrival.
    """
    u = (t - HUD_FADE_LEAD) / HUD_BAR_FADE_T
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u ** HUD_BAR_FADE_POW


def combo_score(count: int) -> int:
    """Weighted combo used by the accuracy formula (wiki)."""
    if count <= 5:
        return count
    if count <= 8:
        return 2 * count - 5
    if count <= 11:
        return 3 * count - 13
    return 4 * count - 24


def judge_times(chart: Chart) -> list[float]:
    """Sorted times of every judgement the game counts toward combo.

    Tap / drag: one judgement at the note time.  Hold: two — the head at
    ``start_sec`` and the tail at ``end_sec`` — so a chart's combo range is
    ``notes + holds``, not ``notes``.
    """
    times: list[float] = []
    for note in chart.notes:
        times.append(note.start_sec)
        if note.kind == NOTE_HOLD:
            times.append(note.end_sec)
    times.sort()
    return times


class _GlyphAtlas:
    __slots__ = ("tex", "cells")

    def __init__(self, tex: Texture, cells: dict):
        self.tex = tex
        self.cells = cells


class Hud:
    def __init__(self, assets: AssetStore, width: int, height: int):
        self.assets = assets
        self.width = int(width)
        self.height = int(height)
        self._judge_times: list[float] = []
        self._total_judges = 0
        self._total_combo = 0
        self._dot_times: list[float | None] = [None] * 7
        self._prepared = None
        self._bar: Texture | None = None
        self._fill: Texture | None = None
        self._bar_key = None
        self._ring: Texture | None = None
        self._ring_key = None
        self._hint: Texture | None = None
        self._hint_key = None
        self._hint_label_tex: Texture | None = None
        self._hint_label_key = None
        self._hint_label_size = (0.0, 0.0)
        self._glyphs: dict[int, _GlyphAtlas] = {}

    @property
    def scale(self) -> float:
        return self.width / 1080.0

    def resize(self, width: int, height: int) -> None:
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = int(width), int(height)
        self._bar = None
        self._fill = None
        self._bar_key = None
        self._ring = None
        self._ring_key = None
        self._hint = None
        self._hint_key = None
        self._hint_label_tex = None
        self._hint_label_key = None
        self._hint_label_size = (0.0, 0.0)
        self._glyphs = {}

    # -- data ------------------------------------------------------------

    def prepare(self, chart: Chart) -> None:
        if chart is self._prepared:
            return
        self._prepared = chart
        self._judge_times = judge_times(chart)
        self._total_judges = len(self._judge_times)
        self._total_combo = combo_score(self._total_judges)
        # When each rating dot lights up (hollow -> filled), baked once.
        self._dot_times = [self._dot_cross(60.0 + 10.0 * i) for i in range(7)]

    def combo_at(self, time_sec: float) -> int:
        """Displayed combo is the wiki's weighted value, not the raw count."""
        raw = bisect.bisect_right(self._judge_times, time_sec + 1e-6)
        return combo_score(raw)

    def _dot_cross(self, threshold: float) -> float | None:
        """First judgement time where accuracy reaches ``threshold`` (perfect play)."""
        times = self._judge_times
        lo, hi = 0, len(times)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.accuracy_at(times[mid]) >= threshold:
                hi = mid
            else:
                lo = mid + 1
        return times[lo] if lo < len(times) else None

    def accuracy_at(self, time_sec: float) -> float:
        """Live accuracy (percentage) of a perfect play.

        Reverse engineered from ``NormalModeInLevelController.GetJudge``
        (PC GameAssembly 0x3C0641..0x3C06A7):

            nowNotePercent  = _comboScore / _maxComboScore
            challengePercent = hitNotes / _allComboNum
            _nowPercent = (nowNotePercent + 0.2 * challengePercent)
                          * _completeRateScale

        ``_completeRateScale`` is 1 for normal play.  The star row needs
        ``_nowPercent`` in 0..1.2, so this returns 0..120.
        """
        if self._total_combo <= 0:
            return 100.0
        passed = bisect.bisect_right(self._judge_times, time_sec + 1e-6)
        note_ratio = combo_score(passed) / self._total_combo
        hit_ratio = passed / self._total_judges
        return 100.0 * (note_ratio + 0.2 * hit_ratio)

    # -- textures --------------------------------------------------------

    def prewarm(self, chart: Chart, accent: tuple[int, int, int]) -> None:
        """Bake every HUD texture once so the entrance has no first-frame hitch."""
        self.prepare(chart)
        self._bar_textures(accent)
        self._ring_texture()
        self._hint_texture()
        self._hint_label()
        s = self.scale
        self._glyph_atlas(max(4, int(round(COMBO_NUMBER * s))))
        self._glyph_atlas(max(4, int(round(COMBO_LABEL * s))))

    def _bar_textures(self, accent: tuple[int, int, int]) -> tuple[Texture, Texture]:
        """Baked life bar layers (prefab: Shadow/Border/BGColor/BGCover/Fill).

        ``base`` holds the capsule shadow, the white body and the pale (82%
        white over accent) empty track; ``fill`` is the accent capsule and is
        drawn on top, cropped horizontally by the life fraction.
        """
        s = self.scale
        px_w = max(8, round(BAR_W * s))
        px_h = max(4, round(BAR_H * s))
        key = (accent, px_w, px_h)
        if self._bar is not None and self._fill is not None and self._bar_key == key:
            return self._bar, self._fill
        ss = 2
        n_w, n_h = px_w * ss, px_h * ss

        def box(inset: float):
            return (
                inset * s * ss,
                inset * s * ss,
                n_w - 1 - inset * s * ss,
                n_h - 1 - inset * s * ss,
            )

        img = Image.new("RGBA", (n_w, n_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(box(0), radius=BAR_RADIUS * s * ss,
                               fill=(0, 0, 0, 51))
        draw.rounded_rectangle(box(BAR_BODY_INSET), radius=(BAR_RADIUS - BAR_BODY_INSET) * s * ss,
                               fill=(255, 255, 255, 255))
        draw.rounded_rectangle(box(BAR_ACCENT_INSET), radius=(BAR_RADIUS - BAR_ACCENT_INSET) * s * ss,
                               fill=(accent[0], accent[1], accent[2], 255))
        cover = Image.new("RGBA", (n_w, n_h), (0, 0, 0, 0))
        ImageDraw.Draw(cover).rounded_rectangle(
            box(BAR_ACCENT_INSET), radius=(BAR_RADIUS - BAR_ACCENT_INSET) * s * ss,
            fill=(255, 255, 255, round(255 * BAR_COVER_ALPHA)),
        )
        img = Image.alpha_composite(img, cover)
        img = img.resize((px_w, px_h), Image.LANCZOS)

        acc_w = max(2, round((BAR_W - 2 * BAR_ACCENT_INSET) * s))
        acc_h = max(2, round((BAR_H - 2 * BAR_ACCENT_INSET) * s))
        fill = Image.new("RGBA", (acc_w * ss, acc_h * ss), (0, 0, 0, 0))
        ImageDraw.Draw(fill).rounded_rectangle(
            (0, 0, acc_w * ss - 1, acc_h * ss - 1),
            radius=(BAR_RADIUS - BAR_ACCENT_INSET) * s * ss,
            fill=(accent[0], accent[1], accent[2], 255),
        )
        fill = fill.resize((acc_w, acc_h), Image.LANCZOS)

        for tex in (self._bar, self._fill):
            if tex is not None:
                try:
                    tex.gl.release()
                except Exception:
                    pass
        self._bar = Texture(self.assets.ctx, img, "hud-bar")
        self._fill = Texture(self.assets.ctx, fill, "hud-bar-fill")
        self._bar_key = key
        return self._bar, self._fill

    def _ring_texture(self) -> Texture:
        s = self.scale
        px = max(6, round(STAR_D * s))
        if self._ring is not None and self._ring_key == px:
            return self._ring
        ss = 4
        n = px * ss
        img = Image.new("RGBA", (n, n), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        w = max(1.0, STAR_STROKE * s * ss)
        # The stroke grows *inwards* from the texture edge, so the visible
        # outer diameter is exactly STAR_D (the prefab LeanCircle size is the
        # outer edge; the recording measures ~47.5 design px).
        draw.ellipse((0, 0, n - 1, n - 1),
                     outline=(255, 255, 255, 255), width=int(round(w)))
        # LeanCircle blur=0.8: soften the ring so lit/empty look the same weight.
        img = img.filter(ImageFilter.GaussianBlur(0.8 * s * ss))
        img = img.resize((px, px), Image.LANCZOS)
        if self._ring is not None:
            try:
                self._ring.gl.release()
            except Exception:
                pass
        tex = Texture(self.assets.ctx, img, "hud-ring")
        self._ring = tex
        self._ring_key = px
        return tex

    def _hint_texture(self) -> Texture:
        """Top-centre pause handle: LeanBox 45x6, radius 50 (clamped), blur 0.2."""
        s = self.scale
        px_w = max(4, round(HINT_W * s))
        px_h = max(2, round(HINT_H * s))
        key = (px_w, px_h)
        if self._hint is not None and self._hint_key == key:
            return self._hint
        ss = 4
        img = Image.new("RGBA", (px_w * ss, px_h * ss), (255, 255, 255, 0))
        ImageDraw.Draw(img).rounded_rectangle(
            (0, 0, px_w * ss - 1, px_h * ss - 1),
            radius=min(px_w, px_h) * ss / 2.0,
            fill=(255, 255, 255, 255),
        )
        img = img.filter(ImageFilter.GaussianBlur(0.2 * s * ss))
        img = img.resize((px_w, px_h), Image.LANCZOS)
        if self._hint is not None:
            try:
                self._hint.gl.release()
            except Exception:
                pass
        self._hint = Texture(self.assets.ctx, img, "hud-pause-hint")
        self._hint_key = key
        return self._hint

    def _hint_label(self) -> Texture:
        """Baked '向下滑动以暂停' (UI.Text ~30 design px, #242424)."""
        s = self.scale
        key = (self.width, self.height)
        if self._hint_label_tex is not None and self._hint_label_key == key:
            return self._hint_label_tex
        path = os.path.join(self.assets.root, "fonts", "hybrid_bold.ttf")
        font = ImageFont.truetype(path, max(8, round(30.0 * s)))
        bbox = font.getbbox(HINT_TEXT)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        img = Image.new("RGBA", (w + 4, h + 4), (0, 0, 0, 0))
        ImageDraw.Draw(img).text(
            (2 - bbox[0], 2 - bbox[1]), HINT_TEXT, font=font, fill=(36, 36, 36, 255)
        )
        if self._hint_label_tex is not None:
            try:
                self._hint_label_tex.gl.release()
            except Exception:
                pass
        self._hint_label_tex = Texture(self.assets.ctx, img, "hud-hint-text")
        self._hint_label_size = ((w + 4) / s, (h + 4) / s)
        self._hint_label_key = key
        return self._hint_label_tex

    def _glyph_atlas(self, px: int) -> _GlyphAtlas:
        cached = self._glyphs.get(px)
        if cached is not None:
            return cached
        font_path = os.path.join(self.assets.root, "fonts", "hybrid_bold.ttf")
        # Draw at 2x and downsample so half-pixel outline/dilate widths are
        # possible (4.5px outline at 65px is between PIL's integer widths).
        ss = 4
        font = ImageFont.truetype(font_path, px * ss)
        stroke = max(1, round(px * ss * STROKE_RATIO))
        dilate = max(0, round(px * ss * DILATE_RATIO))
        cells = {}
        widths = []
        for ch in GLYPHS:
            x0, y0, x1, y1 = font.getbbox(ch)
            w = max(1, x1 - x0) + 2 * stroke
            h = max(1, y1 - y0) + 2 * stroke
            widths.append((w, h, x0, y0))
        cell_w = max(w for w, *_ in widths) + 2
        cell_h = max(h for _, h, *_ in widths) + 2
        img = Image.new("RGBA", (cell_w * len(GLYPHS), cell_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for i, (ch, (w, h, x0, y0)) in enumerate(zip(GLYPHS, widths)):
            pos = (i * cell_w + stroke - x0, stroke - y0)
            # White outline first, then a slightly dilated face on top: this
            # reproduces TMP's _OutlineWidth 0.3 + _FaceDilate 0.5.
            draw.text(
                pos, ch, font=font,
                fill=TEXT_STROKE + (255,),
                stroke_width=stroke, stroke_fill=TEXT_STROKE + (255,),
            )
            draw.text(
                pos, ch, font=font,
                fill=TEXT_FACE + (255,),
                stroke_width=dilate, stroke_fill=TEXT_FACE + (255,),
            )
            cells[ch] = ((i * cell_w) / img.width, w / ss, h / ss)
        small_w = max(1, round(cell_w / ss))
        small_h = max(1, round(cell_h / ss))
        img = img.resize((small_w * len(GLYPHS), small_h), Image.LANCZOS)
        out = {}
        adv_font = ImageFont.truetype(font_path, px)
        for i, (ch, cell) in enumerate(cells.items()):
            _, w, h = cell
            out[ch] = (
                (i * small_w) / img.width, 0.0,
                (i * small_w + w) / img.width, h / img.height,
                w, h, adv_font.getlength(ch),
            )
        tex = Texture(self.assets.ctx, img, f"hud-glyphs-{px}")
        atlas = _GlyphAtlas(tex, out)
        self._glyphs[px] = atlas
        return atlas

    # -- drawing ---------------------------------------------------------

    def draw(self, card, time_sec: float, accent: tuple[int, int, int],
             alpha: float = 1.0, life: float = 1.0, y_off: float = 0.0,
             anim_t: float | None = None) -> None:
        s = self.scale
        at = time_sec if anim_t is None else anim_t
        oy = y_off * s
        # Rating dots ride the slide up but stop when it reaches the resting
        # place; the later dip/bounce applies to the life bar only.
        oy_dots = oy if anim_t is None else hud_slide_offset(at) * s
        base, fill = self._bar_textures(accent)
        # The bar has its own fade that completes when the slide lands (see
        # HUD_BAR_FADE_T); it deliberately ignores the group fade's ceiling.
        bar_a = alpha if anim_t is None else hud_bar_fade(at)
        card.quad(base, BAR_X * s, BAR_Y * s + oy, (BAR_X + BAR_W) * s,
                  (BAR_Y + BAR_H) * s + oy, (1.0, 1.0, 1.0, bar_a))
        life = max(0.0, min(1.0, life))
        if life > 0.001:
            ax = BAR_X + BAR_ACCENT_INSET
            ay = BAR_Y + BAR_ACCENT_INSET
            aw = BAR_W - 2 * BAR_ACCENT_INSET
            ah = BAR_H - 2 * BAR_ACCENT_INSET
            card.quad(fill, ax * s, ay * s + oy, (ax + aw * life) * s,
                      (ay + ah) * s + oy, (1.0, 1.0, 1.0, bar_a),
                      uv=(0.0, 0.0, life, 1.0))
        # Pull-down-to-pause handle; it unfolds into the hint text and back.
        cx = self.width / 2.0
        # User feedback: the pull-down hint starts ~2 frames later than before.
        hint_at = at + HINT_DELAY
        bar_a = hint_bar_alpha(hint_at)
        if bar_a > 0.0:
            hint = self._hint_texture()
            hw = HINT_W * s
            hh = HINT_H * s
            cy = HINT_Y * s + oy
            card.quad(hint, cx - hw / 2.0, cy - hh / 2.0, cx + hw / 2.0, cy + hh / 2.0,
                      (0.0, 0.0, 0.0, HINT_ALPHA * alpha * bar_a))
        text_a, text_scale, text_y = hint_text_state(hint_at)
        if text_a > 0.001:
            tex = self._hint_label()
            tw, th = self._hint_label_size
            dw = tw * text_scale * s
            dh = th * text_scale * s
            cy = text_y * s + oy
            card.quad(tex, cx - dw / 2.0, cy - dh / 2.0, cx + dw / 2.0, cy + dh / 2.0,
                      (1.0, 1.0, 1.0, text_a * alpha))
        # Performance rating (表现评级): 60..120% in 10% steps, 7 dots total but
        # only 5 are shown until the 6th/7th (110%/120%) are reached.
        acc = self.accuracy_at(time_sec)
        lit = min(7, max(0, int((acc - 60.0) // 10.0) + 1))
        shown = 5 + (1 if acc >= 110.0 else 0) + (1 if acc >= 120.0 else 0)
        ring = self._ring_texture()
        disc = self.assets.get("Circle")
        d = STAR_D * s
        fill_d = 27.018 * s
        ink = (STAR_INK[0] / 255.0, STAR_INK[1] / 255.0, STAR_INK[2] / 255.0)
        for i in range(shown):
            cx = (STAR_FIRST_X + i * STAR_DX) * s
            cy = STAR_CY * s + oy_dots
            if i < lit:
                # Fade the dot in over DOT_FADE from the moment it lit up, and
                # cross-fade the hollow ring out at the same time.
                a = alpha
                k = 1.0
                cross = self._dot_times[i] if i < len(self._dot_times) else None
                if cross is not None:
                    k = max(0.0, min(1.0, (time_sec - cross) / DOT_FADE)) ** DOT_EASE
                    a = alpha * k
                if k < 1.0:
                    card.quad(ring, cx - d / 2.0, cy - d / 2.0, cx + d / 2.0, cy + d / 2.0,
                              (*ink, STAR_ALPHA * alpha * (1.0 - k)))
                if a <= 0.002:
                    continue
                card.quad(ring, cx - d / 2.0, cy - d / 2.0, cx + d / 2.0, cy + d / 2.0,
                          (*ink, a))
                card.quad(disc, cx - fill_d / 2.0, cy - fill_d / 2.0,
                          cx + fill_d / 2.0, cy + fill_d / 2.0,
                          (accent[0] / 255.0, accent[1] / 255.0, accent[2] / 255.0, a))
            else:
                card.quad(ring, cx - d / 2.0, cy - d / 2.0, cx + d / 2.0, cy + d / 2.0,
                          (*ink, STAR_ALPHA * alpha))
        # ``combo <n>`` is one right-aligned run: small label + gap + big number.
        # It is hidden until the first hit (combo 0).
        combo = self.combo_at(time_sec)
        if combo <= 0:
            return
        combo = str(combo)
        num_atlas = self._glyph_atlas(max(4, int(round(COMBO_NUMBER * s))))
        label_atlas = self._glyph_atlas(max(4, int(round(COMBO_LABEL * s))))
        num_w = self._advance(num_atlas, combo)
        label_w = self._advance(label_atlas, COMBO_LABEL_TEXT)
        gap = COMBO_GAP * s
        x = COMBO_RIGHT * s - (label_w + gap + num_w)
        baseline = COMBO_BASELINE * s + oy
        self._draw_run(card, label_atlas, COMBO_LABEL_TEXT, x, baseline, alpha)
        self._draw_run(card, num_atlas, combo, x + label_w + gap, baseline, alpha)

    @staticmethod
    def _advance(atlas: _GlyphAtlas, text: str) -> float:
        return sum(atlas.cells[ch][6] for ch in text if ch in atlas.cells)

    def _draw_run(self, card, atlas: _GlyphAtlas, text: str, x: float, baseline: float,
                  alpha: float) -> None:
        pen = x
        for ch in text:
            cell = atlas.cells.get(ch)
            if cell is None:
                continue
            u0, v0, u1, v1, w, h, adv = cell
            card.quad(atlas.tex, pen, baseline - h, pen + w, baseline,
                      (1.0, 1.0, 1.0, alpha), uv=(u0, v0, u1, v1))
            pen += adv

    def _draw_text(self, card, text: str, size_px: float, x: float, baseline: float,
                   align: str, alpha: float = 1.0) -> None:
        px = max(4, int(round(size_px)))
        atlas = self._glyph_atlas(px)
        total = self._advance(atlas, text)
        pen = (x - total) if align == "right" else x
        self._draw_run(card, atlas, text, pen, baseline, alpha)
