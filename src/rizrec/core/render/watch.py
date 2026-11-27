"""Rizline watch card: the white level-intro / level-outro overlay.

The card is the "接下来 / 请观赏" screen the mobile game shows before a level
starts and again when the song ends.  Geometry is expressed in the game's
1080x1920 design units (the values match the Android 2.8.0 prefab); the
animation curves were fitted to the ``16-9_LAIGN_IN.mp4`` device capture and
the node times are tuned to the user's judgement of the real game (HANDOFF
§14.2):

* intro: the card holds, the content slides off the top while the white
  sheet fades out, both ending together (node times in the constants below);
* outro: the sheet fades in and the badge drops in from above over 1.70 s,
  fired at ``max(audio - 0.1, last note)`` like the reference player.

The card is a screen-space overlay, so it is drawn in pixels after the
playfield, the same way the game draws its UI canvas on top.
"""

from __future__ import annotations

import math
import os
from array import array
from dataclasses import dataclass

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .assets import AssetStore, Texture

_CARD_VERT = """#version 330 core
in vec2 in_pos;
in vec2 in_uv;
in vec4 in_col;
uniform vec2 u_res;
out vec2 v_uv;
out vec4 v_col;
void main() {
    vec2 ndc = vec2(in_pos.x / u_res.x * 2.0 - 1.0,
                    1.0 - in_pos.y / u_res.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_col = in_col;
}
"""

_CARD_FRAG = """#version 330 core
uniform sampler2D u_tex;
in vec2 v_uv;
in vec4 v_col;
out vec4 frag;
void main() {
    frag = texture(u_tex, v_uv) * v_col;
}
"""

_CARD_FMT = "2f 2f 4f"
_CARD_ATTRS = ("in_pos", "in_uv", "in_col")


class CardBatch:
    """Tiny single-texture batch for the watch card.

    The card is a handful of quads over a white sheet; keeping it out of the
    main playfield shader (which is already at ten samplers) avoids making the
    per-fragment texture-select chain longer on integrated GPUs.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_CARD_VERT, fragment_shader=_CARD_FRAG)
        self.verts = array("f")
        self.buf = ctx.buffer(reserve=1 << 16)
        self.vao = ctx.vertex_array(self.prog, [(self.buf, _CARD_FMT, *_CARD_ATTRS)])
        self._groups: list[tuple[Texture, list[float]]] = []

    def begin(self) -> None:
        self._groups = []

    def quad(self, tex: Texture, x0: float, y0: float, x1: float, y1: float,
             color, uv: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> None:
        u0, v0, u1, v1 = uv
        if not self._groups or self._groups[-1][0] is not tex:
            self._groups.append((tex, []))
        verts = self._groups[-1][1]
        for (px, py, tu, tv) in (
            (x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1),
            (x0, y0, u0, v0), (x1, y1, u1, v1), (x0, y1, u0, v1),
        ):
            verts.extend((px, py, tu, tv, color[0], color[1], color[2], color[3]))

    def flush(self, width: int, height: int) -> None:
        if not self._groups:
            return
        self.prog["u_res"].value = (float(width), float(height))
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        for tex, verts in self._groups:
            data = array("f", verts)
            self.buf.orphan(max(1 << 16, len(data) * 4))
            self.buf.write(data.tobytes())
            tex.gl.use(0)
            self.prog["u_tex"].value = 0
            self.vao.render(moderngl.TRIANGLES, vertices=len(data) // 8)
        self._groups = []

# ---------------------------------------------------------------------------
# design-space geometry (1080x1920)

ART_D = 412.4096        # cover art diameter
ART_BG = (231, 231, 235)  # #e7e7eb disc behind the art
RIM_D = 451.143         # coloured rim disc
RIM_WHITE = 0.858       # white layer painted over the rim colour
# Thin white outline right at the cover's edge.  The prefab's `White`
# LeanCircle is Ø426.4 / thickness 8 / blur 1, but the device capture shows the
# bright line only ~2 design px wide and sitting at the art's edge (Ø412):
# the thicker part is white-on-pale-rim and invisible, so drawing it at Ø426
# with 8 px looks wrong (user: "圆盘粗细和大小不对").
RING_D = ART_D
RING_W = 2.4
RING_BLUR = 0.6
RING_ALPHA = 0.859
PILL_W = 526.0
PILL_H = 71.0
PILL_R = 35.5
PILL_WHITE = 0.55       # white layer painted over the accent colour

# text anchors (offset from screen centre, design units)
NEXT_DY = -328.39
NEXT_SIZE = 42.0
TITLE_DY = 316.2998
TITLE_SIZE = 53.0
TITLE_MAX_W = 700.0
ARTIST_DY = 378.999
ARTIST_SIZE = 32.0
PILL_DX = -263.0
PILL_DY = 455.5
PILL_TEXT_DX = -181.0
PILL_TEXT_DY = 491.0
PILL_TEXT_SIZE = 26.0
# "watch mode" (auto) icon inside the pill: a white disc whose glyph is cut
# out so the pill colour shows through (measured on the device capture).
PILL_ICON_DX = -227.5
PILL_ICON_DY = 491.0
PILL_ICON_SIZE = 65.0
WATCH_ICON_FILE = "auto.png"
CREDIT_X = 62.0
CREDIT_ILLUSTRATOR_Y = 116.25
CREDIT_DESIGNER_Y = 73.0
CREDIT_SIZE = 24.0
TEXT_COLOR = (74, 74, 74)
# Small vertical calibration (design units) between the game's TMP layout
# and PIL's ink metrics; measured from the device capture.
TEXT_ADJ_NEXT = 2.5
TEXT_ADJ_TITLE = -2.0
TEXT_ADJ_ARTIST = -3.0

# ---------------------------------------------------------------------------
# timing (device capture `23-26-20`, frame by frame)

# The card holds, then the content slides up 1920 design px (bezier ease-in)
# while the white sheet dissolves as (1 - x)^2, roughly together.
# NOTE: this shape is what the user confirmed; the prefab's own `CutIn`
# LocalMove (0.8 s / 1.7 s AnimationCurve) and a fit to the badge positions
# both disagree with the capture and with the user's judgement - do not "fix"
# it back from the prefab without new game-side evidence.
#
# Node times are set directly from the user's frame instructions (60 fps):
#
# | pass                     | static hold | slide end | white fade        | HUD anchor |
# |--------------------------|-------------|-----------|-------------------|------------|
# | pre-feedback             | 1.0000      | 3.2000    | 1.9500-3.2000     | 3.2000     |
# | +10f start / +5f end     | 1.1667      | 3.2833    | 2.1167-3.2833     | 3.2000     |
# | fix -15f                 | 1.1667      | 3.0333    | 2.1167-3.0333     | 3.2000     |
# | -17f (tail)              | 1.1667      | 3.0000    | 1.8333-3.0000     | 3.2000     |
# | white back / -1f / -2f   | 1.1667      | 2.9833    | 2.1167-2.9667     | 2.9167     |
# | +5f HUD                  | 1.1667      | 2.9833    | 2.1167-2.9667     | 3.0000     |
# | +8f white start (now)    | 1.1667      | 2.9833    | 2.2500-2.9667     | 3.0000     |
#
# The slide's end node is one frame before 3.0.  The HUD anchor is
# 3.20 - 17f + 5f = 3.0, i.e. it starts exactly when the sheet is gone, so its
# own fade-in is visible instead of happening behind the white.
# Node times only affect node times: the video length comes from the segment
# and the lead-in.  The white curve history is documented at INTRO_FADE_POW.
INTRO_HOLD = 1.0 + 10.0 / 60.0                     # 1.1667 s static
INTRO_END = 3.0 - 1.0 / 60.0                       # 2.9833 s slide end
INTRO_RISE_DURATION = INTRO_END - INTRO_HOLD       # 1.8167 s
INTRO_FADE_START = 2.1167 + 8.0 / 60.0             # 2.2500 s: starts 8 f later
INTRO_FADE_END = 3.0 - 2.0 / 60.0                  # 2.9667 s (end node untouched)
INTRO_FADE_DURATION = INTRO_FADE_END - INTRO_FADE_START  # 0.7167 s
# Curve history (user iterated on the white sheet alone):
#   1 - x^3.5  held it near-opaque until the last few frames -> "应该更快变淡"
#   1 - x^1.5  -> "消失再快点"
#   (1 - x)^2  -> "又太早了，大概要晚 8 帧才启动" (so the fade START moved from
#                 2.1167 to 2.2500; the end node, the slide and the HUD anchor
#                 were untouched, window 0.7167 s)
#   1 - x^1.0  -> "曲线不对，现在太快了" (the (1 - x)^2 shape dropped the first
#                 half too hard; linear was still judged fast)
#   1 - x^1.5  -> "还是快了点"
#   1 - x^2.0  -> "还是快了"
#   -> 1 - x^2.5 (now): single knob.  Larger POW holds the sheet at opaque
#      longer, smaller dims it sooner; the window never changes.
INTRO_FADE_POW = 2.5    # bigger = sheet stays whiter longer (1.0 = linear)
INTRO_TOTAL = 3.20 - 12.0 / 60.0   # 3.0000 s: HUD entrance anchor (-17f +5f)
INTRO_CONTENT_TOTAL = max(INTRO_END, INTRO_FADE_END) + 1 / 60.0
INTRO_PREROLL = 5.15 + 2.0 / 60.0  # device capture 5.15 + 2 frames (user: chart a bit later)

# Outro card: white sheet 2p - p^2, badge drops in from above with the
# RizPlayer bezier over 1.7 s, then holds 1.0 s.
# Reference (参考/RizPlayer-main resources/js/player.js):
#   * `tryNativeWatchOutro` fires the card at the first frame where
#     `t > audioDuration - 0.1` AND `t >= notesEnd` (last note / hold tail);
#   * `WATCH_OUTRO = 1.7` — the card scene is 1.7 s and the export ends there;
#   * the music is faded with `(2.5 - elapsed) / 2.5` for 1.5 s from the trigger.
OUTRO_DURATION = 1.70
OUTRO_TOTAL = OUTRO_DURATION
OUTRO_LEAD = 0.10


# cubic-bezier easings used by the game (same control points RizPlayer found)
_EASE_IN = (0.52957356, 0.00243309, 0.95873449, 0.01946477)
_EASE_OUT = (0.105226964, 0.91727496, 0.151306748, 1.0)


def _bezier_point(t: float, ax: float, ay: float, bx: float, by: float) -> float:
    u = 1.0 - t
    return 3.0 * u * u * t * ay + 3.0 * u * t * t * by + t * t * t


def _bezier_x(t: float, ax: float, bx: float) -> float:
    u = 1.0 - t
    return 3.0 * u * u * t * ax + 3.0 * u * t * t * bx + t * t * t


def bezier(x: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Cubic bezier easing y(x) with P0=(0,0), P3=(1,1)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = (lo + hi) * 0.5
        if _bezier_x(mid, ax, bx) > x:
            hi = mid
        else:
            lo = mid
    return _bezier_point((lo + hi) * 0.5, ax, ay, bx, by)


def intro_rise(elapsed: float, scale: float) -> float:
    """How far the card content has slid up, in pixels."""
    if elapsed <= INTRO_HOLD:
        return 0.0
    x = min(1.0, (elapsed - INTRO_HOLD) / INTRO_RISE_DURATION)
    return 1920.0 * scale * bezier(x, *_EASE_IN)


def intro_sheet_alpha(elapsed: float) -> float:
    x = (elapsed - INTRO_FADE_START) / INTRO_FADE_DURATION
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 0.0
    return 1.0 - x ** INTRO_FADE_POW


def outro_progress(elapsed: float) -> float:
    return max(0.0, min(1.0, elapsed / OUTRO_DURATION))


def outro_sheet_alpha(elapsed: float) -> float:
    p = outro_progress(elapsed)
    return 2.0 * p - p * p


def outro_badge_offset(elapsed: float, scale: float, height: float) -> float:
    """Pixel offset of the badge centre from the screen centre."""
    p = outro_progress(elapsed)
    return 2000.0 * scale * bezier(p, *_EASE_OUT) - 2000.0 * scale


@dataclass(slots=True)
class WatchMeta:
    title: str = ""
    artist: str = ""
    illustrator: str = ""
    designer: str = ""


def _mix(a: tuple[float, float, float], b: tuple[float, float, float], t: float):
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _circle_mask(size: int, radius: float, ss: int = 4) -> Image.Image:
    n = max(2, int(size * ss))
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = (n - 1) / 2.0
    r = radius * ss
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(r - d + 0.5, 0.0, 1.0)
    img = Image.fromarray((alpha * 255).astype(np.uint8), "L")
    return img.resize((size, size), Image.LANCZOS)


class WatchCard:
    """Builds and draws the watch card for one renderer."""

    def __init__(self, assets: AssetStore, width: int, height: int):
        self.assets = assets
        self.width = int(width)
        self.height = int(height)
        self._badge_key: tuple | None = None
        self._badge: Texture | None = None
        self._pill: Texture | None = None
        self._icon: Texture | None = None
        self._icon_px = 0
        self._lines: dict[str, object] = {}
        self._meta: WatchMeta | None = None
        self._title_px = 0

    # -- layout ----------------------------------------------------------

    @property
    def scale(self) -> float:
        w, h = self.width, self.height
        if w <= 0 or h <= 0:
            return 1.0
        if w / h < 0.5625:
            return w / 1080.0
        return math.sqrt(w * h / (1080.0 * 1920.0))

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    def resize(self, width: int, height: int) -> None:
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = int(width), int(height)
        self._badge_key = None
        self._badge = None
        self._pill = None
        self._icon = None
        self._icon_px = 0
        self._lines = {}
        self._meta = None

    def reset_text(self) -> None:
        """Drop cached atlas lines (the renderer rebuilt the text atlas)."""
        self._lines = {}
        self._meta = None

    # -- text ------------------------------------------------------------

    @staticmethod
    def _fit_size(assets: AssetStore, text: str, px: float, max_w: float) -> int:
        px = max(1, int(round(px)))
        if not text:
            return px
        advance = assets.text._font(px).getlength(text)  # noqa: SLF001 - atlas helper
        if advance > max_w and advance > 0:
            px = max(5, int(px * max_w / advance))
        return px

    def prepare(self, meta: WatchMeta) -> None:
        if meta == self._meta and self._lines:
            return
        self._meta = meta
        s = self.scale
        px_next = max(1, round(NEXT_SIZE * s))
        px_title = self._fit_size(self.assets, meta.title, TITLE_SIZE * s, TITLE_MAX_W * s)
        px_artist = max(1, round(ARTIST_SIZE * s))
        px_pill = max(1, round(PILL_TEXT_SIZE * s))
        px_credit = max(1, round(CREDIT_SIZE * s))
        atlas = self.assets.text
        self._title_px = px_title
        self._lines = {
            "next": atlas.add("接下来", px_next),
            "title": atlas.add(meta.title, px_title) if meta.title else None,
            "artist": atlas.add(meta.artist, px_artist) if meta.artist else None,
            "pill": atlas.add("请观赏", px_pill),
            "illustrator": atlas.add(f"插画设计 : {meta.illustrator}", px_credit)
            if meta.illustrator else None,
            "designer": atlas.add(f"关卡设计 : {meta.designer}", px_credit)
            if meta.designer else None,
        }

    # -- textures --------------------------------------------------------

    def badge_texture(self, accent: tuple[int, int, int], cover_path: str) -> Texture:
        key = (accent, cover_path, max(1, round(RIM_D * self.scale)))
        if self._badge is not None and key == self._badge_key:
            return self._badge
        px = key[2]
        ss = 2
        n = px * ss
        rim = _mix((255.0, 255.0, 255.0), tuple(float(v) for v in accent), 1.0 - RIM_WHITE)
        rim_rgba = (int(round(rim[0])), int(round(rim[1])), int(round(rim[2])), 255)
        # Fill the whole square with the rim colour and cut the alpha later:
        # resizing a transparent-background image would darken the AA edge.
        img = Image.new("RGBA", (n, n), rim_rgba)
        draw = ImageDraw.Draw(img)
        art_px = ART_D / RIM_D * n
        art0 = (n - art_px) / 2.0
        draw.ellipse((art0, art0, art0 + art_px, art0 + art_px), fill=ART_BG + (255,))
        cover = self._load_cover(cover_path)
        if cover is not None:
            art = cover.resize((max(1, int(round(art_px))), max(1, int(round(art_px)))),
                               Image.LANCZOS, box=self._cover_box(cover))
            mask = _circle_mask(art.size[0], art.size[0] / 2.0 - 0.5)
            img.paste(art, (int(round(art0)), int(round(art0))), mask)
        # White ring just outside the cover art (prefab `White`: LeanCircle
        # Ø426.4, thickness 8, blur 1, white alpha 0.859).  It sits between the
        # art edge (Ø412.4) and the coloured rim, and was missing before.
        # Composite through an alpha mask instead of pasting a transparent
        # layer: resizing RGBA with (0,0,0,0) gaps blackens the ring's edges.
        ring_mask = Image.new("L", (n, n), 0)
        # drawn just *outside* the art edge (the capture shows the bright
        # line at r≈205 when the art ends at r≈204)
        ring_r = (RING_D + 2.0 * RING_W) / RIM_D * n
        ring_w = max(1, int(round(RING_W * self.scale * ss)))
        ImageDraw.Draw(ring_mask).ellipse(
            (n / 2.0 - ring_r / 2.0, n / 2.0 - ring_r / 2.0,
             n / 2.0 + ring_r / 2.0, n / 2.0 + ring_r / 2.0),
            outline=round(255 * RING_ALPHA), width=ring_w,
        )
        ring_mask = ring_mask.filter(ImageFilter.GaussianBlur(RING_BLUR * self.scale * ss))
        img = Image.composite(Image.new("RGBA", (n, n), (255, 255, 255, 255)),
                              img, ring_mask)
        img = img.resize((px, px), Image.LANCZOS)
        img.putalpha(_circle_mask(px, px / 2.0 - 0.5))
        if self._badge is not None:
            try:
                self._badge.gl.release()
            except Exception:
                pass
        tex = Texture(self.assets.ctx, img, "watch-badge")
        self._badge = tex
        self._badge_key = key
        return tex

    @staticmethod
    def _cover_box(img: Image.Image) -> tuple[float, float, float, float]:
        w, h = img.size
        side = min(w, h)
        return ((w - side) / 2.0, (h - side) / 2.0, (w + side) / 2.0, (h + side) / 2.0)

    @staticmethod
    def _load_cover(path: str) -> Image.Image | None:
        if not path or not os.path.exists(path):
            return None
        try:
            with Image.open(path) as img:
                rgba = img.convert("RGBA")
                if rgba.getextrema()[3][0] < 255:
                    # Transparent covers sit on the card's grey disc, not black.
                    bg = Image.new("RGBA", rgba.size, ART_BG + (255,))
                    rgba = Image.alpha_composite(bg, rgba)
                return rgba.convert("RGB")
        except Exception:
            return None

    def pill_texture(self) -> Texture:
        if self._pill is not None:
            return self._pill
        s = self.scale
        px_w = max(4, round(PILL_W * s))
        px_h = max(4, round(PILL_H * s))
        ss = 2
        n_w, n_h = px_w * ss, px_h * ss
        # Opaque white canvas + separate alpha mask avoids dark fringes when
        # the rounded rectangle is downsampled.
        img = Image.new("RGBA", (n_w, n_h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        r = PILL_R * s * ss
        draw.rounded_rectangle((0, 0, n_w - 1, n_h - 1), radius=r, fill=(255, 255, 255, 255))
        mask = Image.new("L", (n_w, n_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, n_w - 1, n_h - 1), radius=r, fill=255)
        img = img.resize((px_w, px_h), Image.LANCZOS)
        img.putalpha(mask.resize((px_w, px_h), Image.LANCZOS))
        tex = Texture(self.assets.ctx, img, "watch-pill")
        self._pill = tex
        return tex

    def icon_texture(self, size_px: int) -> Texture | None:
        """White "watch mode" disc with the glyph knocked out.

        The source art is a coloured disc with a light glyph; on the card the
        disc is white and the pill colour shows through the glyph, so we use
        only the disc alpha minus the glyph.
        """
        if self._icon is not None and self._icon_px == size_px:
            return self._icon
        path = os.path.join(self.assets.root, "textures", WATCH_ICON_FILE)
        if not os.path.isfile(path):
            return None
        try:
            with Image.open(path) as raw:
                src = raw.convert("RGBA")
        except Exception:
            return None
        arr = np.asarray(src, dtype=np.float32)
        alpha = arr[..., 3] / 255.0
        glyph = (arr[..., :3].min(axis=2) > 200.0).astype(np.float32)
        mask = np.clip(alpha * (1.0 - glyph), 0.0, 1.0)
        img = Image.new("RGBA", (size_px, size_px), (255, 255, 255, 255))
        img.putalpha(
            Image.fromarray((mask * 255.0).astype(np.uint8), "L").resize(
                (size_px, size_px), Image.LANCZOS
            )
        )
        tex = Texture(self.assets.ctx, img, "watch-auto-icon")
        self._icon = tex
        self._icon_px = size_px
        return tex

    # -- drawing ---------------------------------------------------------

    def draw_intro(self, card: CardBatch, elapsed: float, meta: WatchMeta,
                   accent: tuple[int, int, int], cover_path: str,
                   show_pill: bool = True) -> None:
        s = self.scale
        self.prepare(meta)
        badge = self.badge_texture(accent, cover_path)
        pill_tex = self.pill_texture()
        atlas = self.assets.atlas_texture()
        sheet = intro_sheet_alpha(elapsed)
        if sheet <= 0.001:
            return
        card.quad(self.assets.white, 0.0, 0.0, self.width, self.height,
                  (1.0, 1.0, 1.0, sheet))
        rise = intro_rise(elapsed, s)
        cy = self.cy - rise
        badge_px = RIM_D * s
        card.quad(badge, self.cx - badge_px / 2.0, cy - badge_px / 2.0,
                  self.cx + badge_px / 2.0, cy + badge_px / 2.0, (1.0, 1.0, 1.0, 1.0))
        text_col = _rgba_f(TEXT_COLOR, sheet)
        fade_col = text_col
        if self._lines.get("next") is not None:
            self._text_center(card, atlas, self._lines["next"], self.cx,
                              cy + (NEXT_DY + TEXT_ADJ_NEXT) * s, text_col)
        if self._lines.get("title") is not None:
            design_px = self._title_px / s if s > 0 else TITLE_SIZE
            title_dy = TITLE_DY - 2.0 * design_px / TITLE_SIZE + TEXT_ADJ_TITLE
            self._text_center(card, atlas, self._lines["title"], self.cx,
                              cy + title_dy * s, text_col)
        if self._lines.get("artist") is not None:
            self._text_center(card, atlas, self._lines["artist"], self.cx,
                              cy + (ARTIST_DY + TEXT_ADJ_ARTIST) * s, text_col)
        # "请观赏" pill (prefab BottomInfo): fades with the white sheet.
        if show_pill and sheet > 0.001:
            pill_w, pill_h = PILL_W * s, PILL_H * s
            accent_col = _mix((255.0, 255.0, 255.0), tuple(float(v) for v in accent),
                              1.0 - PILL_WHITE)
            card.quad(
                pill_tex,
                self.cx + PILL_DX * s, cy + PILL_DY * s,
                self.cx + PILL_DX * s + pill_w, cy + PILL_DY * s + pill_h,
                (accent_col[0] / 255.0, accent_col[1] / 255.0, accent_col[2] / 255.0, sheet),
            )
            icon_px = max(4, round(PILL_ICON_SIZE * s))
            icon = self.icon_texture(icon_px)
            if icon is not None:
                icon_size = PILL_ICON_SIZE * s
                card.quad(
                    icon,
                    self.cx + PILL_ICON_DX * s - icon_size / 2.0,
                    cy + PILL_ICON_DY * s - icon_size / 2.0,
                    self.cx + PILL_ICON_DX * s + icon_size / 2.0,
                    cy + PILL_ICON_DY * s + icon_size / 2.0,
                    (1.0, 1.0, 1.0, sheet),
                )
            if self._lines.get("pill") is not None:
                # left-anchored at the pill's text origin, vertically centred
                self._text(card, atlas, self._lines["pill"],
                           self.cx + PILL_TEXT_DX * s, cy + PILL_TEXT_DY * s, fade_col,
                           align="left")
        # credits stay put while the content rises
        if self._lines.get("illustrator") is not None:
            self._text(
                card, atlas, self._lines["illustrator"],
                CREDIT_X * s, self.height - CREDIT_ILLUSTRATOR_Y * s, fade_col, align="left",
            )
        if self._lines.get("designer") is not None:
            self._text(
                card, atlas, self._lines["designer"],
                CREDIT_X * s, self.height - CREDIT_DESIGNER_Y * s, fade_col, align="left",
            )

    def draw_outro(self, card: CardBatch, elapsed: float, accent: tuple[int, int, int],
                   cover_path: str) -> None:
        s = self.scale
        badge = self.badge_texture(accent, cover_path)
        sheet = outro_sheet_alpha(elapsed)
        if sheet <= 0.001:
            return
        card.quad(self.assets.white, 0.0, 0.0, self.width, self.height,
                  (1.0, 1.0, 1.0, sheet))
        badge_px = RIM_D * s
        cy = self.cy + outro_badge_offset(elapsed, s, self.height)
        card.quad(badge, self.cx - badge_px / 2.0, cy - badge_px / 2.0,
                  self.cx + badge_px / 2.0, cy + badge_px / 2.0, (1.0, 1.0, 1.0, 1.0))
    # -- primitives ------------------------------------------------------

    @staticmethod
    def _text_scaled(card: CardBatch, atlas: Texture, line, cx: float,
                     ink_bottom: float, k: float, color) -> None:
        w = line.width * k
        h = line.height * k
        left = cx - w / 2.0
        top = ink_bottom - h
        card.quad(atlas, left, top, left + w, top + h, color,
                  uv=(line.u0, line.v0, line.u1, line.v1))

    @staticmethod
    def _text_center(card: CardBatch, atlas: Texture, line, cx: float, cy: float,
                     color) -> None:
        left = cx + line.x_off - 1 - line.width / 2.0
        top = cy + line.y_off - 1 - line.height / 2.0
        card.quad(atlas, left, top, left + line.width, top + line.height, color,
                  uv=(line.u0, line.v0, line.u1, line.v1))

    @staticmethod
    def _text(card: CardBatch, atlas: Texture, line, x: float, cy: float, color,
              align: str = "center") -> None:
        """Draw one atlas line: vertically ink-centred, optional horizontal anchor."""
        if align == "left":
            left = x + line.x_off - 1
        else:
            left = x + line.x_off - 1 - line.width / 2.0
        top = cy + line.y_off - 1 - line.height / 2.0
        card.quad(atlas, left, top, left + line.width, top + line.height, color,
                  uv=(line.u0, line.v0, line.u1, line.v1))


def _rgba_f(rgb: tuple[int, int, int], alpha: float):
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, max(0.0, min(1.0, alpha)))
