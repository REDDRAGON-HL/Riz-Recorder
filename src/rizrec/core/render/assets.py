"""Texture loading and text atlas."""

from __future__ import annotations

import os
from dataclasses import dataclass

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEX_WHITE = 0
TEX_NOTEBG = 1
TEX_CIRCLE = 2
TEX_DRAG = 3
TEX_RING = 4
TEX_SIDE = 5
TEX_HITFX = 6
TEX_TEXT = 7
TEX_SOFT = 8
TEX_HOLD = 9

# Rizline ships a Source Han Sans CN + Comfortaa hybrid; use it for all text.
FONT_NAME = "hybrid.ttf"


class Texture:
    __slots__ = ("gl", "width", "height", "name")

    def __init__(self, ctx: moderngl.Context, image: Image.Image, name: str = "",
                 mipmaps: bool = False):
        img = image.convert("RGBA")
        self.width, self.height = img.size
        self.name = name
        self.gl = ctx.texture(img.size, 4, img.tobytes())
        if mipmaps and min(self.width, self.height) >= 2:
            # Notes / sprites are drawn heavily minified (512px -> ~160px);
            # without mipmaps bilinear sampling aliases their silhouettes.
            self.gl.build_mipmaps()
            self.gl.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            try:
                self.gl.anisotropy = 8.0
            except Exception:
                pass
        else:
            self.gl.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.gl.repeat_x = False
        self.gl.repeat_y = False

    def use(self, unit: int) -> None:
        self.gl.use(unit)


@dataclass(slots=True)
class GlyphLine:
    u0: float
    v0: float
    u1: float
    v1: float
    width: int
    height: int
    x_off: int
    y_off: int
    advance: float
    line_height: int


class TextAtlas:
    """A simple shelf-packed alpha atlas for the Rizline font."""

    def __init__(self, ctx: moderngl.Context, font_path: str, size: int = 2048):
        self.ctx = ctx
        self.font_path = font_path
        self.size = size
        self.image = Image.new("L", (size, size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.x = 1
        self.y = 1
        self.row_h = 0
        self.lines: dict[tuple[str, int], GlyphLine] = {}
        self.texture: Texture | None = None
        self._dirty = False

    def _font(self, px: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.font_path, px)

    def add(self, text: str, px: int) -> GlyphLine:
        key = (text, px)
        cached = self.lines.get(key)
        if cached is not None:
            return cached
        if not text:
            text = " "
        font = self._font(px)
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        bbox = font.getbbox(text)
        x0, y0, x1, y1 = bbox
        w = max(1, x1 - x0) + 2
        h = max(1, y1 - y0) + 2
        if self.x + w >= self.size:
            self.x = 1
            self.y += self.row_h + 1
            self.row_h = 0
        if self.y + h >= self.size:
            raise RuntimeError("text atlas is full")
        # Place the ink's top-left at (x+1, y+1).
        self.draw.text((self.x + 1 - x0, self.y + 1 - y0), text, font=font, fill=255)
        advance = float(font.getlength(text))
        line = GlyphLine(
            u0=self.x / self.size,
            v0=self.y / self.size,
            u1=(self.x + w) / self.size,
            v1=(self.y + h) / self.size,
            width=w,
            height=h,
            x_off=x0,
            y_off=y0,
            advance=advance,
            line_height=line_h,
        )
        self.lines[key] = line
        self.x += w + 1
        self.row_h = max(self.row_h, h)
        self._dirty = True
        return line

    def build_texture(self) -> Texture:
        if self.texture is None or self._dirty:
            arr = np.array(self.image, dtype=np.uint8)
            rgba = np.empty((self.size, self.size, 4), dtype=np.uint8)
            rgba[..., 0:3] = 255
            rgba[..., 3] = arr
            self.texture = Texture(self.ctx, Image.fromarray(rgba, "RGBA"), "text")
            self._dirty = False
        return self.texture


class AssetStore:
    def __init__(self, ctx: moderngl.Context, root: str):
        self.ctx = ctx
        self.root = root
        self._tex: dict[str, Texture] = {}
        self.white = self._make_white()
        self.soft = self._make_soft()
        self.text = TextAtlas(ctx, os.path.join(root, "fonts", FONT_NAME))

    def _make_white(self) -> Texture:
        img = Image.new("RGBA", (1, 1), (255, 255, 255, 255))
        return Texture(self.ctx, img, "white")

    def _make_soft(self) -> Texture:
        n = 64
        yy, xx = np.mgrid[0:n, 0:n]
        cx = cy = (n - 1) / 2.0
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n / 2.0)
        alpha = np.clip((1.0 - d) / 0.12, 0.0, 1.0)
        rgba = np.zeros((n, n, 4), dtype=np.uint8)
        rgba[..., 0:3] = 255
        rgba[..., 3] = (alpha * 255).astype(np.uint8)
        return Texture(self.ctx, Image.fromarray(rgba, "RGBA"), "soft", mipmaps=True)

    def get(self, name: str) -> Texture:
        tex = self._tex.get(name)
        if tex is not None:
            return tex
        path = os.path.join(self.root, "textures", name + ".png")
        if not os.path.exists(path):
            tex = self.white
        else:
            with Image.open(path) as img:
                tex = Texture(self.ctx, img, name, mipmaps=True)
        self._tex[name] = tex
        return tex

    def atlas_texture(self) -> Texture:
        return self.text.build_texture()

    def reset_text(self) -> None:
        self.text = TextAtlas(self.ctx, os.path.join(self.root, "fonts", FONT_NAME))
