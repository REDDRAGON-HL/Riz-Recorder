"""Single-batch quad renderer.

All layers are submitted in paint order into one vertex buffer and drawn with
one call.  A flat per-vertex texture id selects between a fixed set of texture
units, which keeps the layer order intact without per-texture draw calls.
"""

from __future__ import annotations

from array import array

import moderngl

from .assets import (
    TEX_CIRCLE,
    TEX_DRAG,
    TEX_HITFX,
    TEX_NOTEBG,
    TEX_RING,
    TEX_SIDE,
    TEX_SOFT,
    TEX_TEXT,
    AssetStore,
)

VERT = """#version 330 core
in vec2 in_pos;
in vec2 in_uv;
in vec4 in_col;
in vec4 in_col2;
in float in_mode;
in float in_tid;

uniform vec2 u_res;
uniform vec4 u_circle; // xy = centre (px), z = radius, w = enabled

out vec2 v_uv;
out vec4 v_col;
out vec4 v_col2;
out float v_mode;
out float v_tid;
out vec2 v_pos;

void main() {
    vec2 ndc = vec2(in_pos.x / u_res.x * 2.0 - 1.0,
                    1.0 - in_pos.y / u_res.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_col = in_col;
    v_col2 = in_col2;
    v_mode = in_mode;
    v_tid = in_tid;
    v_pos = in_pos;
}
"""

FRAG = """#version 330 core
uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform sampler2D u_tex2;
uniform sampler2D u_tex3;
uniform sampler2D u_tex4;
uniform sampler2D u_tex5;
uniform sampler2D u_tex6;
uniform sampler2D u_tex7;
uniform sampler2D u_tex8;
uniform sampler2D u_tex9;
uniform vec4 u_circle;

in vec2 v_uv;
in vec4 v_col;
in vec4 v_col2;
in float v_mode;
in float v_tid;
in vec2 v_pos;

out vec4 frag;

void main() {
    int tid = int(v_tid + 0.5);
    vec4 t;
    if (tid == 0) t = texture(u_tex0, v_uv);
    else if (tid == 1) t = texture(u_tex1, v_uv);
    else if (tid == 2) t = texture(u_tex2, v_uv);
    else if (tid == 3) t = texture(u_tex3, v_uv);
    else if (tid == 4) t = texture(u_tex4, v_uv);
    else if (tid == 5) t = texture(u_tex5, v_uv);
    else if (tid == 6) t = texture(u_tex6, v_uv);
    else if (tid == 7) t = texture(u_tex7, v_uv);
    else if (tid == 8) t = texture(u_tex8, v_uv);
    else t = texture(u_tex9, v_uv);

    if (v_mode > 0.5) t.rgb = vec3(1.0);

    float m = 0.0;
    if (u_circle.w > 0.5) {
        float d = distance(v_pos, u_circle.xy);
        m = 1.0 - smoothstep(u_circle.z - 0.8, u_circle.z + 0.8, d);
    }
    vec4 c = mix(v_col, v_col2, m);
    if (v_mode > 1.5) {
        // Analytic AA for line quads: v goes 0..1 across the stroke.
        float fw = max(fwidth(v_uv.y), 1e-5);
        float d = min(v_uv.y, 1.0 - v_uv.y) / fw;
        c.a *= clamp(d + 0.5, 0.0, 1.0);
    }
    frag = t * c;
}
"""

FMT = "2f 2f 4f 4f 1f 1f"
ATTRS = ("in_pos", "in_uv", "in_col", "in_col2", "in_mode", "in_tid")


class MeshBuilder:
    def __init__(self, ctx: moderngl.Context, assets: AssetStore):
        self.ctx = ctx
        self.assets = assets
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        for i in range(10):
            if f"u_tex{i}" in self.prog:
                self.prog[f"u_tex{i}"].value = i
        self.verts = array("f")
        self._cap = 1 << 20
        self.buf = ctx.buffer(reserve=self._cap)
        self.vao = ctx.vertex_array(self.prog, [(self.buf, FMT, *ATTRS)])

    # -- primitives ------------------------------------------------------

    def _vertex(
        self,
        x: float,
        y: float,
        u: float,
        v: float,
        c: tuple[float, float, float, float],
        c2: tuple[float, float, float, float],
        mode: float,
        tid: float,
    ) -> None:
        self.verts.extend((x, y, u, v, c[0], c[1], c[2], c[3], c2[0], c2[1], c2[2], c2[3], mode, tid))

    def quad4(
        self,
        tid: int,
        corners: tuple[tuple[float, float], ...],
        uvs: tuple[tuple[float, float], ...],
        color,
        color2=None,
        mode: int = 0,
    ) -> None:
        c = color
        c2 = color2 if color2 is not None else color
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners
        (u0, v0), (u1, v1), (u2, v2), (u3, v3) = uvs
        m = float(mode)
        t = float(tid)
        self._vertex(x0, y0, u0, v0, c, c2, m, t)
        self._vertex(x1, y1, u1, v1, c, c2, m, t)
        self._vertex(x2, y2, u2, v2, c, c2, m, t)
        self._vertex(x0, y0, u0, v0, c, c2, m, t)
        self._vertex(x2, y2, u2, v2, c, c2, m, t)
        self._vertex(x3, y3, u3, v3, c, c2, m, t)

    def quad4_colors(
        self,
        tid: int,
        corners: tuple[tuple[float, float], ...],
        uvs: tuple[tuple[float, float], ...],
        colors: tuple,
        color2s: tuple | None = None,
        mode: int = 0,
    ) -> None:
        c2s = color2s if color2s is not None else colors
        m = float(mode)
        t = float(tid)
        for k in (0, 1, 2, 0, 2, 3):
            x, y = corners[k]
            u, v = uvs[k]
            self._vertex(x, y, u, v, colors[k], c2s[k], m, t)

    def quad(
        self,
        tid: int,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color,
        color2=None,
        mode: int = 0,
        uv: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ) -> None:
        u0, v0, u1, v1 = uv
        self.quad4(
            tid,
            ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
            ((u0, v0), (u1, v0), (u1, v1), (u0, v1)),
            color,
            color2,
            mode,
        )

    def gradient_quad(
        self,
        tid: int,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        top_color,
        bottom_color,
        color2_top=None,
        color2_bottom=None,
        mode: int = 0,
    ) -> None:
        c2t = color2_top if color2_top is not None else top_color
        c2b = color2_bottom if color2_bottom is not None else bottom_color
        m = float(mode)
        t = float(tid)
        self._vertex(x0, y0, 0.0, 0.0, top_color, c2t, m, t)
        self._vertex(x1, y0, 1.0, 0.0, top_color, c2t, m, t)
        self._vertex(x1, y1, 1.0, 1.0, bottom_color, c2b, m, t)
        self._vertex(x0, y0, 0.0, 0.0, top_color, c2t, m, t)
        self._vertex(x1, y1, 1.0, 1.0, bottom_color, c2b, m, t)
        self._vertex(x0, y1, 0.0, 1.0, bottom_color, c2b, m, t)

    def line_quad(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: float,
        color,
        color2=None,
    ) -> None:
        dx = x1 - x0
        dy = y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 1e-6:
            return
        nx = -dy / length * (width * 0.5)
        ny = dx / length * (width * 0.5)
        self.quad4(
            0,
            ((x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)),
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            color,
            color2,
            2,
        )

    def line_quad_grad(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: float,
        color0,
        color1,
        color0b=None,
        color1b=None,
    ) -> None:
        """One quad with endpoint colours (linear gradient along the line)."""
        dx = x1 - x0
        dy = y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 1e-6:
            return
        nx = -dy / length * (width * 0.5)
        ny = dx / length * (width * 0.5)
        corners = (
            (x0 + nx, y0 + ny),
            (x1 + nx, y1 + ny),
            (x1 - nx, y1 - ny),
            (x0 - nx, y0 - ny),
        )
        colors = (color0, color1, color1, color0)
        c0b = color0b if color0b is not None else color0
        c1b = color1b if color1b is not None else color1
        colors2 = (c0b, c1b, c1b, c0b)
        self.quad4_colors(
            0,
            corners,
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            colors,
            colors2,
            2,
        )

    def line_strip(
        self,
        xs: list[float],
        ys: list[float],
        colors: list[tuple[float, float, float, float]],
        width: float,
        cap_start: bool = True,
        cap_end: bool = True,
    ) -> None:
        """Unity LineRenderer strip emulation (numCornerVertices = 0).

        The game feeds a sampled ``SetPositions`` array to the native
        renderer, so the geometry is not a raster stroke: one cross-section is
        emitted per submitted position, and it is emitted only when the *next*
        position arrives.  Its normal is therefore the join of the previous
        and current chords, the join is blended toward the intersection of the
        two offset lines, and the miter length is capped by a chord length that
        lags one more edge than the previous one:

            cap = max(1, min(3, 3 * limit_chord / (4 * radius)))

        Instant-x (ease 13/14) seams are not part of this strip: the game
        draws those horizontal jumpers with a separate thin line, so the
        renderer splits the run there and draws the jumper itself.
        """
        n = len(xs)
        if n < 2:
            return
        r = width * 0.5

        def slope(nx: float, ny: float) -> float:
            # JS/C semantics of -nx/ny: +-infinity clamps to +-1e5.
            if ny != 0.0:
                v = -nx / ny
            else:
                v = 1e5 if nx < 0.0 else -1e5
            return max(-1e5, min(1e5, v))

        cross: list[tuple[float, float, float, float, tuple]] = []
        px, py, pc = xs[0], ys[0], colors[0]
        has_edge = False
        prev_n = (0.0, 0.0)
        # Keep the native bookkeeping order: ``limit`` lags one more edge than
        # ``edge`` because the cross-section is written before the lengths are
        # advanced.
        limit_len = 0.0
        edge_len = 0.0

        for i in range(1, n):
            x, y, c = xs[i], ys[i], colors[i]
            dx = x - px
            dy = y - py
            length = (dx * dx + dy * dy) ** 0.5
            if length <= 1e-9 or length != length:
                continue
            nx, ny = -dy / length, dx / length
            jx, jy = nx, ny
            if has_edge:
                qx, qy = prev_n
                jx, jy = qx, qy
                m0 = slope(qx, qy)
                m1 = slope(nx, ny)
                diff = m0 - m1
                if abs(diff) >= 1e-5:
                    ix = (ny - nx * m1 - qy + qx * m0) / diff
                    iy = m0 * (ix - qx) + qy
                    dot = qx * nx + qy * ny
                    w = max(0.0, min(1.0, 100.0 * (1.0 - abs(dot))))
                    jx += (ix - jx) * w
                    jy += (iy - jy) * w
                    cap = max(1.0, min(3.0, 3.0 * limit_len / (4.0 * r)))
                    mag2 = jx * jx + jy * jy
                    if mag2 > cap * cap:
                        k = cap / (mag2 ** 0.5)
                        jx *= k
                        jy *= k
            # The cross-section belongs to the previous position.
            cross.append((px, py, jx * r, jy * r, pc))
            limit_len = edge_len if has_edge else length
            edge_len = length
            has_edge = True
            px, py, pc, prev_n = x, y, c, (nx, ny)

        if not has_edge:
            return
        # Final position uses the last chord's normal (no join), like the
        # renderer's closing cross-section.
        cross.append((px, py, prev_n[0] * r, prev_n[1] * r, pc))

        for i in range(1, len(cross)):
            x0, y0, na0, nb0, c0 = cross[i - 1]
            x1, y1, na1, nb1, c1 = cross[i]
            self.quad4_colors(
                0,
                (
                    (x0 + na0, y0 + nb0),
                    (x1 + na1, y1 + nb1),
                    (x1 - na1, y1 - nb1),
                    (x0 - na0, y0 - nb0),
                ),
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                (c0, c1, c1, c0),
                mode=2,
            )

        if cap_start:
            self.quad(
                TEX_CIRCLE,
                xs[0] - r,
                ys[0] - r,
                xs[0] + r,
                ys[0] + r,
                colors[0],
                mode=1,
            )
        if cap_end:
            self.quad(
                TEX_CIRCLE,
                xs[-1] - r,
                ys[-1] - r,
                xs[-1] + r,
                ys[-1] + r,
                colors[-1],
                mode=1,
            )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        color,
        color2=None,
    ) -> None:
        self.quad(
            TEX_SOFT,
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            color,
            color2,
            mode=1,
        )

    # -- submission ------------------------------------------------------

    def flush(self, width: int, height: int, circle=(0.0, 0.0, 0.0, 0.0)) -> None:
        if not self.verts:
            return
        import os

        debug = bool(os.environ.get("RIZREC_DEBUG"))
        data = self.verts.tobytes()
        size = len(data)
        if size > self._cap:
            self._cap = max(size * 2, self._cap * 2)
            self.buf.release()
            self.vao.release()
            self.buf = self.ctx.buffer(reserve=self._cap)
            self.vao = self.ctx.vertex_array(self.prog, [(self.buf, FMT, *ATTRS)])
        else:
            self.buf.orphan(self._cap)
        self.buf.write(data)
        if debug:
            print("flush: after upload err", self.ctx.error, "bytes", size, flush=True)

        bindings = (
            (TEX_TEXT, self.assets.atlas_texture()),
        )
        fixed = {
            0: self.assets.white,
            1: self.assets.get("NoteBackground"),
            2: self.assets.get("Circle"),
            3: self.assets.get("project_tl_drag"),
            4: self.assets.get("Ring_40px"),
            5: self.assets.get("side"),
            6: self.assets.get("HitFX"),
            8: self.assets.soft,
            9: self.assets.get("HoldLine512px"),
        }
        for unit, tex in fixed.items():
            tex.use(unit)
        for unit, tex in bindings:
            tex.use(unit)
        if debug:
            print("flush: after bind err", self.ctx.error, flush=True)

        self.prog["u_res"].value = (float(width), float(height))
        self.prog["u_circle"].value = tuple(float(v) for v in circle)
        vertex_count = len(self.verts) // 14
        if debug:
            import numpy as np

            arr = np.frombuffer(data, dtype="<f4")
            bad = np.flatnonzero(~np.isfinite(arr))
            if bad.size:
                print("flush: non-finite floats at", bad[:12], flush=True)
            print("flush: before draw err", self.ctx.error, "verts", vertex_count, flush=True)
        if os.environ.get("RIZREC_DEBUG_BIG"):
            import numpy as np

            arr = np.frombuffer(data, dtype="<f4").reshape(-1, 14)
            quads = arr.reshape(-1, 6, 14)
            items = []
            for qi, q in enumerate(quads):
                xs = q[:, 0]
                ys = q[:, 1]
                w = float(xs.max() - xs.min())
                h = float(ys.max() - ys.min())
                if w * h > width * height * 0.15:
                    items.append((w * h, qi, float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()), int(q[0, 13]), tuple(round(float(v), 2) for v in q[0, 4:8])))
            items.sort(reverse=True)
            for item in items[:8]:
                print("bigquad", item, flush=True)
        self.vao.render(vertices=vertex_count)
        if debug:
            print("flush: after draw err", self.ctx.error, flush=True)
        self.verts = array("f")
