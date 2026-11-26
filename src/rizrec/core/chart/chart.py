"""Rizline chart model, parsing and timeline precomputation.

The format is the in-game chart JSON (``fileVersion`` 0):
``bPM``, ``bpmShifts``, ``canvasMoves``, ``lines``, ``cameraMove``,
``themes``, ``challengeTimes``.  One tick is one beat at the base BPM;
``bpmShifts[].value`` is a *multiplier* of the base BPM.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .easing import ease

DEFAULT_BG = (26, 26, 46, 255)
DEFAULT_NOTE = (200, 200, 200, 255)
DEFAULT_POINT = (200, 200, 200, 255)

NOTE_TAP = 0
NOTE_DRAG = 1
NOTE_HOLD = 2


@dataclass(frozen=True, slots=True)
class Color:
    r: int
    g: int
    b: int
    a: int = 255

    def rgba(self) -> tuple[float, float, float, float]:
        return (self.r / 255.0, self.g / 255.0, self.b / 255.0, self.a / 255.0)

    def mix(self, overlay: "Color | None") -> "Color":
        """Alpha-overlay: overlay rgb replaces/lerps, base alpha kept."""
        if overlay is None or overlay.a == 0:
            return self
        if overlay.a == 255:
            return Color(overlay.r, overlay.g, overlay.b, self.a)
        t = overlay.a / 255.0
        return Color(
            round(self.r * (1.0 - t) + overlay.r * t),
            round(self.g * (1.0 - t) + overlay.g * t),
            round(self.b * (1.0 - t) + overlay.b * t),
            self.a,
        )

    def lerp(self, other: "Color", t: float) -> "Color":
        return Color(
            round(self.r + (other.r - self.r) * t),
            round(self.g + (other.g - self.g) * t),
            round(self.b + (other.b - self.b) * t),
            round(self.a + (other.a - self.a) * t),
        )


WHITE = Color(255, 255, 255, 255)
BLACK = Color(0, 0, 0, 255)


@dataclass(slots=True)
class BpmShift:
    time: float
    value: float
    floor: float


class Timing:
    """Piecewise tick <-> seconds conversion (floorPosition anchored)."""

    __slots__ = ("bpm", "times", "floors", "values")

    def __init__(self, bpm: float, shifts: Sequence[BpmShift]):
        self.bpm = bpm
        shifts = sorted(shifts, key=lambda s: s.time) or [BpmShift(0.0, 1.0, 0.0)]
        self.times = [s.time for s in shifts]
        self.floors = [s.floor for s in shifts]
        self.values = [s.value for s in shifts]

    def tick_to_seconds(self, tick: float) -> float:
        times = self.times
        if tick <= times[0]:
            return tick * 60.0 / (self.bpm * self.values[0])
        i = bisect.bisect_right(times, tick) - 1
        if i >= len(times) - 1:
            return self.floors[-1] + (tick - times[-1]) * 60.0 / (
                self.bpm * self.values[-1]
            )
        span = times[i + 1] - times[i]
        if span <= 0:
            return self.floors[i]
        t = (tick - times[i]) / span
        return self.floors[i] + (self.floors[i + 1] - self.floors[i]) * t

    def seconds_to_tick(self, seconds: float) -> float:
        floors = self.floors
        if seconds <= floors[0]:
            return seconds * self.bpm * self.values[0] / 60.0
        i = bisect.bisect_right(floors, seconds) - 1
        if i >= len(floors) - 1:
            return self.times[-1] + (seconds - floors[-1]) * self.bpm * self.values[-1] / 60.0
        span = floors[i + 1] - floors[i]
        if span <= 0:
            return self.times[i]
        t = (seconds - floors[i]) / span
        return self.times[i] + (self.times[i + 1] - self.times[i]) * t


# --------------------------------------------------------------------------
# Keyframes


@dataclass(slots=True)
class KeyPoint:
    time: float
    value: float
    ease: int = 0
    floor: float | None = None


@dataclass(slots=True)
class ColorKeyPoint:
    time: float
    start: Color
    end: Color


def find_value(points: Sequence[KeyPoint], tick: float) -> float:
    """Same semantics as RizPlayer's findValue()."""
    if not points:
        return 0.0
    if len(points) == 1:
        return points[0].value if tick >= points[0].time else 0.0
    last = points[-1]
    if tick > last.time:
        return last.value
    i = bisect.bisect_right([p.time for p in points], tick) - 1
    if i < 0:
        return 0.0
    if i >= len(points) - 1:
        return last.value
    a = points[i]
    b = points[i + 1]
    span = b.time - a.time
    if span <= 0:
        return a.value
    t = ease(a.ease, (tick - a.time) / span)
    return a.value + (b.value - a.value) * t


def color_at(events: Sequence[ColorKeyPoint], tick: float) -> Color | None:
    """Same semantics as RizPlayer's getCurrentLineColor()."""
    if not events:
        return None
    if tick < events[0].time:
        return events[0].start
    if len(events) == 1:
        return events[0].start
    if tick >= events[-1].time:
        return events[-1].start
    i = bisect.bisect_right([e.time for e in events], tick) - 1
    if i < 0:
        return events[0].start
    seg = events[i]
    nxt = events[i + 1]
    span = nxt.time - seg.time
    t = max(0.0, min(1.0, (tick - seg.time) / span)) if span > 0 else 1.0
    return seg.start.lerp(seg.end, t)


# --------------------------------------------------------------------------
# Chart entities


@dataclass(slots=True)
class LinePoint:
    time: float
    x: float
    canvas: int
    ease: int
    color: Color
    floor_pos: float | None = None
    fp: float = 0.0


@dataclass(slots=True)
class CanvasState:
    index: int
    x_points: list[KeyPoint]
    x_times: list[float]
    speed_times: list[float]
    speed_values: list[float]
    speed_floors: list[float]
    activation_tick: float
    x: float = 0.0  # current screen-space x (pixels, relative to centre)
    fp: float = 0.0

    def x_at(self, tick: float) -> float:
        pts = self.x_points
        if not pts:
            return 0.0
        if len(pts) == 1:
            return pts[0].value if tick >= pts[0].time else 0.0
        if tick > pts[-1].time:
            return pts[-1].value
        i = bisect.bisect_right(self.x_times, tick) - 1
        if i < 0:
            return 0.0
        if i >= len(pts) - 1:
            return pts[-1].value
        a = pts[i]
        b = pts[i + 1]
        span = b.time - a.time
        if span <= 0:
            return a.value
        return a.value + (b.value - a.value) * ease(a.ease, (tick - a.time) / span)

    def floor_at(self, seconds: float) -> float:
        if not self.speed_times:
            return 0.0
        i = bisect.bisect_right(self.speed_times, seconds) - 1
        if i < 0:
            i = 0
        return self.speed_floors[i] + (seconds - self.speed_times[i]) * self.speed_values[i]

    def native_floor_at(self, seconds: float) -> float:
        """Floor used by the judged-line visibility window.

        The game picks the last speed key strictly *before* the tick (an
        exact-tick equal belongs to the previous key), unlike ``floor_at``
        which uses ``<=``.  Keep both so notes/rings and line culling can
        follow the game independently.
        """
        if not self.speed_times:
            return 0.0
        i = bisect.bisect_left(self.speed_times, seconds) - 1
        if i < 0:
            i = 0
        return self.speed_floors[i] + (seconds - self.speed_times[i]) * self.speed_values[i]


@dataclass(slots=True)
class NoteState:
    kind: int
    tick: float
    end_tick: float
    start_sec: float
    end_sec: float
    line_index: int
    start_canvas: int
    start_floor: float
    end_canvas: int
    end_floor: float
    # line segment at note time: indices + eased progress
    p0: int
    p1: int
    x_progress: float
    x_ease: int
    # line x endpoints (world units before canvas offset/scale) and eased lerp
    x_world0: float
    x_world1: float
    x_eased: float
    index: int = 0
    # cached hit position (screen x at note time), filled by renderer
    hit_x: float = 0.0


@dataclass(slots=True)
class LineState:
    points: list[LinePoint]
    notes: list[NoteState]
    line_color: list[ColorKeyPoint]
    ring_color: list[ColorKeyPoint]
    times: list[float] = field(default_factory=list)

    def segment_index(self, tick: float) -> int:
        if self.times:
            i = bisect.bisect_right(self.times, tick) - 1
        else:
            i = 0
        return max(0, min(i, len(self.points) - 2))

    def point_index(self, tick: float) -> int:
        """Last point at or before tick (RizPlayer's findPoints)."""
        if self.times:
            i = bisect.bisect_right(self.times, tick) - 1
        else:
            i = 0
        return max(0, min(i, len(self.points) - 1))


@dataclass(slots=True)
class Challenge:
    start: float
    end: float
    transition: float
    checkpoint: float | None


@dataclass(slots=True)
class Transition:
    kind: str  # "entering" / "exiting"
    challenge: int
    event_time: float
    progress: float
    from_theme: int = 0
    to_theme: int = 0
    outside_theme: int = 0
    inside_theme: int = 0
    mask_progress: float = 0.0
    color_progress: float = 0.0


@dataclass(slots=True)
class ThemeState:
    active_challenge: int = -1
    active_theme: int = 0
    transition: Transition | None = None


# --------------------------------------------------------------------------
# Parsing helpers


def _fmt_color(raw: dict | None, fallback: Color = WHITE) -> Color:
    if not raw:
        return fallback
    return Color(
        max(0, min(255, int(raw.get("r", 255)))),
        max(0, min(255, int(raw.get("g", 255)))),
        max(0, min(255, int(raw.get("b", 255)))),
        max(0, min(255, int(raw.get("a", 255)))),
    )


def _keypoints(raw: Iterable[dict], default_floor: float | None = None) -> list[KeyPoint]:
    out: list[KeyPoint] = []
    for p in raw:
        out.append(
            KeyPoint(
                time=float(p.get("time", 0.0)),
                value=float(p.get("value", 0.0)),
                ease=int(p.get("easeType", 0) or 0),
                floor=None if p.get("floorPosition") is None else float(p["floorPosition"]),
            )
        )
    out.sort(key=lambda k: k.time)
    return out


def _color_keypoints(raw: Iterable[dict]) -> list[ColorKeyPoint]:
    out = [
        ColorKeyPoint(
            time=float(p.get("time", 0.0)),
            start=_fmt_color(p.get("startColor"), WHITE),
            end=_fmt_color(p.get("endColor"), WHITE),
        )
        for p in raw
    ]
    out.sort(key=lambda k: k.time)
    return out


# --------------------------------------------------------------------------
# Main chart object


class Chart:
    def __init__(self, data: dict):
        self.raw = data
        self.file_version = int(data.get("fileVersion", 0) or 0)
        self.bpm = float(data.get("bPM", 120.0))
        if not math.isfinite(self.bpm) or self.bpm <= 0:
            raise ValueError("bPM must be a positive number")

        shifts = [
            BpmShift(
                time=float(s.get("time", 0.0)),
                value=float(s.get("value", 1.0)),
                floor=float(s.get("floorPosition", 0.0)),
            )
            for s in data.get("bpmShifts", [])
        ]
        self.timing = Timing(self.bpm, shifts)

        self.canvases: list[CanvasState] = []
        self.canvas_by_index: dict[int, CanvasState] = {}
        for i, raw in enumerate(data.get("canvasMoves", [])):
            index = int(raw.get("index", i))
            xp = _keypoints(raw.get("xPositionKeyPoints", []))
            sp = _keypoints(raw.get("speedKeyPoints", []))
            times = [self.timing.tick_to_seconds(p.time) for p in sp]
            floors: list[float] = []
            values: list[float] = []
            prev_floor = 0.0
            prev_time = 0.0
            for j, p in enumerate(sp):
                speed = float(p.value)
                if p.floor is not None:
                    # Explicit precomputed value: use it as an anchor.
                    floor = p.floor
                else:
                    floor = prev_floor + (times[j] - prev_time) * (values[-1] if values else speed)
                floors.append(floor)
                values.append(speed)
                prev_floor = floor
                prev_time = times[j]
            act_ticks = [p.time for p in (xp[:1] + sp[:1])]
            activation = min(act_ticks) if act_ticks else -math.inf
            canvas = CanvasState(
                index, xp, [p.time for p in xp], times, values, floors, activation
            )
            self.canvases.append(canvas)
            self.canvas_by_index[index] = canvas

        default_scale = [KeyPoint(0.0, 1.0, 0)]
        default_x = [KeyPoint(0.0, 0.0, 0)]
        camera = data.get("cameraMove") or {}
        self.camera_scale = _keypoints(camera.get("scaleKeyPoints", [])) or default_scale
        self.camera_x = _keypoints(camera.get("xPositionKeyPoints", [])) or default_x

        self.themes: list[list[Color]] = []
        for theme in data.get("themes", []):
            self.themes.append([_fmt_color(c, WHITE) for c in theme.get("colorsList", [])])
        if not self.themes:
            self.themes = [
                [Color(*DEFAULT_BG), Color(*DEFAULT_NOTE), WHITE]
            ]

        self.challenges: list[Challenge] = []
        for c in data.get("challengeTimes", []):
            start = float(c.get("start", 0.0))
            # ``transTime`` is authored in seconds while the theme state works
            # in ticks; convert with the local tick length at the challenge.
            trans_seconds = float(c.get("transTime", 0.0) or 0.0)
            step = self.timing.tick_to_seconds(start + 1.0) - self.timing.tick_to_seconds(start)
            self.challenges.append(
                Challenge(
                    start=start,
                    end=float(c.get("end", 0.0)),
                    transition=trans_seconds / step if step > 0.0 else 0.0,
                    checkpoint=None if c.get("checkPoint") is None else float(c["checkPoint"]),
                )
            )
        self.challenges.sort(key=lambda c: c.start)

        self.chart_delay = float(data.get("chartDelayMs", 0.0) or 0.0) / 1000.0
        self.song_name = data.get("songsName") or data.get("songName") or ""

        # lines / notes -----------------------------------------------------
        self.lines: list[LineState] = []
        notes: list[NoteState] = []
        max_tick = 0.0
        for line_index, raw in enumerate(data.get("lines", [])):
            points: list[LinePoint] = []
            for p in raw.get("linePoints", []):
                canvas_index = int(p.get("canvasIndex", 0))
                point = LinePoint(
                    time=float(p.get("time", 0.0)),
                    x=float(p.get("xPosition", 0.0)),
                    canvas=canvas_index,
                    ease=int(p.get("easeType", 0) or 0),
                    color=_fmt_color(p.get("color"), Color(*DEFAULT_POINT)),
                    floor_pos=None if p.get("floorPosition") is None else float(p["floorPosition"]),
                )
                points.append(point)
            points.sort(key=lambda p: p.time)
            if not points:
                continue
            # Explicit floorPosition first, otherwise integrate the canvas.
            for point in points:
                if point.floor_pos is not None:
                    point.fp = point.floor_pos
                else:
                    point.fp = self._canvas(point.canvas).floor_at(
                        self.timing.tick_to_seconds(point.time)
                    )
            max_tick = max(max_tick, points[-1].time)

            line = LineState(
                points=points,
                notes=[],
                line_color=_color_keypoints(raw.get("lineColor", [])),
                ring_color=_color_keypoints(raw.get("judgeRingColor", [])),
                times=[p.time for p in points],
            )

            raw_notes = raw.get("notes", []) or []
            raw_notes.sort(key=lambda n: float(n.get("time", 0.0)))
            for note_raw in raw_notes:
                ntype = int(note_raw.get("type", -1))
                if ntype not in (NOTE_TAP, NOTE_DRAG, NOTE_HOLD):
                    continue
                tick = float(note_raw.get("time", 0.0))
                other = note_raw.get("otherInformations", []) or []
                end_tick = float(other[0]) if ntype == NOTE_HOLD and other else tick
                start_sec = self.timing.tick_to_seconds(tick)
                end_sec = self.timing.tick_to_seconds(end_tick)
                seg = line.point_index(tick)
                p0 = line.points[seg]
                p1 = line.points[seg + 1] if seg + 1 < len(line.points) else p0
                start_canvas = p0.canvas
                start_floor = self._canvas(start_canvas).floor_at(start_sec)
                end_canvas = start_canvas
                end_floor = start_floor
                if ntype == NOTE_HOLD:
                    if len(other) > 1:
                        end_canvas = int(round(float(other[1])))
                    if len(other) > 2:
                        end_floor = float(other[2])
                    else:
                        end_floor = self._canvas(end_canvas).floor_at(end_sec)
                    max_tick = max(max_tick, end_tick)

                span = p1.time - p0.time
                progress = 0.0 if span <= 0 else (tick - p0.time) / span
                x_progress = max(0.0, min(1.0, progress))
                x_ease = ease(p0.ease, x_progress)
                x0 = p0.x
                x1 = p1.x
                note = NoteState(
                    kind=ntype,
                    tick=tick,
                    end_tick=end_tick,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    line_index=line_index,
                    start_canvas=start_canvas,
                    start_floor=start_floor,
                    end_canvas=end_canvas,
                    end_floor=end_floor,
                    p0=seg,
                    p1=seg + 1 if seg + 1 < len(line.points) else seg,
                    x_progress=x_progress,
                    x_ease=p0.ease,
                    x_world0=x0,
                    x_world1=x1,
                    x_eased=x_ease,
                    index=len(notes),
                )
                line.notes.append(note)
                notes.append(note)

            self.lines.append(line)

        # The game sorts line groups by depth: depth = 0.6 - startTime*1e-5,
        # painted back-to-front.  That means an earlier first point is lower
        # (painted first); the static backdrop (early ticks) ends up at the very
        # bottom, and later lines stay visible on top while it fades.
        self.draw_lines = sorted(self.lines, key=lambda ln: ln.points[0].time)

        notes.sort(key=lambda n: n.start_sec)
        for i, note in enumerate(notes):
            note.index = i
        self.notes = notes
        self.note_starts = [n.start_sec for n in notes]
        # Tail HitFX must be found by end time: a long hold starts outside the
        # 0.55 s effect window even though its end is inside it.
        self.hold_ends = sorted(
            ((n.end_sec, n) for n in notes if n.kind == NOTE_HOLD), key=lambda item: item[0]
        )
        self.hold_end_secs = [end for end, _ in self.hold_ends]
        # Last moment a playable note is still on screen (hold tails included).
        # The outro card must not start before it (RizPlayer's `notesEnd`).
        self.notes_end = max(
            (n.end_sec if n.kind == NOTE_HOLD else n.start_sec for n in notes),
            default=0.0,
        )

        for challenge in self.challenges:
            max_tick = max(max_tick, challenge.end + challenge.transition)
        self.max_tick = max_tick
        self.duration = self.timing.tick_to_seconds(max_tick) + self.chart_delay + 1.5

        self._theme_cache: dict[tuple[float, float], ThemeState] = {}

    # -- helpers ---------------------------------------------------------

    def _canvas(self, index: int) -> CanvasState:
        try:
            return self.canvas_by_index[index]
        except KeyError:
            # Charts occasionally reference a missing canvas; use index 0.
            return self.canvases[0]

    @property
    def note_count(self) -> int:
        return len(self.notes)

    def theme_index(self, challenge_index: int) -> int:
        if challenge_index < 0 or len(self.themes) < 2:
            return 0
        return min(challenge_index + 1, len(self.themes) - 1)

    def theme_color(self, theme_index: int, color_index: int) -> Color:
        if not self.themes:
            return WHITE
        ti = max(0, min(theme_index, len(self.themes) - 1))
        colors = self.themes[ti]
        if 0 <= color_index < len(colors):
            return colors[color_index]
        colors = self.themes[0]
        if 0 <= color_index < len(colors):
            return colors[color_index]
        if color_index == 0:
            return Color(*DEFAULT_BG)
        if color_index in (1, 2):
            return Color(*DEFAULT_NOTE)
        return WHITE

    # -- Riztime ---------------------------------------------------------

    def _find_top_challenge(self, tick: float, exclude: int = -1, entering: int = -1) -> int:
        best = -1
        best_start = -math.inf
        for i, ch in enumerate(self.challenges):
            if i == exclude:
                continue
            if tick < ch.start or tick >= ch.end:
                continue
            if entering >= 0 and i != entering:
                if ch.start == self.challenges[entering].start and i > entering:
                    continue
            if ch.start > best_start or (ch.start == best_start and i > best):
                best_start = ch.start
                best = i
        return best

    def theme_state(self, tick: float, lookahead: float = 0.0) -> ThemeState:
        active = self._find_top_challenge(tick)
        state = ThemeState(active, self.theme_index(active), None)
        if not self.challenges:
            return state

        chosen: Transition | None = None
        chosen_key = None
        for i, ch in enumerate(self.challenges):
            trans = ch.transition
            if trans <= 0:
                continue
            if ch.start <= tick < ch.start + trans + lookahead:
                progress = max(0.0, min(1.0, (tick + lookahead - ch.start) / trans))
                cand = Transition("entering", i, ch.start, progress)
                key = (cand.event_time, 2, i)
                if chosen_key is None or key > chosen_key:
                    chosen, chosen_key = cand, key
            if ch.end <= tick < ch.end + trans + lookahead:
                progress = max(0.0, min(1.0, (tick + lookahead - ch.end) / trans))
                cand = Transition("exiting", i, ch.end, progress)
                key = (cand.event_time, 1, i)
                if chosen_key is None or key > chosen_key:
                    chosen, chosen_key = cand, key

        if chosen is None:
            return state

        challenge_theme = self.theme_index(chosen.challenge)
        underlying = self._find_top_challenge(
            chosen.event_time,
            exclude=chosen.challenge,
            entering=chosen.challenge if chosen.kind == "entering" else -1,
        )
        underlying_theme = self.theme_index(underlying)
        eased_forward = ease(2, chosen.progress)
        eased_reverse = ease(2, 1.0 - chosen.progress)
        if chosen.kind == "entering":
            chosen.from_theme = underlying_theme
            chosen.to_theme = challenge_theme
            chosen.outside_theme = underlying_theme
            chosen.inside_theme = challenge_theme
            chosen.mask_progress = eased_forward
            chosen.color_progress = eased_forward
        else:
            chosen.from_theme = challenge_theme
            chosen.to_theme = underlying_theme
            chosen.outside_theme = underlying_theme
            chosen.inside_theme = challenge_theme
            chosen.mask_progress = eased_reverse
            chosen.color_progress = 1.0 - eased_reverse
        state.transition = chosen
        return state

    def effect_color(self, tick: float, state: ThemeState | None = None) -> Color:
        st = state or self.theme_state(tick)
        if st.transition is None:
            return self.theme_color(st.active_theme, 2)
        return self.theme_color(st.transition.from_theme, 2).lerp(
            self.theme_color(st.transition.to_theme, 2),
            st.transition.color_progress,
        )


def load_chart(path: str) -> Chart:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    chart = Chart(data)
    if not chart.song_name:
        stem = os.path.splitext(os.path.basename(path))[0]
        chart.song_name = re.sub(r"_(IN|AT|HD|EZ|SP|MR|CN)$", "", stem, flags=re.IGNORECASE)
    return chart
