"""HitFX particle maths, ported from the game (via RizPlayer's native port).

The effect is an independent world node: distances use ``height / 10`` pixels
per unit and the 1.55 prefab scale, ignoring the chart camera zoom.  Particles
are deterministic per judgement tick and emission key so preview and export
frame identical effects.
"""

from __future__ import annotations

import math

HITFX_DURATION = 0.55
PARTICLE_LIFETIME = 0.5
PARTICLE_INTERVAL = 0.01
PARTICLE_COUNT = 5
PARTICLE_FADE_HOLD = 32768.0 / 65535.0
PARTICLE_SIZE_HOLD = 0.5004164
PARTICLE_SIZE_END_SLOPE = -4.257264
MOTION_SPEED_MIN = 3.0
MOTION_SPEED_RANGE = 5.0
MOTION_RADIUS = 0.20
MOTION_INNER_RADIUS_SQ = 0.64

SPRITE_WORLD_SIZE = 655.0 / 576.0
PREFAB_SCALE = 1.55
ATLAS_FRAME_SIZE = 256
ATLAS_COLUMNS = 7
FRAME_FIRST = 1
FRAME_LAST = 40
FPS = 60.0

PERFECT_LIFETIME = 0.5
PERFECT_SPRAY_RATE = 100.0
PERFECT_SPRAY_COUNT = 5
PERFECT_SPRAY_SCALE = 1.55
PERFECT_TRAVEL_SAMPLES = 100


class Rng:
    """xorshift32, matching the deterministic particle sampler."""

    def __init__(self, seed: int):
        self.s = (seed & 0xFFFFFFFF) or 1

    def next(self) -> float:
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s & 0xFFFFFFFF
        return self.s / 0x100000000


def _particle_seed(judgement_tick: float, emission_key: int) -> int:
    a = (int(round(judgement_tick * 65536)) * 0x9E3779B1) & 0xFFFFFFFF
    b = ((emission_key + 1) * 0x85EBCA6B) & 0xFFFFFFFF
    return (a ^ b) & 0xFFFFFFFF or 1


_particle_cache: dict[tuple[int, int], list[dict]] = {}
_spray_cache: dict[int, list[dict]] = {}


def create_particles(judgement_tick: float, emission_key: int) -> list[dict]:
    key = (int(round(judgement_tick * 65536)), emission_key)
    cached = _particle_cache.get(key)
    if cached is not None:
        return cached
    rng = Rng(_particle_seed(judgement_tick, emission_key))
    particles = []
    for i in range(PARTICLE_COUNT):
        angle = rng.next() * math.tau
        particles.append(
            {
                "delay": (i + 1) * PARTICLE_INTERVAL,
                "dx": math.cos(angle),
                "dy": math.sin(angle),
                "diameter": 0.15 + rng.next() * 0.15,
            }
        )
    for i, p in enumerate(particles):
        p["launch_speed"] = MOTION_SPEED_MIN + rng.next() * MOTION_SPEED_RANGE
        p["start_radius"] = MOTION_RADIUS * math.sqrt(
            MOTION_INNER_RADIUS_SQ + (1.0 - MOTION_INNER_RADIUS_SQ) * rng.next()
        )
        p["phase"] = i
    if len(_particle_cache) > 4096:
        _particle_cache.clear()
    _particle_cache[key] = particles
    return particles


def _drag(u: float) -> float:
    u2 = u * u
    u3 = u2 * u
    return (
        (2 * u3 - 3 * u2 + 1) * 0.00885009765625
        + (u3 - 2 * u2 + u) * 24.159265518188477
        + (3 * u2 - 2 * u3)
        - (u3 - u2) * 8.176969528198242
    )


_travel_cache: dict[tuple[int, int], float] = {}


def particle_travel(age: float, speed: float) -> float:
    """Integrate the drag curve from the particle's birth (0.25 ms steps)."""
    if age <= 0.0:
        return 0.0
    age = min(age, PARTICLE_LIFETIME)
    key = (int(round(age * 2000)), int(round(speed * 1000)))
    cached = _travel_cache.get(key)
    if cached is not None:
        return cached
    steps = max(1, key[0])
    dt = age / steps
    offset = 0.0
    v = speed
    lifetime = PARTICLE_LIFETIME
    for i in range(steps):
        u = (i * dt) / lifetime
        # Inlined _drag(u) (same operations/order): the function call was the
        # dominant cost in this loop and this runs ~thousands of steps/frame.
        u2 = u * u
        u3 = u2 * u
        drag = (
            (2 * u3 - 3 * u2 + 1) * 0.00885009765625
            + (u3 - 2 * u2 + u) * 24.159265518188477
            + (3 * u2 - 2 * u3)
            - (u3 - u2) * 8.176969528198242
        )
        nv = v - dt * drag * v * v
        # max(0.0, nv) without the builtin call (same result).
        v = nv if nv > 0.0 else 0.0
        offset += dt * v
    if len(_travel_cache) > 20000:
        _travel_cache.clear()
    _travel_cache[key] = offset
    return offset


def particle_size(progress: float) -> float:
    t = max(0.0, (progress - PARTICLE_SIZE_HOLD) / (1.0 - PARTICLE_SIZE_HOLD))
    t2 = t * t
    return max(
        0.0,
        1.0
        + t2 * (2.0 * t - 3.0)
        + t2 * (t - 1.0) * (1.0 - PARTICLE_SIZE_HOLD) * PARTICLE_SIZE_END_SLOPE,
    )


def particle_alpha(progress: float) -> float:
    return min(1.0, (1.0 - progress) / (1.0 - PARTICLE_FADE_HOLD))


def _build_perfect_travel() -> list[float]:
    table = [0.0] * (PERFECT_TRAVEL_SAMPLES + 1)
    out_weight = 0.078651674
    out_value = 1.0 - 9.493332863 * out_weight
    in_value = 0.104364902
    previous = 1.0
    for i in range(1, PERFECT_TRAVEL_SAMPLES + 1):
        progress = i / PERFECT_TRAVEL_SAMPLES
        low, high = 0.0, 1.0
        for _ in range(16):
            param = (low + high) * 0.5
            x = 3.0 * (1.0 - param) ** 2 * param * out_weight + param**3
            if x < progress:
                low = param
            else:
                high = param
        param = (low + high) * 0.5
        inv = 1.0 - param
        speed = (
            inv**3
            + 3.0 * inv**2 * param * out_value
            + 3.0 * inv * param**2 * in_value
        )
        table[i] = table[i - 1] + (previous + speed) / (2.0 * PERFECT_TRAVEL_SAMPLES)
        previous = speed
    return table


PERFECT_TRAVEL = _build_perfect_travel()


def perfect_sample(progress: float) -> float:
    sample = max(0.0, min(1.0, progress)) * PERFECT_TRAVEL_SAMPLES
    index = min(int(math.floor(sample)), PERFECT_TRAVEL_SAMPLES - 1)
    return PERFECT_TRAVEL[index] + (sample - index) * (
        PERFECT_TRAVEL[index + 1] - PERFECT_TRAVEL[index]
    )


def create_spray(judgement_tick: float) -> list[dict]:
    key = int(round(judgement_tick * 65536))
    cached = _spray_cache.get(key)
    if cached is not None:
        return cached
    rng = Rng(_particle_seed(judgement_tick, 0x5BD1E995))
    spray = []
    for i in range(PERFECT_SPRAY_COUNT):
        angle = math.radians(86.0 + rng.next() * 8.0)
        spray.append(
            {
                "delay": (i + 1) / PERFECT_SPRAY_RATE,
                "x_dir": math.cos(angle),
                "y_dir": -math.sin(angle),
                "speed": 25.0 + rng.next() * 25.0,
                "diameter": 0.05 + rng.next() * 0.05,
                "start_radius": 0.2 * (0.8 + rng.next() * 0.2),
            }
        )
    if len(_spray_cache) > 4096:
        _spray_cache.clear()
    _spray_cache[key] = spray
    return spray


def spray_size(progress: float) -> float:
    t = max(0.0, (progress - PARTICLE_SIZE_HOLD) / (1.0 - PARTICLE_SIZE_HOLD))
    square = t * t
    return max(
        0.0,
        1.0
        + square * (2.0 * t - 3.0)
        - square * (t - 1.0) * 4.257264 * (1.0 - PARTICLE_SIZE_HOLD),
    )
