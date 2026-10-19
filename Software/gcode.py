from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_MOVE_RE = re.compile(r"^G([0123])\b", re.IGNORECASE)
_WORD_RE = re.compile(r"([XYZEF])\s*(-?\d*\.?\d+)", re.IGNORECASE)
_LAYER_RE = re.compile(r"^;\s*LAYER:\s*(-?\d+)", re.IGNORECASE)
_TYPE_RE = re.compile(r"^;\s*TYPE:\s*(\S+)", re.IGNORECASE)

_NON_MOTION = {
    "M104", "M109", "M140", "M190", "M141", "M191",   # temperatures
    "M106", "M107",                                    # fans
    "M84", "M18", "M17",                               # steppers
    "M105", "M115", "M117", "M118", "M73", "M400",     # status / sync
    "M201", "M203", "M204", "M205", "M220", "M221",    # limits / factors
    "M300", "M302", "M400", "M420", "M500", "M501",
    "M900", "M204", "M107", "M82", "M83",
    "G4",                                              # dwell
    "T0", "T1",                                        # tool select
}


_NON_MODEL_FEATURE_MARKERS = ("skirt", "brim", "raft", "support", "prime")


def is_non_model_feature(feature: str) -> bool:
    f = (feature or "").lower()
    return any(mark in f for mark in _NON_MODEL_FEATURE_MARKERS)


@dataclass
class Move:
    x0: float; y0: float; z0: float
    x1: float; y1: float; z1: float
    t0: float; t1: float           # cumulative print-time bounds, seconds
    extruding: bool
    layer: int
    line: int                      # source line number, for step 3's annotator
    feature: str = ""              # Cura ;TYPE: (WALL-OUTER, SKIN, ...)


@dataclass
class Toolpath:
    moves: list[Move]
    total_time: float
    layer_end_time: dict[int, float]
    warnings: list[str] = field(default_factory=list)
    source: str = ""

    def bounds_time(self, layer: int) -> float | None:
        return self.layer_end_time.get(layer)

    @property
    def end_times(self) -> list:
        """Move end-times, for binary search. Built once and cached.

        Rebuilding this per call is O(number of moves) on a lookup that
        should be O(log n), which is invisible on a toy file and fatal
        on a real one: a 200-layer print is ~100k moves and ~150k replay
        samples, so the planner went from seconds to hours. Cached here
        rather than at each call site so every caller gets it."""
        cached = getattr(self, "_end_times", None)
        if cached is None or len(cached) != len(self.moves):
            cached = [m.t1 for m in self.moves]
            object.__setattr__(self, "_end_times", cached)
        return cached

    @property
    def bbox(self):
        """(xmin,ymin,zmin),(xmax,ymax,zmax) over EXTRUDING moves only --
        travels can legitimately run off to a prime line or wipe.

        Includes brim/skirt/support: this is the extent of everything
        actually deposited, which is what collision checking needs."""
        return self._bbox_over(lambda m: m.extruding)

    @property
    def model_bbox(self):
        """Extent of the PART alone -- brim, skirt, raft, support and
        prime tower excluded.

        This is what to match against an STL's bounding box. Cura's
        default 8mm brim adds ~15.6mm to each axis, so comparing the
        plain bbox against the chunk makes a correctly sliced file look
        like the wrong object entirely. Falls back to the full bbox when
        the file has no ;TYPE: markers to tell them apart."""
        pts = self._bbox_over(
            lambda m: m.extruding and not is_non_model_feature(m.feature))
        if pts == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)):
            return self.bbox
        return pts

    def _bbox_over(self, keep):
        pts = []
        for m in self.moves:
            if keep(m):
                pts.append((m.x0, m.y0, m.z0))
                pts.append((m.x1, m.y1, m.z1))
        if not pts:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs, ys, zs = zip(*pts)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def feature_extents(self) -> dict:
        """Per-;TYPE: XY extents, for diagnosing a size mismatch."""
        out: dict = {}
        for m in self.moves:
            if not m.extruding:
                continue
            f = m.feature or "(untyped)"
            lo, hi = out.get(f, ((1e9, 1e9), (-1e9, -1e9)))
            out[f] = ((min(lo[0], m.x0, m.x1), min(lo[1], m.y0, m.y1)),
                      (max(hi[0], m.x0, m.x1), max(hi[1], m.y0, m.y1)))
        return out


def parse(path: Path, default_feedrate: float = 1500.0) -> Toolpath:
    x = y = z = 0.0
    e = 0.0                 # current absolute extruder position
    f = default_feedrate
    t = 0.0
    layer = -1
    feature = ""
    relative_e = False      # M83
    relative_xyz = False    # G91
    homed = False
    moves: list[Move] = []
    layer_end: dict[int, float] = {}
    warnings: list[str] = []
    unknown: dict[str, int] = {}

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh):
            stripped = raw.strip()

            lm = _LAYER_RE.match(stripped)
            if lm:
                layer = int(lm.group(1))
                continue
            tm = _TYPE_RE.match(stripped)
            if tm:
                feature = tm.group(1)
                continue

            code = raw.split(";", 1)[0].strip()
            if not code:
                continue
            head = code.split()[0].upper()

            # --- state-only commands, no motion -----------------------
            if head in ("M82",):
                relative_e = False
                continue
            if head in ("M83",):
                relative_e = True
                continue
            if head == "G90":
                relative_xyz = False
                continue
            if head == "G91":
                relative_xyz = True
                continue
            if head == "G92":
                words = {k.upper(): float(v) for k, v in _WORD_RE.findall(code)}
                # A reset of the coordinate system, NOT a move.
                if "E" in words:
                    e = words["E"]
                x = words.get("X", x)
                y = words.get("Y", y)
                z = words.get("Z", z)
                continue
            if head == "G28":
                # Homing teleports; record the new position without
                # emitting a move (see module docstring).
                words = {k.upper(): float(v) for k, v in _WORD_RE.findall(code)}
                axes = [a for a in ("X", "Y", "Z") if a in words] or ["X", "Y", "Z"]
                if "X" in axes: x = 0.0
                if "Y" in axes: y = 0.0
                if "Z" in axes: z = 0.0
                homed = True
                continue

            m = _MOVE_RE.match(code)
            if not m:
                if head not in _NON_MOTION and head.startswith(("M", "T", "G")):
                    unknown[head] = unknown.get(head, 0) + 1
                continue
            if m.group(1) in ("2", "3"):
                unknown["G2/G3 arc"] = unknown.get("G2/G3 arc", 0) + 1
                continue

            words = {k.upper(): float(v) for k, v in _WORD_RE.findall(code)}
            if "F" in words:
                f = words["F"]

            if relative_xyz:
                nx = x + words.get("X", 0.0)
                ny = y + words.get("Y", 0.0)
                nz = z + words.get("Z", 0.0)
            else:
                nx, ny, nz = words.get("X", x), words.get("Y", y), words.get("Z", z)

            if "E" in words:
                de = words["E"] if relative_e else words["E"] - e
                e = e + words["E"] if relative_e else words["E"]
            else:
                de = 0.0

            dist = math.dist((x, y, z), (nx, ny, nz))
            if dist > 0:
                dt = dist / max(f / 60.0, 1e-9)
                moves.append(Move(x, y, z, nx, ny, nz, t, t + dt,
                                   de > 1e-9, layer, line_no, feature))
                t += dt
                layer_end[layer] = t
            x, y, z = nx, ny, nz

    if not homed:
        warnings.append("no G28 in this file -- assuming it starts homed at (0,0,0)")
    if layer < 0:
        warnings.append("no ';LAYER:' markers found -- layer-dependency events "
                         "(the sloped-seam ordering) cannot be generated. In Cura "
                         "this comes from the default 'Layer' comment setting.")
    for k, n in unknown.items():
        warnings.append(f"skipped {n} x {k} (not modelled)")
    if not moves:
        raise ValueError(f"{path}: no usable G0/G1 moves found -- is this a G-code file?")

    return Toolpath(moves, t, layer_end, warnings, source=str(path))


def position_at(tp: Toolpath, t: float) -> tuple[float, float, float]:
    """Local (x,y,z) at print-time t, uninterrupted. A pure function of
    (toolpath, t) -- no state carried between calls, so it cannot go
    stale the way the old per-move progress counter could."""
    if not tp.moves:
        return (0.0, 0.0, 0.0)
    if t <= tp.moves[0].t0:
        m = tp.moves[0]
        return (m.x0, m.y0, m.z0)
    if t >= tp.total_time:
        m = tp.moves[-1]
        return (m.x1, m.y1, m.z1)
    i = bisect.bisect_left(tp.end_times, t)
    m = tp.moves[i]
    span = m.t1 - m.t0
    frac = 0.0 if span <= 0 else (t - m.t0) / span
    return (m.x0 + frac * (m.x1 - m.x0),
            m.y0 + frac * (m.y1 - m.y0),
            m.z0 + frac * (m.z1 - m.z0))


def describe(tp: Toolpath) -> str:
    lo, hi = tp.bbox
    n_ext = sum(1 for m in tp.moves if m.extruding)
    layers = sorted(l for l in tp.layer_end_time if l >= 0)
    return (f"{Path(tp.source).name}: {len(tp.moves)} moves "
            f"({n_ext} extruding), {len(layers)} layers, "
            f"{tp.total_time:.0f}s modelled, "
            f"extents X[{lo[0]:.1f},{hi[0]:.1f}] Y[{lo[1]:.1f},{hi[1]:.1f}] "
            f"Z[{lo[2]:.1f},{hi[2]:.1f}]")
