"""Put a Cura-sliced chunk back where step 1 intended it to sit.

Cura re-centres every model it loads, and there is no dependable way to
switch that off across versions. Loading both chunks therefore stacks
them in the middle of the bed, and step 3 correctly reports that the two
halves cover the same region -- the plan would then be built around a
part that does not exist.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import gcode

_MOVE_RE = re.compile(r"^\s*(G[0123])(\b.*)$", re.IGNORECASE)
_XY_RE = re.compile(r"\b([XY])\s*(-?\d*\.?\d+)", re.IGNORECASE)


@dataclass
class Placement:
    dx: float
    dy: float
    measured: tuple
    intended: tuple

    @property
    def moved(self) -> bool:
        return abs(self.dx) > 0.05 or abs(self.dy) > 0.05

    def describe(self, name: str) -> str:
        mlo, mhi = self.measured
        ilo, ihi = self.intended
        return (f"{name}: model sliced at X[{mlo[0]:.1f},{mhi[0]:.1f}] "
                f"Y[{mlo[1]:.1f},{mhi[1]:.1f}], "
                f"needs X[{ilo[0]:.1f},{ihi[0]:.1f}] Y[{ilo[1]:.1f},{ihi[1]:.1f}] "
                f"-> shift ({self.dx:+.2f}, {self.dy:+.2f}) mm")


def adhesion_note(tp: gcode.Toolpath) -> str | None:
    """One line about brim/skirt, if present -- so a user who did not
    knowingly enable it can see it is there and how much room it takes."""
    feats = tp.feature_extents()
    extra = {f: e for f, e in feats.items() if gcode.is_non_model_feature(f)}
    if not extra:
        return None
    mlo, mhi = tp.model_bbox
    alo, ahi = tp.bbox
    pad_x = max(mlo[0] - alo[0], ahi[0] - mhi[0])
    pad_y = max(mlo[1] - alo[1], ahi[1] - mhi[1])
    return (f"includes {', '.join(sorted(extra))} extending up to "
            f"{max(pad_x, pad_y):.1f}mm beyond the part "
            f"(excluded when matching position, still printed and still "
            f"moved with it)")


def fingerprint(tp: gcode.Toolpath) -> dict:
    """Identify the exact toolpath a plan was built against.

    A plan's events are move INDICES into a specific parse of a specific
    file. Handing steps 4/5 a different file (typically the un-placed
    original instead of the placed one) leaves the indices in range but
    pointing at the wrong moves, which models a print that will not
    happen -- a silent wrong answer rather than an error. Cheap to
    record, so it is recorded."""
    lo, hi = tp.bbox
    return {"moves": len(tp.moves),
            "bbox_min": [round(v, 3) for v in lo],
            "bbox_max": [round(v, 3) for v in hi]}


def check_fingerprint(tp: gcode.Toolpath, expected: dict, head: str) -> list[str]:
    """Empty list if this toolpath is the one the plan was built for."""
    if not expected:
        return []
    got = fingerprint(tp)
    problems = []
    if got["moves"] != expected.get("moves"):
        problems.append(f"{head}: {Path(tp.source).name} has {got['moves']} moves, "
                         f"the plan was built against {expected.get('moves')}")
    for key in ("bbox_min", "bbox_max"):
        a, b = got[key], expected.get(key)
        if b and any(abs(x - y) > 0.05 for x, y in zip(a, b)):
            problems.append(f"{head}: {Path(tp.source).name} {key} is {a}, "
                             f"the plan was built against {b}")
    return problems


def load_placement(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if "chunks" not in data:
        raise ValueError(f"{path} is not a placement file from step 1")
    return data


def compute(tp: gcode.Toolpath, intended_min, intended_max,
            size_tolerance: float = 1.5) -> Placement:
    """Offset that moves this toolpath onto its intended footprint.

    Matches on the MODEL's extent, not everything deposited: Cura's
    default 8mm brim adds ~15.6mm to each axis, which would otherwise
    make a perfectly good slice look like a completely different object.
    The brim still gets shifted along with everything else -- it is real
    material and the toolhead really does travel over it -- it just does
    not get a vote on where the part belongs.

    The tolerance absorbs the outer wall being inset half a line width
    from the model surface (~0.4mm on each axis at default settings).

    Raises if the sliced object is still not the same SIZE as the chunk,
    since that means the G-code is not a slice of this chunk at all
    (wrong file, scaled or rotated model) and shifting it would only
    disguise the problem."""
    lo, hi = tp.model_bbox
    for axis, name in ((0, "X"), (1, "Y")):
        got = hi[axis] - lo[axis]
        want = intended_max[axis] - intended_min[axis]
        if abs(got - want) > size_tolerance:
            raise ValueError(
                f"{Path(tp.source).name} is {got:.1f}mm across in {name} (model "
                f"only, brim/skirt excluded) but the chunk it should be is "
                f"{want:.1f}mm.\n" + _diagnose(tp, got - want))
    return Placement(dx=intended_min[0] - lo[0], dy=intended_min[1] - lo[1],
                     measured=(lo, hi), intended=(intended_min, intended_max))


def _diagnose(tp: gcode.Toolpath, delta: float) -> str:
    """Say something useful about WHY the size is wrong, instead of
    leaving the user to guess between a dozen possibilities."""
    lines = []
    feats = tp.feature_extents()
    if feats:
        lines.append("  what this file contains:")
        for f, (lo, hi) in sorted(feats.items()):
            tag = "  (excluded from the match)" if gcode.is_non_model_feature(f) else ""
            lines.append(f"    {f:<20s} X[{lo[0]:7.1f},{hi[0]:7.1f}] "
                          f"Y[{lo[1]:7.1f},{hi[1]:7.1f}]{tag}")
    else:
        lines.append("  this file has no ;TYPE: markers, so brim and skirt "
                      "cannot be told apart from the part.")
        lines.append("  In Cura: Preferences -> Configure Cura -> Settings, or "
                      "check the 'Layer'/'Feature' comment options are on.")
    if delta > 0:
        lines.append(f"  it is {delta:.1f}mm TOO BIG. Common causes: the model was "
                      f"scaled in Cura, or an adhesion type this build does not "
                      f"recognise is being counted as part of the model.")
    else:
        lines.append(f"  it is {-delta:.1f}mm TOO SMALL. Common causes: the wrong "
                      f"chunk paired with the wrong G-code, or the model was "
                      f"scaled down.")
    lines.append("  Check the file pairing, and that neither chunk was scaled, "
                  "rotated or mirrored in Cura.")
    return "\n".join(lines)


def apply_to_lines(lines, dx: float, dy: float):
    """Shift every X/Y on a motion command. Deliberately leaves G28
    (homing, machine coordinates) and G92 (coordinate reset) alone --
    shifting those changes what the coordinates MEAN rather than where
    the part sits."""
    out = []
    for raw in lines:
        m = _MOVE_RE.match(raw.split(";", 1)[0])
        if not m:
            out.append(raw)
            continue
        code_part, rest = raw, ""
        if ";" in raw:
            code_part, rest = raw.split(";", 1)
            rest = ";" + rest

        def shift(match):
            axis = match.group(1).upper()
            val = float(match.group(2)) + (dx if axis == "X" else dy)
            return f"{axis}{val:.3f}"

        out.append(_XY_RE.sub(shift, code_part) + rest)
    return out


def apply_to_file(src: Path, dst: Path, dx: float, dy: float) -> None:
    text = Path(src).read_text(encoding="utf-8", errors="replace").splitlines()
    Path(dst).write_text("\n".join(apply_to_lines(text, dx, dy)) + "\n")


def check_within_bed(tp: gcode.Toolpath,
                     x_min: float, x_max: float, y_min: float, y_max: float):
    """Does anything this toolpath does leave the bed?

    Takes an ALREADY-SHIFTED toolpath (parse the written file) rather
    than the original plus an offset. Adding the offset to the original's
    coordinates counts the post-G28 origin at (0,0) as if it moved too,
    which it does not -- homing is left alone by apply_to_lines -- and
    that produced a confident, wrong "goes off the front of the bed"
    warning on files that were perfectly fine.

    Checks ALL moves, not just extruding ones: a prime line or a wipe
    that ends up off the edge still crashes."""
    problems = []
    xs = [v for m in tp.moves for v in (m.x0, m.x1)]
    ys = [v for m in tp.moves for v in (m.y0, m.y1)]
    if xs and (min(xs) < x_min - 0.5 or max(xs) > x_max + 0.5):
        problems.append(f"X reaches [{min(xs):.1f},{max(xs):.1f}], "
                         f"bed is [{x_min:g},{x_max:g}]")
    if ys and (min(ys) < y_min - 0.5 or max(ys) > y_max + 0.5):
        problems.append(f"Y reaches [{min(ys):.1f},{max(ys):.1f}], "
                         f"bed is [{y_min:g},{y_max:g}]")
    return problems
