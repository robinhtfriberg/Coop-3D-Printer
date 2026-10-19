"""Shared motion + collision model.
"""
from __future__ import annotations

from pathlib import Path

import gcode as _gc

from geometry import (
    BED_X_MIN, BED_X_MAX, MOUNT_WIDTH_X, MOUNT_DEPTH_Y, MOUNT_HEIGHT_Z,
    BEAM_THICKNESS_Y, Toolhead,
)

Move = _gc.Move
Toolpath = _gc.Toolpath
position_at = _gc.position_at
parse_toolpath = _gc.parse


def envelope(toolhead: Toolhead, local_pos: tuple, clearance: float = 0.0):
    """(mount_box, beam_box) in WORLD coordinates for a local position.
    Boxes are (xmin, xmax, ymin, ymax, zmin, zmax)."""
    x, y, z = toolhead.to_world(*local_pos)
    mount = (x - MOUNT_WIDTH_X / 2 - clearance, x + MOUNT_WIDTH_X / 2 + clearance,
             y - MOUNT_DEPTH_Y / 2 - clearance, y + MOUNT_DEPTH_Y / 2 + clearance,
             z, z + MOUNT_HEIGHT_Z)
    beam = (BED_X_MIN, BED_X_MAX,   # the beam bridges the full bed in X
            y - BEAM_THICKNESS_Y / 2 - clearance, y + BEAM_THICKNESS_Y / 2 + clearance,
            z, z + MOUNT_HEIGHT_Z)
    return mount, beam


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    return all(max(a[2 * i], b[2 * i]) < min(a[2 * i + 1], b[2 * i + 1]) for i in range(3))


def collision(th1: Toolhead, pos1: tuple, th2: Toolhead, pos2: tuple,
              clearance: float = 0.0) -> tuple[bool, str]:
    m1, b1 = envelope(th1, pos1, clearance)
    m2, b2 = envelope(th2, pos2, clearance)
    for reason, a, b in (("mount-mount", m1, m2), ("beam-beam", b1, b2),
                          ("beam1-mount2", b1, m2), ("beam2-mount1", b2, m1)):
        if _boxes_overlap(a, b):
            return True, reason
    return False, ""
