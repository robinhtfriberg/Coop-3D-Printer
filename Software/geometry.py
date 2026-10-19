from dataclasses import dataclass

# --- bed and travel limits (world frame) --------------------------------
BED_X_MIN, BED_X_MAX = 0.0, 280.0
BED_Y_MIN, BED_Y_MAX = 0.0, 280.0
Z_MIN, Z_MAX = 0.0, 250.0

PARK_CLEARANCE = 95.0                       # travel beyond the bed edge
PARK_X = BED_X_MAX                          # carriage fully extended
PARK_Y_TOOLHEAD1 = BED_Y_MAX + PARK_CLEARANCE   # 315, beyond the +Y edge
PARK_Y_TOOLHEAD2 = BED_Y_MIN - PARK_CLEARANCE   # -35, beyond the -Y edge
MOUNT_WIDTH_X = 70.0
MOUNT_DEPTH_Y = 80.0
MOUNT_HEIGHT_Z = 20.0
BEAM_THICKNESS_Y = 20.0   # the beam is a bridge spanning the full bed in X

DEFAULT_CLEARANCE = 5.0  # extra safety margin added around every envelope


# --- per-toolhead reach -----------------------------------------------
REACH_Y = {
    "toolhead1": (BED_Y_MIN, BED_Y_MAX),
    "toolhead2": (BED_Y_MIN, BED_Y_MAX),
}


def reach_of(head: str) -> tuple:
    return REACH_Y[head]


def combined_reach() -> tuple:
    lo = min(r[0] for r in REACH_Y.values())
    hi = max(r[1] for r in REACH_Y.values())
    return (lo, hi)


def best_single_reach() -> tuple:
    """The largest span one toolhead could cover on its own."""
    return max(REACH_Y.values(), key=lambda r: r[1] - r[0])


def reaches_overlap() -> float:
    """How much of the bed both heads can reach. The seam has to sit in
    this band -- outside it, one chunk contains material its own toolhead
    physically cannot get to."""
    r1, r2 = REACH_Y["toolhead1"], REACH_Y["toolhead2"]
    return min(r1[1], r2[1]) - max(r1[0], r2[0])


def build_volume_gain() -> float:
    """Extra Y span the pair covers versus the better single head, mm."""
    lo, hi = combined_reach()
    s = best_single_reach()
    return (hi - lo) - (s[1] - s[0])


def parked_clearance_guarantee() -> float:
    """The smallest gap the machine can ever have while behaving
    correctly: one toolhead parked, the other printing at the very edge
    of the bed nearest it."""
    return PARK_CLEARANCE - MOUNT_DEPTH_Y / 2 - MOUNT_DEPTH_Y / 2


def _check_park_clears_bed(safety: float = DEFAULT_CLEARANCE) -> None:
    """A parked toolhead's mount must clear not just the bed edge, but
    whichever of the other, still-active toolhead's envelopes reaches
    furthest past the edge when that toolhead is printing right at the
    boundary -- and since sim.collision() checks mount-mount as well as
    beam-beam/beam-mount, that's not necessarily the beam: a mount deeper
    than the beam is thick (as ours now is, 80mm vs 20mm) reaches further
    than the beam does, so it's the mount that sets the real margin. This
    is what makes the fail-safe park-zone argument in Sec. 6.5 actually
    true -- if it doesn't hold, "parked" isn't safe, regardless of what
    the coordination logic does."""
    active_reach = max(MOUNT_DEPTH_Y, BEAM_THICKNESS_Y) / 2 + safety
    required = (MOUNT_DEPTH_Y / 2 + safety) + active_reach
    assert PARK_CLEARANCE >= required, (
        f"PARK_CLEARANCE ({PARK_CLEARANCE}mm) does not clear a parked "
        f"toolhead's mount ({MOUNT_DEPTH_Y}mm deep) against the other, "
        f"active toolhead's furthest-reaching envelope (mount "
        f"{MOUNT_DEPTH_Y}mm deep vs. beam {BEAM_THICKNESS_Y}mm thick) "
        f"printing at the bed edge, both plus a {safety}mm safety margin "
        f"-- needs >= {required}mm. Either shrink the footprints to what "
        f"the hardware actually measures, or increase the mechanical "
        f"park travel."
    )


_check_park_clears_bed()


@dataclass(frozen=True)
class Toolhead:
    name: str
    mirrored: bool  # True for the toolhead facing the opposite way

    def to_world(self, x: float, y: float, z: float) -> tuple:
        wy = (BED_Y_MAX - y) if self.mirrored else y
        return x, wy, z

    def to_local(self, wx: float, wy: float, wz: float) -> tuple:
        # The Y-mirror is its own inverse, so to_local and to_world share
        # one formula -- there is no separate, divergent "reverse" path
        # to get out of sync with the forward one.
        return self.to_world(wx, wy, wz)

    @property
    def park_pose_local(self) -> tuple:
        """(x, y) to command LOCALLY to reach this toolhead's own park zone.
        Both toolheads command the same local pose (280, 315) -- the mirror
        in to_world() is what makes that land on opposite bed edges. Z is
        deliberately omitted: parking applies no Z offset."""
        return (PARK_X, BED_Y_MAX + PARK_CLEARANCE)


TOOLHEAD1 = Toolhead("toolhead1", mirrored=False)
TOOLHEAD2 = Toolhead("toolhead2", mirrored=True)
TOOLHEADS = {"toolhead1": TOOLHEAD1, "toolhead2": TOOLHEAD2}


def in_bed_bounds(x: float, y: float, z: float) -> bool:
    return (BED_X_MIN <= x <= BED_X_MAX and BED_Y_MIN <= y <= BED_Y_MAX
            and Z_MIN <= z <= Z_MAX)


if __name__ == "__main__":
    # Quick self-check: both park poses should round-trip and land on
    # opposite sides of the bed with the same clearance travel.
    for th in (TOOLHEAD1, TOOLHEAD2):
        x, y = th.park_pose_local
        wx, wy, _ = th.to_world(x, y, 0.0)
        lx, ly, _ = th.to_local(wx, wy, 0.0)
        assert (lx, ly) == (x, y), f"{th.name}: transform is not self-inverse"
        print(f"{th.name}: local park=({x},{y})  world=({wx},{wy})")
    assert TOOLHEAD1.to_world(*TOOLHEAD1.park_pose_local, 0.0)[1] == PARK_Y_TOOLHEAD1
    assert TOOLHEAD2.to_world(*TOOLHEAD2.park_pose_local, 0.0)[1] == PARK_Y_TOOLHEAD2
    print("geometry self-check OK")
