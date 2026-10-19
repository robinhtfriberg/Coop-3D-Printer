from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from geometry import (BED_X_MIN, BED_X_MAX, BED_Y_MIN, BED_Y_MAX,
                       Z_MAX, TOOLHEAD2)

EPS = 1e-9


# ---------------------------------------------------------------------
# STL I/O
# ---------------------------------------------------------------------

def read_stl(path: Path) -> np.ndarray:
    """Returns (N,3,3) triangle vertex array. Handles binary and ASCII."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + count * 50:
            tris = np.empty((count, 3, 3), dtype=np.float64)
            off = 84
            for i in range(count):
                off += 12
                for v in range(3):
                    tris[i, v] = struct.unpack("<3f", data[off:off + 12])
                    off += 12
                off += 2
            return tris
    verts = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts or len(verts) % 3:
        raise ValueError(f"{path} is neither a valid binary nor ASCII STL")
    return np.array(verts, dtype=np.float64).reshape(-1, 3, 3)


def write_stl(path: Path, tris: np.ndarray, name: str = "chunk") -> None:
    out = bytearray()
    out += name.encode("ascii", "replace")[:80].ljust(80, b"\0")
    out += struct.pack("<I", len(tris))
    for tri in tris:
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        ln = np.linalg.norm(n)
        n = n / ln if ln > EPS else np.zeros(3)
        out += struct.pack("<3f", *n)
        for v in tri:
            out += struct.pack("<3f", *v)
        out += struct.pack("<H", 0)
    Path(path).write_bytes(bytes(out))


# ---------------------------------------------------------------------
# Plane clipping
# ---------------------------------------------------------------------

def sloped_plane(offset: float, z_min: float, slope_deg: float):
    """Plane whose seam at height z sits at y = offset + (z-z_min)*tan(slope).

    Returned normal points toward the DISCARD side for keep_lower, i.e.
    signed_distance(p) <= 0 means "keep" when keeping the lower chunk.
    slope_deg=0 is a plain vertical cut -- a generalisation, not a
    different mechanism (same construction partition.py uses).
    """
    slope = np.tan(np.radians(slope_deg))
    normal = np.array([0.0, 1.0, -slope])
    normal = normal / np.linalg.norm(normal)
    origin = np.array([0.0, offset, z_min])
    return normal, origin


def _signed(pts: np.ndarray, normal: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return (pts - origin) @ normal


def clip_mesh(tris: np.ndarray, normal: np.ndarray, origin: np.ndarray,
              keep_negative: bool) -> tuple[np.ndarray, list]:
    """Clip a triangle soup by a plane. Returns (kept triangles, cut edges).

    Cut edges are the segments lying exactly in the plane, wound so the
    cap built from them closes the mesh with outward-facing normals.
    """
    n = normal if keep_negative else -normal
    kept: list[np.ndarray] = []
    cut_edges: list[tuple[np.ndarray, np.ndarray]] = []

    for tri in tris:
        d = _signed(tri, n, origin)
        inside = d <= EPS
        n_in = int(inside.sum())

        if n_in == 3:
            kept.append(tri)
            continue
        if n_in == 0:
            continue

        # Walk the triangle's edges, building the clipped polygon and
        # recording where we crossed the plane.
        poly: list[np.ndarray] = []
        crossings: list[np.ndarray] = []
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            da, db = d[i], d[(i + 1) % 3]
            if da <= EPS:
                poly.append(a)
            if (da > EPS) != (db > EPS):
                t = da / (da - db)
                p = a + t * (b - a)
                poly.append(p)
                crossings.append(p)

        if len(poly) >= 3:
            for i in range(1, len(poly) - 1):
                kept.append(np.array([poly[0], poly[i], poly[i + 1]]))
        if len(crossings) == 2:
            cut_edges.append((crossings[0], crossings[1]))

    return (np.array(kept) if kept else np.zeros((0, 3, 3))), cut_edges


# ---------------------------------------------------------------------
# Capping the cut face
# ---------------------------------------------------------------------

def _plane_basis(normal: np.ndarray):
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(normal @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _chain_loops(edges: list, tol: float = 1e-4) -> list[list[np.ndarray]]:
    """Chain plane-lying segments into closed loops."""
    def key(p):
        return (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))

    adj: dict = {}
    for a, b in edges:
        adj.setdefault(key(a), []).append((a, b))
        adj.setdefault(key(b), []).append((b, a))

    used: set = set()
    loops = []
    for idx, (a0, b0) in enumerate(edges):
        if idx in used:
            continue
        edge_ids = {}
        for i, (a, b) in enumerate(edges):
            edge_ids.setdefault(frozenset([key(a), key(b)]), []).append(i)

        loop = [a0]
        used.add(idx)
        cur, prev_key = b0, key(a0)
        guard = 0
        while key(cur) != key(loop[0]) and guard < len(edges) + 5:
            guard += 1
            loop.append(cur)
            nxt = None
            for (a, b) in adj.get(key(cur), []):
                if key(b) == prev_key:
                    continue
                eid_list = edge_ids.get(frozenset([key(a), key(b)]), [])
                fresh = [e for e in eid_list if e not in used]
                if fresh:
                    used.add(fresh[0])
                    nxt = b
                    break
            if nxt is None:
                break
            prev_key = key(cur)
            cur = nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _ear_clip(poly2d: list[np.ndarray]) -> list[tuple[int, int, int]]:
    """Triangulate a simple (possibly non-convex) polygon by ear clipping."""
    n = len(poly2d)
    if n < 3:
        return []
    idx = list(range(n))

    def area(p):
        s = 0.0
        for i in range(len(p)):
            x1, y1 = p[i]
            x2, y2 = p[(i + 1) % len(p)]
            s += x1 * y2 - x2 * y1
        return s / 2.0

    if area(poly2d) < 0:
        idx.reverse()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def point_in_tri(p, a, b, c):
        d1 = cross(a, b, p); d2 = cross(b, c, p); d3 = cross(c, a, p)
        neg = (d1 < -EPS) or (d2 < -EPS) or (d3 < -EPS)
        pos = (d1 > EPS) or (d2 > EPS) or (d3 > EPS)
        return not (neg and pos)

    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        clipped = False
        for i in range(len(idx)):
            ia, ib, ic = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
            a, b, c = poly2d[ia], poly2d[ib], poly2d[ic]
            if cross(a, b, c) <= EPS:
                continue  # reflex
            if any(point_in_tri(poly2d[j], a, b, c)
                    for j in idx if j not in (ia, ib, ic)):
                continue
            tris.append((ia, ib, ic))
            idx.pop(i)
            clipped = True
            break
        if not clipped:
            break  # degenerate; fall through to the fan below
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def cap_cut_face(cut_edges: list, normal: np.ndarray, keep_negative: bool) -> np.ndarray:
    """Build the flat face that closes the mesh where the plane cut it."""
    if not cut_edges:
        return np.zeros((0, 3, 3))
    n = normal if keep_negative else -normal
    u, v = _plane_basis(n)
    caps = []
    for loop in _chain_loops(cut_edges):
        poly2d = [np.array([p @ u, p @ v]) for p in loop]
        for (i, j, k) in _ear_clip(poly2d):
            tri = np.array([loop[i], loop[j], loop[k]])
            # Face outward: the cap's normal should point along +n (the
            # discard direction), since that side is now open air.
            fn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if fn @ n < 0:
                tri = tri[::-1]
            caps.append(tri)
    return np.array(caps) if caps else np.zeros((0, 3, 3))


# ---------------------------------------------------------------------
# Volume balancing
# ---------------------------------------------------------------------

def mesh_volume(tris: np.ndarray) -> float:
    if len(tris) == 0:
        return 0.0
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def split_at(tris, offset, z_min, slope_deg):
    normal, origin = sloped_plane(offset, z_min, slope_deg)
    lower_body, lower_cuts = clip_mesh(tris, normal, origin, keep_negative=True)
    upper_body, upper_cuts = clip_mesh(tris, normal, origin, keep_negative=False)
    lower = np.concatenate([lower_body, cap_cut_face(lower_cuts, normal, True)]) \
        if len(lower_body) else lower_body
    upper = np.concatenate([upper_body, cap_cut_face(upper_cuts, normal, False)]) \
        if len(upper_body) else upper_body
    return lower, upper


def find_balanced_offset(tris, z_min, slope_deg, tolerance=0.01, max_iters=30):
    """Binary-search the seam offset so the two chunks have equal volume."""
    total = mesh_volume(tris)
    target = total / 2.0
    lo, hi = float(tris[:, :, 1].min()), float(tris[:, :, 1].max())
    best = (lo + hi) / 2
    for i in range(max_iters):
        mid = (lo + hi) / 2.0
        lower, _ = split_at(tris, mid, z_min, slope_deg)
        kept = mesh_volume(lower)
        err = abs(kept - target) / max(total, EPS)
        print(f"  iter {i:2d}: seam Y0={mid:8.3f}  volume error={err:.4f}")
        best = mid
        if err < tolerance:
            return mid
        lo, hi = (mid, hi) if kept < target else (lo, mid)
    print(f"  (stopped at {max_iters} iterations; using Y0={best:.3f})")
    return best


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def check_bed_fit(tris: np.ndarray) -> None:
    lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    problems = []
    if lo[0] < BED_X_MIN or hi[0] > BED_X_MAX:
        problems.append(f"X [{lo[0]:.1f},{hi[0]:.1f}] outside bed [{BED_X_MIN},{BED_X_MAX}]")
    if lo[1] < BED_Y_MIN or hi[1] > BED_Y_MAX:
        problems.append(f"Y [{lo[1]:.1f},{hi[1]:.1f}] outside bed [{BED_Y_MIN},{BED_Y_MAX}]")
    if hi[2] > Z_MAX:
        problems.append(f"Z max {hi[2]:.1f} above {Z_MAX}")
    if problems:
        raise ValueError("part does not fit the shared bed defined in geometry.py:\n  "
                          + "\n  ".join(problems)
                          + "\n(use --center to place it on the bed automatically)")


def check_reach(lower: np.ndarray, upper_world: np.ndarray) -> list[str]:
    """Each chunk must lie inside its OWN toolhead's reach.

    Splitting by volume alone can put material where the head assigned to
    it physically cannot go -- the plan would then be collision-free and
    still unprintable. Where the goal is build volume rather than speed,
    this is the check that the whole approach rests on."""
    import geometry as g
    problems = []
    for label, chunk, head in (("toolhead1", lower, "toolhead1"),
                                ("toolhead2", upper_world, "toolhead2")):
        lo = float(chunk[:, :, 1].min())
        hi = float(chunk[:, :, 1].max())
        r0, r1 = g.reach_of(head)
        if lo < r0 - 1e-6 or hi > r1 + 1e-6:
            problems.append(
                f"{label}'s chunk spans world Y [{lo:.1f},{hi:.1f}] but that "
                f"toolhead only reaches [{r0:g},{r1:g}]. Move the seam "
                f"(--seam-y) so each chunk falls inside its own head's "
                f"reach, or correct REACH_Y in geometry.py.")
    return problems


def report_build_volume(tris: np.ndarray) -> None:
    """State plainly whether two heads were needed for this part at all."""
    import geometry as g
    lo = float(tris[:, :, 1].min())
    hi = float(tris[:, :, 1].max())
    single = g.best_single_reach()
    gain = g.build_volume_gain()
    span = hi - lo
    print(f"\nBuild volume:")
    print(f"  part spans world Y [{lo:.1f}, {hi:.1f}]  ({span:.1f}mm)")
    print(f"  one toolhead alone reaches [{single[0]:g}, {single[1]:g}]  "
           f"({single[1] - single[0]:g}mm)")
    if gain <= 0:
        print(f"  !! the two toolheads cover no more of the bed than one does.")
        print(f"     With REACH_Y as set in geometry.py, a single head could "
               f"print this whole part,")
        print(f"     so the pair buys reliability or speed -- not build volume. "
               f"If the real arms")
        print(f"     cannot each cross the whole bed, set REACH_Y to the "
               f"measured travel.")
    elif span > (single[1] - single[0]) + 1e-6:
        print(f"  -> this part is {span - (single[1] - single[0]):.1f}mm longer "
               f"than one toolhead can reach.")
        print(f"     It needs both heads; that is the build-volume gain.")
    else:
        print(f"  the pair reaches {gain:.1f}mm more than one head, but THIS "
               f"part still fits within")
        print(f"     a single head's reach -- so it did not need splitting.")


def check_touches_plate(tris: np.ndarray, z_min: float, label: str, tol=1e-3) -> None:
    if tris[:, :, 2].min() > z_min + tol:
        print(f"  WARNING: {label} does not touch the build plate "
               f"(its lowest point is z={tris[:, :, 2].min():.3f}); it would "
               f"print unsupported. Check the seam angle.")


def _edge_manifold_report(tris: np.ndarray) -> tuple[int, int]:
    """Count boundary edges (used once). 0 means watertight."""
    counts: dict = {}
    for tri in tris:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            ka = tuple(np.round(a, 4)); kb = tuple(np.round(b, 4))
            counts[frozenset([ka, kb])] = counts.get(frozenset([ka, kb]), 0) + 1
    boundary = sum(1 for v in counts.values() if v == 1)
    return boundary, len(counts)


# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_stl", type=Path)
    ap.add_argument("--slope-angle", type=float, default=12.0,
                     help="degrees from vertical; 0 = flat cut (default: 12)")
    ap.add_argument("--seam-y", type=float, default=None,
                     help="seam Y at the build plate; default = volume-balanced search")
    ap.add_argument("--center", action="store_true",
                     help="translate the part to the centre of the shared bed first")
    ap.add_argument("--outdir", type=Path, default=Path("chunks"))
    args = ap.parse_args()

    if not args.input_stl.exists():
        print(f"ERROR: {args.input_stl} does not exist", file=sys.stderr)
        sys.exit(1)

    tris = read_stl(args.input_stl)
    print(f"Loaded {args.input_stl} -- {len(tris)} triangles")

    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    if args.center:
        shift = np.array([(BED_X_MIN + BED_X_MAX) / 2 - (lo[0] + hi[0]) / 2,
                           (BED_Y_MIN + BED_Y_MAX) / 2 - (lo[1] + hi[1]) / 2,
                           -lo[2]])
        tris = tris + shift
        print(f"Centred on the bed: shifted by ({shift[0]:.1f}, {shift[1]:.1f}, {shift[2]:.1f})")
    check_bed_fit(tris)

    z_min = float(tris[:, :, 2].min())
    import geometry as _g
    if args.seam_y is None:
        print(f"Searching for a volume-balanced seam (slope={args.slope_angle} deg)...")
        seam_y = find_balanced_offset(tris, z_min, args.slope_angle)
        r1, r2 = _g.reach_of("toolhead1"), _g.reach_of("toolhead2")
        lo_ok, hi_ok = max(r1[0], r2[0]), min(r1[1], r2[1])
        if hi_ok < lo_ok:
            print(f"ERROR: the two toolheads' reaches do not overlap "
                  f"({r1} and {r2}), so no seam can be reachable by both.",
                  file=sys.stderr)
            sys.exit(1)
        clamped = min(max(seam_y, lo_ok), hi_ok)
        if abs(clamped - seam_y) > 1e-6:
            print(f"  volume-balanced seam Y={seam_y:.1f} is outside the band "
                  f"both heads reach [{lo_ok:g},{hi_ok:g}]; using Y={clamped:.1f}")
            seam_y = clamped
    else:
        seam_y = args.seam_y

    lower, upper = split_at(tris, seam_y, z_min, args.slope_angle)
    if not len(lower) or not len(upper):
        raise ValueError("the seam produced an empty chunk -- check --seam-y / --slope-angle")

    v_lo, v_hi = mesh_volume(lower), mesh_volume(upper)
    print(f"Seam at Y0={seam_y:.2f}, slope={args.slope_angle} deg")
    print(f"  toolhead1 (supporting/lower): {len(lower):5d} tris, volume {v_lo:10.1f} mm^3")
    print(f"  toolhead2 (dependent/upper):  {len(upper):5d} tris, volume {v_hi:10.1f} mm^3")
    print(f"  balance: {100 * v_lo / max(v_lo + v_hi, EPS):.1f}% / "
           f"{100 * v_hi / max(v_lo + v_hi, EPS):.1f}%")

    check_touches_plate(lower, z_min, "toolhead1 chunk")
    check_touches_plate(upper, z_min, "toolhead2 chunk")

    reach_problems = check_reach(lower, upper)
    for p in reach_problems:
        print(f"  !! {p}")
    report_build_volume(tris)
    for label, chunk in (("toolhead1", lower), ("toolhead2", upper)):
        boundary, total = _edge_manifold_report(chunk)
        if boundary:
            print(f"  WARNING: {label} chunk has {boundary} open edges of {total} "
                   f"-- Cura may need 'fix model' / union enabled for it.")
        else:
            print(f"  {label} chunk is watertight ({total} edges, all shared).")

    # toolhead2's chunk is written in ITS OWN local frame, because the
    # hardware is mounted mirrored -- see this module's docstring.
    upper_local = upper.copy()
    for i in range(len(upper_local)):
        for v in range(3):
            x, y, z = upper_local[i, v]
            upper_local[i, v] = TOOLHEAD2.to_local(x, y, z)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out1 = args.outdir / "chunk_toolhead1.stl"
    out2 = args.outdir / "chunk_toolhead2.stl"
    write_stl(out1, lower, "COOP_CHUNK_TOOLHEAD1")
    write_stl(out2, upper_local, "COOP_CHUNK_TOOLHEAD2_LOCAL")

    placement = {
        "generated_by": "step1_split_stl.py",
        "source": str(args.input_stl),
        "seam_y": round(seam_y, 4),
        "slope_deg": args.slope_angle,
        "note": ("bbox is in the frame each STL is WRITTEN in, which is the "
                  "frame Cura will slice and therefore the frame its G-code "
                  "will be in. toolhead2's chunk is already mirrored into "
                  "its own local frame."),
        "chunks": {},
    }
    for head, chunk, frame, fname in (
            ("toolhead1", lower, "world", out1.name),
            ("toolhead2", upper_local, "toolhead2-local", out2.name)):
        v = chunk.reshape(-1, 3)
        placement["chunks"][head] = {
            "file": fname,
            "frame": frame,
            "bbox_min": [round(float(x), 4) for x in v.min(axis=0)],
            "bbox_max": [round(float(x), 4) for x in v.max(axis=0)],
        }
    place_path = args.outdir / "placement.json"
    place_path.write_text(json.dumps(placement, indent=1))

    print(f"\nWrote:\n  {out1}  (slice for toolhead 1)\n  {out2}  (slice for toolhead 2 "
           f"-- already mirrored into toolhead2's frame)\n  {place_path}  "
           f"(intended positions, for step 3)")
    print("\nNext: slice BOTH in Cura, however it wants to place them, then run\n"
           f"  python3 step3_deconflict.py t1.gcode t2.gcode --placement {place_path}\n"
           "which puts them back where they belong. See PIPELINE.md step 2.")


if __name__ == "__main__":
    main()
