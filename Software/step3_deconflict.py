from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import annotate
import gcode
import place
import geometry
import planner
import replay
import sim


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600 \
        else f"{s // 60:d}m{s % 60:02d}s"


def _check_placement(tp1: gcode.Toolpath, tp2: gcode.Toolpath) -> list[str]:

    notes = []
    lo1, hi1 = tp1.model_bbox
    lo2, hi2 = tp2.model_bbox

    for name, lo, hi in (("toolhead1", lo1, hi1), ("toolhead2", lo2, hi2)):
        if (lo[0] < geometry.BED_X_MIN - 1 or hi[0] > geometry.BED_X_MAX + 1
                or lo[1] < geometry.BED_Y_MIN - 1 or hi[1] > geometry.BED_Y_MAX + 1):
            notes.append(
                f"!! {name}'s extrusion extents X[{lo[0]:.1f},{hi[0]:.1f}] "
                f"Y[{lo[1]:.1f},{hi[1]:.1f}] fall outside the shared bed "
                f"X[{geometry.BED_X_MIN:g},{geometry.BED_X_MAX:g}] "
                f"Y[{geometry.BED_Y_MIN:g},{geometry.BED_Y_MAX:g}]. Cura is "
                f"probably re-centring the model -- see PIPELINE.md step 2.")

    w1_lo = geometry.TOOLHEAD1.to_world(*lo1)[1]
    w1_hi = geometry.TOOLHEAD1.to_world(*hi1)[1]
    w2_a = geometry.TOOLHEAD2.to_world(*lo2)[1]
    w2_b = geometry.TOOLHEAD2.to_world(*hi2)[1]
    w1 = (min(w1_lo, w1_hi), max(w1_lo, w1_hi))
    w2 = (min(w2_a, w2_b), max(w2_a, w2_b))
    overlap = min(w1[1], w2[1]) - max(w1[0], w2[0])
    gap = max(w1[0], w2[0]) - min(w1[1], w2[1])
    notes.append(f"   toolhead1 occupies world Y [{w1[0]:.1f}, {w1[1]:.1f}]")
    notes.append(f"   toolhead2 occupies world Y [{w2[0]:.1f}, {w2[1]:.1f}]  "
                 f"(after the mirror)")
    if overlap > 0.6 * min(w1[1] - w1[0], w2[1] - w2[0]):
        notes.append(
            "!! the two chunks cover almost the same region of the bed. If "
            "you sliced both halves without preserving position, or fed the "
            "same file twice, the plan below will be meaningless.")
    elif gap > 5:
        notes.append(f"!! the two chunks are {gap:.1f}mm apart in Y and never "
                      f"meet -- they will not bond into one part.")
    
    def _world_span(tp, th, box):
        lo, hi = box
        a, b = th.to_world(*lo)[1], th.to_world(*hi)[1]
        return (min(a, b), max(a, b))

    f1 = _world_span(tp1, geometry.TOOLHEAD1, tp1.bbox)
    f2 = _world_span(tp2, geometry.TOOLHEAD2, tp2.bbox)
    adhesion_overlap = min(f1[1], f2[1]) - max(f1[0], f2[0])
    has_adhesion = any(gcode.is_non_model_feature(f)
                       for tp in (tp1, tp2) for f in tp.feature_extents())
    if has_adhesion and adhesion_overlap > max(overlap, 0.0) + 2.0:
        notes.append(
            f"!! the parts are placed correctly, but their brim/skirt rings "
            f"overlap by {adhesion_overlap:.1f}mm across the seam "
            f"(toolhead1 reaches world Y {f1[1]:.1f}, toolhead2 down to "
            f"{f2[0]:.1f}). Both toolheads would put material in that band, "
            f"and the second one there prints on top of the first. Turn "
            f"adhesion off for these chunks, or use a skirt instead of a "
            f"brim -- a skirt does not touch the part.")
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gcode1", type=Path, help="toolhead1 (supporting / lower chunk)")
    ap.add_argument("gcode2", type=Path, help="toolhead2 (dependent / upper chunk)")
    ap.add_argument("--outdir", type=Path, default=Path("ready"))
    ap.add_argument("--placement", type=Path, default=None,
                     help="chunks/placement.json from step 1. Undoes Cura's "
                          "re-centring and puts each chunk back where the "
                          "split intended. Strongly recommended.")
    ap.add_argument("--no-placement-check", action="store_true",
                     help="skip the sanity check on where the two chunks sit")
    ap.add_argument("--clearance", type=float, default=geometry.DEFAULT_CLEARANCE,
                     help=f"extra safety margin around each envelope, mm "
                          f"(default {geometry.DEFAULT_CLEARANCE:g})")
    ap.add_argument("--max-iterations", type=int, default=30,
                     help="planner fixpoint iterations (default 30)")
    args = ap.parse_args()

    for p in (args.gcode1, args.gcode2):
        if not p.exists():
            print(f"ERROR: {p} does not exist", file=sys.stderr)
            sys.exit(1)

    print("=" * 68)
    print("STEP 3  de-conflict")
    print("=" * 68)

    args.outdir.mkdir(parents=True, exist_ok=True)
    src1, src2 = args.gcode1, args.gcode2

    tp1 = gcode.parse(src1)
    tp2 = gcode.parse(src2)
    print(f"  {gcode.describe(tp1)}")
    print(f"  {gcode.describe(tp2)}")
    for tp in (tp1, tp2):
        for w in tp.warnings:
            print(f"  NOTE ({Path(tp.source).name}): {w}")
        note = place.adhesion_note(tp)
        if note:
            print(f"  NOTE ({Path(tp.source).name}): {note}")

    # --- restore the placement  ------------------------
    if args.placement:
        print("\nRestoring placement from", args.placement)
        try:
            pdata = place.load_placement(args.placement)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        fixed = {}
        for head, src, tp in (("toolhead1", src1, tp1), ("toolhead2", src2, tp2)):
            spec = pdata["chunks"].get(head)
            if spec is None:
                print(f"ERROR: {args.placement} has no entry for {head}",
                      file=sys.stderr)
                sys.exit(1)
            try:
                pl = place.compute(tp, spec["bbox_min"], spec["bbox_max"])
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
            print("  " + pl.describe(head))
            if not pl.moved:
                print(f"    already in position, left alone")
                fixed[head] = src
                continue
            out = args.outdir / f"{Path(src).stem}_placed.gcode"
            place.apply_to_file(src, out, pl.dx, pl.dy)
            fixed[head] = out
            
            problems = place.check_within_bed(
                gcode.parse(out), geometry.BED_X_MIN, geometry.BED_X_MAX,
                geometry.BED_Y_MIN, geometry.BED_Y_MAX)
            for p in problems:
                print(f"    !! after the shift, {p}")
            print(f"    wrote {out}")

        src1, src2 = fixed["toolhead1"], fixed["toolhead2"]
        tp1, tp2 = gcode.parse(src1), gcode.parse(src2)

    if not args.no_placement_check:
        print("\nPlacement check (world frame):")
        for note in _check_placement(tp1, tp2):
            print(note)
        if any(n.startswith("!!") for n in _check_placement(tp1, tp2)) \
                and not args.placement:
            print("\n  Most likely cause: Cura re-centred each chunk on load, so")
            print("  both landed in the middle of the bed. Re-run with the")
            print("  placement file step 1 wrote:")
            print(f"    python3 {Path(sys.argv[0]).name} {args.gcode1} "
                   f"{args.gcode2} --placement chunks/placement.json")

    # --- baseline: what would happen with no coordination at all -----
    print("\nBaseline (running both files as-is, no coordination):")
    print(f"  simulating {len(tp1.moves) + len(tp2.moves):,} moves over "
          f"{max(tp1.total_time, tp2.total_time) / 60:.0f} modelled minutes...",
          flush=True)
    baseline = replay.replay(tp1, tp2, {"protocol": "coop-plan-v2", "events": []},
                              clearance=args.clearance, dt=0.15)
    n_base = len(baseline["collisions"])
    if n_base:
        first = baseline["collisions"][0]
        print(f"  {n_base} colliding samples; first at t={first['t']:.1f}s ({first['reason']})")
    else:
        print("  no collisions found -- these two files are already safe to run "
               "together as-is. The annotated copies below will be equivalent.")

    # --- plan --------------------------------------------------------
    print("\nPlanning...")
    t_plan = time.time()
    try:
        manifest = planner.plan(src1, src2, max_iterations=args.max_iterations,
                                 on_progress=lambda m: print(f"  {m}", flush=True))
    except RuntimeError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        sys.exit(2)

    events = manifest["events"]
    kinds = {}
    for e in events:
        kinds[(e["head"], e["kind"])] = kinds.get((e["head"], e["kind"]), 0) + 1

    # --- verify ------------------------------------------------------
    print(f"  planned in {time.time() - t_plan:.0f}s", flush=True)
    verified = replay.replay(tp1, tp2, manifest, clearance=args.clearance, dt=0.15,
                              record_frames=True)
    if verified["collisions"]:
        print(f"\nFAILED: {len(verified['collisions'])} collisions survive the plan. "
               f"Refusing to write G-code.", file=sys.stderr)
        sys.exit(2)

    # --- write -------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    out1 = args.outdir / f"{args.gcode1.stem}_coop.gcode"
    out2 = args.outdir / f"{args.gcode2.stem}_coop.gcode"
    out1.write_text("\n".join(annotate.annotate(src1, tp1, events, "toolhead1")) + "\n")
    out2.write_text("\n".join(annotate.annotate(src2, tp2, events, "toolhead2")) + "\n")
    manifest["sources"] = {
        "toolhead1": dict(place.fingerprint(tp1), file=str(src1)),
        "toolhead2": dict(place.fingerprint(tp2), file=str(src2)),
    }
    (args.outdir / "plan.json").write_text(json.dumps(manifest, indent=1))

    frames = verified.get("frames", [])
    both = sum(1 for f in frames
               if f["toolhead1"]["state"] in ("printing", "detour")
               and f["toolhead2"]["state"] in ("printing", "detour"))
    either = sum(1 for f in frames
                 if f["toolhead1"]["state"] in ("printing", "detour")
                 or f["toolhead2"]["state"] in ("printing", "detour"))
    concurrency = (100.0 * both / either) if either else 0.0
    serial_time = tp1.total_time + tp2.total_time
    saving = 100.0 * (1 - verified["duration_s"] / serial_time) if serial_time else 0.0

    # --- report ------------------------------------------------------
    park_events = [e for e in events if e["kind"] == "wait"]
    lines = [
        "Cooperative print -- de-confliction report",
        "=" * 68,
        f"toolhead1 source : {args.gcode1}",
        f"toolhead2 source : {args.gcode2}",
        (f"placement       : restored from {args.placement}"
         if args.placement else
         "placement       : NOT restored (no --placement given)"),
        f"clearance margin : {args.clearance:g} mm around every envelope",
        "",
        "Collisions predicted with no coordination : "
        f"{n_base} sampled instants",
        "Collisions remaining after the plan       : 0  (verified by replay)",
        "",
        f"Modelled duration, uncoordinated : {_fmt_hms(baseline['duration_s'])}",
        f"Modelled duration, coordinated   : {_fmt_hms(verified['duration_s'])}",
        f"Cost of coordination             : "
        f"{_fmt_hms(max(0.0, verified['duration_s'] - baseline['duration_s']))} of waiting",
        "",
        "What the second toolhead bought:",
    ]
    
    part_lo = min(geometry.TOOLHEAD1.to_world(*tp1.model_bbox[0])[1],
                  geometry.TOOLHEAD1.to_world(*tp1.model_bbox[1])[1],
                  geometry.TOOLHEAD2.to_world(*tp2.model_bbox[0])[1],
                  geometry.TOOLHEAD2.to_world(*tp2.model_bbox[1])[1])
    part_hi = max(geometry.TOOLHEAD1.to_world(*tp1.model_bbox[0])[1],
                  geometry.TOOLHEAD1.to_world(*tp1.model_bbox[1])[1],
                  geometry.TOOLHEAD2.to_world(*tp2.model_bbox[0])[1],
                  geometry.TOOLHEAD2.to_world(*tp2.model_bbox[1])[1])
    single = geometry.best_single_reach()
    single_span = single[1] - single[0]
    part_span = part_hi - part_lo
    lines += [
        f"  BUILD VOLUME  part spans {part_span:.1f}mm in Y; one toolhead "
        f"alone reaches {single_span:g}mm",
    ]
    if part_span > single_span + 1e-6:
        lines.append(f"                -> {part_span - single_span:.1f}mm of this part is "
                      f"beyond a single head's reach.")
        lines.append(f"                   It could not be printed on one machine. "
                      f"That is the gain.")
    else:
        lines.append(f"                -> this part fits within one head's reach, so the "
                      f"pair added no volume here.")
        lines.append(f"                   (If the real arms cannot each cross the bed, "
                      f"set REACH_Y in geometry.py.)")
    lines += [
        f"  SPEED         both heads printing at once: {concurrency:.0f}% of "
        f"working time ({saving:+.0f}% vs one machine)",
    ]
    if concurrency < 15:
        lines += [
            "                -> effectively serialised: safe, but no faster than",
            "                   one machine doing both halves. Only a concern if",
            "                   speed was the goal; for build volume it is not.",
        ]
    lines += [
        "",
        "Events inserted:",
    ]
    strategy = manifest.get("strategy")
    if strategy:
        blk = manifest.get("block_layers", 1)
        how = ("one chunk completely, then the other" if blk >= 10 ** 9
               else f"the two heads alternate every {blk} layer(s)")
        lines[-1:-1] = [
            f"Strategy: STRICT HAND-OFF -- {how}.",
            "  Interleaving the two heads was tried first and could not be made",
            "  collision-free: with this mount depth and this seam, any instant",
            "  of overlap is a contact. Hand-off is not a failure -- it prints",
            "  the part safely -- but it is why the concurrency above is 0%.",
            "",
        ]
    for (head, kind), n in sorted(kinds.items()):
        lines.append(f"  {head:10s} {kind:8s} x{n}")
    
    tp_of = {"toolhead1": tp1, "toolhead2": tp2}
    parks_by_head = {}
    for e in park_events:
        parks_by_head[e["head"]] = parks_by_head.get(e["head"], 0) + 1
    lines += [
        "",
        "Parks inserted: "
        + ", ".join(f"{h} x{n}" for h, n in sorted(parks_by_head.items())),
        f"Park pose (local): X={geometry.PARK_X:g} "
        f"Y={geometry.BED_Y_MAX + geometry.PARK_CLEARANCE:g}",
        "",
        "First park points (that head's own print-time -> layer):",
    ]
    for e in park_events[:40]:
        mv = tp_of[e["head"]].moves[e["move_index"]]
        lines.append(f"  {e['head']:10s} t={mv.t0:8.1f}s  layer {mv.layer:3d}  "
                      f"waits for event {e['params']['event_index']}")
    if len(park_events) > 40:
        lines.append(f"  ... and {len(park_events) - 40} more")
    lines += [
        "",
        "NOTE: the durations above use a straight-line feedrate model with no",
        "acceleration, so treat them as relative, not as print-time estimates.",
        "The ORDER and the collision count are what this tool is asserting.",
        "",
        "Both output files require macros.cfg (COOP_WAIT / COOP_NOTIFY) to be",
        "included in each printer's Klipper config, or they will error on the",
        "first COOP_ command.",
    ]
    report = args.outdir / "deconfliction_report.txt"
    report.write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines[5:12]))
    print(f"\nWrote:\n  {out1}\n  {out2}\n  {args.outdir / 'plan.json'}\n  {report}")
    print(f"\nNext:\n"
          f"  python3 step5_visualise.py {src1} {src2} \\\n"
          f"      --plan {args.outdir / 'plan.json'} --compare\n"
          f"  python3 step4_live_guard.py --simulate {args.outdir / 'plan.json'} \\\n"
          f"      {src1} {src2}\n"
          f"Print {out1.name} and {out2.name}; use the files above (not your\n"
          f"originals) for steps 4 and 5 -- the plan indexes into those.")


if __name__ == "__main__":
    main()
