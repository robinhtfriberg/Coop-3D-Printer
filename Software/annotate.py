from __future__ import annotations

import json
from pathlib import Path

import geometry
import sim


def annotate(gcode_path: Path, tp: sim.Toolpath, events: list[dict], head: str) -> list[str]:
    by_line_before: dict[int, list[str]] = {}
    by_line_after: dict[int, list[str]] = {}
    for e in events:
        if e["head"] != head:
            continue
        line = tp.moves[e["move_index"]].line
        if e["kind"] == "notify":
            idx = e["params"]["event_index"]
            by_line_after.setdefault(line, []).append(f"COOP_NOTIFY EVENT_INDEX={idx}")
        elif e["kind"] == "wait":
            th = geometry.TOOLHEADS[head]
            px, py = th.park_pose_local
            idx = e["params"]["event_index"]
            by_line_before.setdefault(line, []).append(
                f"COOP_WAIT EVENT_INDEX={idx} PARK_X={px:g} PARK_Y={py:g}")
        elif e["kind"] == "detour":
            offset = e["params"]["offset_y"]
            by_line_before.setdefault(line, []).append(f"COOP_DETOUR_BEGIN OFFSET_Y={offset:g}")
            by_line_after.setdefault(line, []).append("COOP_DETOUR_END")

    out = []
    with open(gcode_path, encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh):
            out.extend(by_line_before.get(line_no, []))
            out.append(raw.rstrip("\n"))
            out.extend(by_line_after.get(line_no, []))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gcode1", type=Path)
    ap.add_argument("gcode2", type=Path)
    ap.add_argument("plan", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("./annotated"))
    args = ap.parse_args()

    manifest = json.loads(args.plan.read_text())
    tp1, tp2 = sim.parse_toolpath(args.gcode1), sim.parse_toolpath(args.gcode2)
    args.outdir.mkdir(parents=True, exist_ok=True)

    out1 = annotate(args.gcode1, tp1, manifest["events"], "toolhead1")
    out2 = annotate(args.gcode2, tp2, manifest["events"], "toolhead2")
    (args.outdir / args.gcode1.name).write_text("\n".join(out1) + "\n")
    (args.outdir / args.gcode2.name).write_text("\n".join(out2) + "\n")
    print(f"Wrote {args.outdir / args.gcode1.name} and {args.outdir / args.gcode2.name}")


if __name__ == "__main__":
    main()
