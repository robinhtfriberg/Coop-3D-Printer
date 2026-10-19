from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import geometry
import place
import replay
import sim

ACCENT = {"toolhead1": "#c1691f", "toolhead2": "#1f7d82"}
PARKED = "#c9c2ae"
DETOUR = "#7a5fb0"

BED = (geometry.BED_X_MIN, geometry.BED_X_MAX,
       geometry.BED_Y_MIN, geometry.BED_Y_MAX)


def _world(head: str, pos):
    return geometry.TOOLHEADS[head].to_world(*pos)


def build_axes(ax, title):
    bx0, bx1, by0, by1 = BED
    pc = geometry.PARK_CLEARANCE
    ax.set_xlim(bx0 - 12, bx1 + 12)
    ax.set_ylim(by0 - pc - 12, by1 + pc + 12)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, facecolor="#fffdf9",
                            edgecolor="#1c2126", linewidth=1.2, zorder=1))
    for y0 in (by1, by0 - pc):
        ax.add_patch(Rectangle((bx0, y0), bx1 - bx0, pc, facecolor="#e7e2d4",
                                edgecolor="none", zorder=0))
    ax.axhline(by1, color="#b9b2a2", lw=0.8, zorder=1)
    ax.axhline(by0, color="#b9b2a2", lw=0.8, zorder=1)
    ax.text(bx1 + 2, by1 + pc / 2, "park", fontsize=6.5, color="#8b8371",
            va="center", ha="right")
    ax.text(bx1 + 2, by0 - pc / 2, "park", fontsize=6.5, color="#8b8371",
            va="center", ha="right")


class Panel:
    """One animated view of one replay."""

    def __init__(self, ax, frames, title, extruding_lookup):
        self.frames = frames
        self.extruding = extruding_lookup
        build_axes(ax, title)
        mw, md = geometry.MOUNT_WIDTH_X, geometry.MOUNT_DEPTH_Y
        bt = geometry.BEAM_THICKNESS_Y
        # Opposing machines: toolhead1's arm is anchored on the left of
        # the bed and reaches right, toolhead2's on the right reaching
        # left.
        self.anchor = {"toolhead1": BED[0], "toolhead2": BED[1]}
        self.mw, self.md, self.bt = mw, md, bt

        self.arm, self.mount, self.dep = {}, {}, {}
        self.dep_xy = {}
        for h in ("toolhead1", "toolhead2"):
            self.arm[h] = Rectangle((0, 0), 0, bt, facecolor=ACCENT[h], alpha=0.22,
                                     edgecolor="none", zorder=2)
            self.mount[h] = Rectangle((0, 0), mw, md, facecolor=ACCENT[h],
                                       edgecolor="#1c2126", lw=0.8, zorder=4)
            self.dep[h] = Line2D([], [], color=ACCENT[h], lw=1.4, alpha=0.9,
                                  zorder=3, solid_capstyle="round")
            self.dep_xy[h] = ([], [])
            ax.add_patch(self.arm[h]); ax.add_patch(self.mount[h])
            ax.add_line(self.dep[h])

        self.flash = Rectangle((BED[0], BED[2] - geometry.PARK_CLEARANCE),
                                BED[1] - BED[0],
                                (BED[3] + geometry.PARK_CLEARANCE)
                                - (BED[2] - geometry.PARK_CLEARANCE),
                                facecolor="none", edgecolor="#c53f3f", lw=0, zorder=6)
        ax.add_patch(self.flash)
        self.txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                            ha="left", fontsize=7.5, family="monospace")
        self.was_laying = {"toolhead1": False, "toolhead2": False}
        self.cum = []
        n = 0
        for f in frames:
            n += 1 if f["collision"] else 0
            self.cum.append(n)

    def artists(self):
        return (list(self.arm.values()) + list(self.mount.values())
                + list(self.dep.values()) + [self.flash, self.txt])

    def update(self, i):
        if i >= len(self.frames):
            return self.artists()
        f = self.frames[i]
        for h in ("toolhead1", "toolhead2"):
            x, y, z = _world(h, f[h]["pos"])
            state = f[h]["state"]
            self.mount[h].set_xy((x - self.mw / 2, y - self.md / 2))
            a0, a1 = sorted((self.anchor[h], x))
            self.arm[h].set_xy((a0, y - self.bt / 2))
            self.arm[h].set_width(max(a1 - a0, 0.01))

            colour = ACCENT[h] if state == "printing" else (
                DETOUR if state == "detour" else PARKED)
            live = state in ("printing", "detour")
            self.mount[h].set_facecolor(colour)
            self.mount[h].set_alpha(1.0 if live else 0.45)
            self.arm[h].set_facecolor(colour)
            self.arm[h].set_alpha(0.22 if live else 0.09)

            xs, ys = self.dep_xy[h]
            laying = live and self.extruding[h](f[h]["progress"])
            if laying:
                if xs and not self.was_laying[h]:
                    xs.append(float("nan")); ys.append(float("nan"))
                xs.append(x); ys.append(y)
                self.dep[h].set_data(xs, ys)
            self.was_laying[h] = laying

        self.flash.set_linewidth(3 if f["collision"] else 0)
        self.txt.set_text(
            f"t = {f['t']:7.1f}s\n"
            f"th1: {f['toolhead1']['state']:>8}\n"
            f"th2: {f['toolhead2']['state']:>8}\n"
            f"collisions: {self.cum[i]}")
        return self.artists()


def _extruding_lookup(tp):
    """progress-seconds -> was the toolhead extruding at that moment."""
    ends = [m.t1 for m in tp.moves]
    flags = [m.extruding for m in tp.moves]
    import bisect

    def f(t):
        if not ends:
            return False
        i = bisect.bisect_left(ends, t)
        return flags[min(i, len(flags) - 1)]
    return f


def run(gcode1: Path, gcode2: Path, plan: dict, compare: bool, save: str | None,
        stride: int, fps: int, dt: float):
    tp1, tp2 = sim.parse_toolpath(gcode1), sim.parse_toolpath(gcode2)

    srcs = plan.get("sources") or {}
    mismatches = (place.check_fingerprint(tp1, srcs.get("toolhead1"), "toolhead1")
                  + place.check_fingerprint(tp2, srcs.get("toolhead2"), "toolhead2"))
    if mismatches:
        print("ERROR: this plan was not built for these G-code files:", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        for head in ("toolhead1", "toolhead2"):
            f = (srcs.get(head) or {}).get("file")
            if f:
                print(f"  plan's {head} was: {f}", file=sys.stderr)
        sys.exit(1)

    lookup = {"toolhead1": _extruding_lookup(tp1), "toolhead2": _extruding_lookup(tp2)}

    runs = [("With the plan  (coordinated)", plan)]
    if compare:
        runs.append(("No coordination", {"protocol": "coop-plan-v2", "events": []}))

    reports = []
    for label, manifest in runs:
        rep = replay.replay(tp1, tp2, manifest, record_frames=True, dt=dt)
        rep["frames"] = rep["frames"][::stride]
        reports.append((label, rep))
        print(f"  {label}: {rep['duration_s']:.0f}s modelled, "
               f"{len(rep['collisions'])} colliding samples")

    fig, axes = plt.subplots(1, len(reports),
                              figsize=(6.8 * len(reports), 6.6), squeeze=False)

    panels = []
    for ax, (label, rep) in zip(axes[0], reports):
        n_coll = len(rep["collisions"])
        title = f"{label}   ({n_coll} collisions)"
        panels.append(Panel(ax, rep["frames"], title, lookup))

    handles = [
        Line2D([], [], color=ACCENT["toolhead1"], lw=3, label="toolhead1 material"),
        Line2D([], [], color=ACCENT["toolhead2"], lw=3, label="toolhead2 material"),
        Line2D([], [], color=PARKED, lw=8, label="parked"),
        Line2D([], [], color="#c53f3f", lw=3, label="collision"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)

    n_frames = max(len(p.frames) for p in panels)

    def update(i):
        out = []
        for p in panels:
            out += p.update(i)
        return out

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / fps, blit=False)

    if save:
        out = Path(save)
        if out.suffix.lower() == ".gif":
            anim.save(out, writer="pillow", fps=fps)
        else:
            try:
                anim.save(out, writer="ffmpeg", fps=fps, dpi=130,
                           extra_args=["-pix_fmt", "yuv420p"])
            except Exception as exc:
                print(f"  ffmpeg unavailable ({exc}); writing a GIF instead")
                anim.save(out.with_suffix(".gif"), writer="pillow", fps=fps)
                out = out.with_suffix(".gif")
        print(f"Wrote {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gcode1", type=Path)
    ap.add_argument("gcode2", type=Path)
    ap.add_argument("--plan", type=Path, default=None,
                     help="plan.json from step 3 (omit to show the raw, "
                          "uncoordinated pair)")
    ap.add_argument("--compare", action="store_true",
                     help="show with-plan and without-plan side by side")
    ap.add_argument("--save", default=None, help="write to .mp4 / .gif "
                     "instead of opening a window")
    ap.add_argument("--stride", type=int, default=3,
                     help="use every Nth sampled frame (default 3)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dt", type=float, default=0.15,
                     help="replay sample interval, simulated seconds")
    args = ap.parse_args()

    for p in (args.gcode1, args.gcode2):
        if not p.exists():
            print(f"ERROR: {p} does not exist", file=sys.stderr)
            sys.exit(1)

    if args.plan:
        plan = json.loads(args.plan.read_text())
    else:
        plan = {"protocol": "coop-plan-v2", "events": []}
        print("  (no --plan given: showing the uncoordinated pair)")

    if args.save:
        matplotlib.use("Agg")
    run(args.gcode1, args.gcode2, plan, args.compare, args.save,
        args.stride, args.fps, args.dt)


if __name__ == "__main__":
    main()
