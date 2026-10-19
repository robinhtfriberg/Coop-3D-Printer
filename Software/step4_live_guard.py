from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import geometry
import place
import sim

_GUARANTEE = geometry.parked_clearance_guarantee()
WARN_MM = round(_GUARANTEE * 0.80, 1)      # 12.0mm at the current geometry
RETREAT_MM = round(_GUARANTEE * 0.50, 1)   #  7.5mm
CRITICAL_MM = round(_GUARANTEE * 0.25, 1)  #  3.8mm

ALERT_REPEAT_S = 5.0        # don't spam the same alert every poll
STALE_AFTER_S = 3.0         # no fresh position this long => fail safe

HYSTERESIS_MM = 6.0


def envelope_gap(pos1_local, pos2_local) -> tuple[float, str]:
    """Smallest clearance between the two toolheads' envelopes, in mm."""
    th1, th2 = geometry.TOOLHEAD1, geometry.TOOLHEAD2
    boxes1 = sim.envelope(th1, pos1_local)
    boxes2 = sim.envelope(th2, pos2_local)

    best = math.inf
    worst_pair = ""
    names = ("mount", "arm")
    for i, a in enumerate(boxes1):
        for j, b in enumerate(boxes2):
            # gap per axis: positive if separated, negative if overlapping
            gaps = []
            for k in range(3):
                lo_a, hi_a = a[2 * k], a[2 * k + 1]
                lo_b, hi_b = b[2 * k], b[2 * k + 1]
                gaps.append(max(lo_a - hi_b, lo_b - hi_a))
            if all(g <= 0 for g in gaps):
                d = max(gaps)            # overlapping: penetration depth
            else:
                d = math.sqrt(sum(g * g for g in gaps if g > 0))
            if d < best:
                best = d
                worst_pair = f"{names[i]}1-{names[j]}2"
    return best, worst_pair


@dataclass
class GuardState:
    closest_seen: float = math.inf
    closest_at: float = 0.0
    alerts: int = 0
    retreats: int = 0
    last_alert_t: float = -1e9
    last_level: str = "ok"
    stopped: bool = False
    log: list = field(default_factory=list)


_LEVELS = ("ok", "warn", "retreat", "critical")


def level_for(gap: float, current: str = "ok") -> str:
    """Escalate the moment a threshold is crossed; de-escalate only once
    the gap has recovered past it by HYSTERESIS_MM."""
    def raw(g):
        if g <= CRITICAL_MM:
            return "critical"
        if g <= RETREAT_MM:
            return "retreat"
        if g <= WARN_MM:
            return "warn"
        return "ok"

    new = raw(gap)
    if _LEVELS.index(new) >= _LEVELS.index(current):
        return new              # escalation is always immediate

    threshold = {"warn": WARN_MM, "retreat": RETREAT_MM,
                 "critical": CRITICAL_MM}.get(current)
    if threshold is None or gap > threshold + HYSTERESIS_MM:
        return new
    return current


class Actions:
    """What the guard does when a threshold trips. Subclassed by the
    simulator so the decision logic under test is literally the same
    code that runs against hardware."""

    def alert(self, msg: str) -> None:
        sys.stdout.write(f"\a  ** ALERT ** {msg}\n")
        sys.stdout.flush()

    def retreat_secondary(self, reason: str) -> None:
        raise NotImplementedError

    def emergency_stop(self, reason: str) -> None:
        raise NotImplementedError

    def positions(self) -> tuple[tuple, tuple]:
        """(toolhead1_local, toolhead2_local). Raise to signal 'unknown'."""
        raise NotImplementedError

    def printing(self) -> bool:
        return True

    def clock(self) -> float:
        """Seconds since the print started. Wall-clock live; the print's
        own modelled time when simulated, so a simulated run's timeline
        reads the same as the real one instead of collapsing to zero."""
        return time.time()


class LiveActions(Actions):
    def __init__(self, c1, c2, webhook: str | None = None, dry_run: bool = False):
        self.c1, self.c2 = c1, c2
        self.webhook = webhook
        self.dry_run = dry_run

    def positions(self):
        return self.c1.local_position(), self.c2.local_position()

    def printing(self) -> bool:
        try:
            return self.c1.print_state() == "printing" or self.c2.print_state() == "printing"
        except Exception:
            return True     # can't tell => assume live, keep guarding

    def alert(self, msg: str) -> None:
        super().alert(msg)
        if self.webhook:
            try:
                import requests
                requests.post(self.webhook, json={"text": f"[coop-print] {msg}"}, timeout=5)
            except Exception as exc:
                print(f"     (webhook failed: {exc})")

    def retreat_secondary(self, reason: str) -> None:
        px, py = geometry.TOOLHEAD2.park_pose_local
        print(f"  >> RETREAT toolhead2 to park ({px:g},{py:g}): {reason}")
        if self.dry_run:
            print("     (dry run -- no command sent)")
            return
        self.c2.pause()
        self.c2.run_script(
            "SAVE_GCODE_STATE NAME=coop_guard_retreat\n"
            "G91\nG1 E-2 F2400\nG90\n"
            f"G1 X{px:g} Y{py:g} F9000\n"
            "M400")

    def emergency_stop(self, reason: str) -> None:
        print(f"  >> EMERGENCY STOP both machines: {reason}")
        if self.dry_run:
            print("     (dry run -- no command sent)")
            return
        for c in (self.c1, self.c2):
            try:
                c.emergency_stop()
            except Exception as exc:
                print(f"     !! could not reach a printer to stop it: {exc}")


class SimulatedActions(Actions):
    """Replays a planned print through the identical decision logic."""

    def __init__(self, tp1, tp2, manifest, speed: float):
        import replay as _replay
        self.report = _replay.replay(tp1, tp2, manifest, record_frames=True, dt=0.15)
        self.frames = self.report["frames"]
        self.i = 0
        self.speed = speed
        self.retreated = []
        self.stopped = False

    def positions(self):
        if self.i >= len(self.frames):
            raise StopIteration
        f = self.frames[self.i]
        self.i += 1
        self.t = f["t"]
        # frames carry LOCAL positions, same as a printer would report
        return tuple(f["toolhead1"]["pos"]), tuple(f["toolhead2"]["pos"])

    def clock(self) -> float:
        return getattr(self, "t", 0.0)

    def printing(self) -> bool:
        return self.i < len(self.frames)

    def retreat_secondary(self, reason: str) -> None:
        self.retreated.append((self.i, reason))
        print(f"  >> [sim] would retreat toolhead2: {reason}")

    def emergency_stop(self, reason: str) -> None:
        self.stopped = True
        print(f"  >> [sim] would EMERGENCY STOP: {reason}")


def guard_loop(actions: Actions, poll_s: float, state: GuardState,
               retreat_hold_s: float = 5.0, verbose: bool = True) -> GuardState:
    t0 = actions.clock()
    last_good = time.time()
    retreat_until = -1e9

    while True:
        try:
            p1, p2 = actions.positions()
            last_good = time.time()
        except StopIteration:
            break
        except Exception as exc:
            
            if time.time() - last_good > STALE_AFTER_S:
                actions.alert(f"lost contact with a printer for "
                               f"{time.time() - last_good:.1f}s ({exc}) -- "
                               f"cannot verify clearance, failing safe")
                try:
                    actions.retreat_secondary("comms lost -- failing safe")
                except Exception as inner:
                    print(f"     !! could not retreat either: {inner}")
                state.stopped = True
                break
            time.sleep(poll_s)
            continue

        now = actions.clock()
        gap, pair = envelope_gap(p1, p2)
        if gap < state.closest_seen:
            state.closest_seen = gap
            state.closest_at = now - t0

        lvl = level_for(gap, state.last_level)
        if lvl != "ok":
            state.log.append((round(now - t0, 2), round(gap, 2), lvl, pair))

        if lvl == "critical":
            actions.emergency_stop(
                f"{gap:.1f}mm between {pair} (limit {CRITICAL_MM:g}mm)")
            state.stopped = True
            break

        if lvl == "retreat" and now >= retreat_until:
            state.retreats += 1
            actions.retreat_secondary(
                f"{gap:.1f}mm between {pair} (limit {RETREAT_MM:g}mm)")
            retreat_until = now + retreat_hold_s

        if lvl == "warn" and (now - state.last_alert_t) > ALERT_REPEAT_S:
            state.alerts += 1
            state.last_alert_t = now
            actions.alert(f"toolheads {gap:.1f}mm apart ({pair}) at t={now - t0:.0f}s")

        if verbose and lvl != state.last_level:
            arrow = "^" if _LEVELS.index(lvl) > _LEVELS.index(state.last_level) else "v"
            print(f"  t={now - t0:7.1f}s  gap={gap:7.1f}mm  {pair:14s} "
                  f"{arrow} {lvl.upper()}")
        state.last_level = lvl

        if not actions.printing():
            break
        if poll_s:
            time.sleep(poll_s)
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host1", default=None, help="toolhead1 Moonraker host")
    ap.add_argument("--host2", default=None, help="toolhead2 Moonraker host")
    ap.add_argument("--port", type=int, default=7125)
    ap.add_argument("--poll", type=float, default=0.25, help="seconds between polls")
    ap.add_argument("--webhook", default=None,
                     help="POST alerts here as {'text': ...} (Slack-compatible)")
    ap.add_argument("--dry-run", action="store_true",
                     help="live positions, but never send a command")
    ap.add_argument("--simulate", nargs=3, metavar=("PLAN", "GCODE1", "GCODE2"),
                     help="replay a plan through the same logic, no hardware")
    ap.add_argument("--sim-speed", type=float, default=200.0,
                     help="simulated-mode speedup (default 200x)")
    ap.add_argument("--warn", type=float, default=None)
    ap.add_argument("--retreat", type=float, default=None)
    ap.add_argument("--critical", type=float, default=None)
    args = ap.parse_args()

    global WARN_MM, RETREAT_MM, CRITICAL_MM
    if args.warn is not None:
        WARN_MM = args.warn
    if args.retreat is not None:
        RETREAT_MM = args.retreat
    if args.critical is not None:
        CRITICAL_MM = args.critical
    if not (WARN_MM > RETREAT_MM > CRITICAL_MM):
        print("ERROR: thresholds must satisfy warn > retreat > critical",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 68)
    print("STEP 4  live clearance guard")
    print(f"  warn <= {WARN_MM:g}mm | retreat <= {RETREAT_MM:g}mm | "
          f"E-STOP <= {CRITICAL_MM:g}mm")
    print(f"  (a correctly parked toolhead sits {_GUARANTEE:g}mm clear, so "
          f"thresholds stay below that)")
    if WARN_MM >= _GUARANTEE:
        print(f"  !! WARNING: warn threshold {WARN_MM:g}mm is at or above the "
               f"{_GUARANTEE:g}mm a normal park provides -- this will alert "
               f"continuously during correct operation.")
    print("=" * 68)

    if args.simulate:
        plan_path, g1, g2 = args.simulate
        manifest = json.loads(Path(plan_path).read_text())
        tp1, tp2 = sim.parse_toolpath(Path(g1)), sim.parse_toolpath(Path(g2))
        srcs = manifest.get("sources") or {}
        mismatches = (place.check_fingerprint(tp1, srcs.get("toolhead1"), "toolhead1")
                      + place.check_fingerprint(tp2, srcs.get("toolhead2"), "toolhead2"))
        if mismatches:
            print("ERROR: this plan was not built for these G-code files:",
                  file=sys.stderr)
            for m in mismatches:
                print(f"  {m}", file=sys.stderr)
            sys.exit(1)
        actions = SimulatedActions(tp1, tp2, manifest, args.sim_speed)
        print(f"  SIMULATED against {plan_path} ({len(actions.frames)} samples)\n")
        state = guard_loop(actions, 0.0, GuardState())
    else:
        if not (args.host1 and args.host2):
            print("ERROR: give --host1 and --host2, or use --simulate",
                  file=sys.stderr)
            sys.exit(1)
        import orchestrator
        c1 = orchestrator.MoonrakerClient(orchestrator.Endpoint(args.host1, args.port))
        c2 = orchestrator.MoonrakerClient(orchestrator.Endpoint(args.host2, args.port))
        for name, c in (("toolhead1", c1), ("toolhead2", c2)):
            if not c.is_ready():
                print(f"ERROR: {name} at {c.endpoint.base_url} is not ready",
                      file=sys.stderr)
                sys.exit(1)
        print(f"  watching {args.host1} and {args.host2}"
               + ("  [DRY RUN]" if args.dry_run else "") + "\n")
        state = guard_loop(LiveActions(c1, c2, args.webhook, args.dry_run),
                            args.poll, GuardState())

    print("\n" + "=" * 68)
    print(f"  closest approach : {state.closest_seen:.1f}mm "
          f"(at t={state.closest_at:.1f}s)")
    print(f"  alerts raised    : {state.alerts}")
    print(f"  retreats ordered : {state.retreats}")
    print(f"  emergency stop   : {'YES' if state.stopped else 'no'}")
    if state.closest_seen > WARN_MM:
        print("  -> the two toolheads never came within the warning distance.")
    print("=" * 68)
    return 0 if not state.stopped else 3


if __name__ == "__main__":
    sys.exit(main())
