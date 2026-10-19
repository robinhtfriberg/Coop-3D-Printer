"""Offline planner: turns two independent toolpaths into a single
manifest of checkpoint-gated coordination events, decided once before
printing rather than reactively during it
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import geometry
from geometry import TOOLHEADS, DEFAULT_CLEARANCE
import replay
import sim

MAX_PLAN_ITERATIONS = 8

DT = 0.1                    # prediction time-step, seconds
DETOUR_MAX_DURATION = 4.0   # only detour conflicts shorter than this
DETOUR_OFFSET = 30.0        # lateral Y nudge, mm
SEAM_LAYER_BUFFER = 3       # supporting layers kept ahead of the seam
SUBLAYER_CHECKPOINTS = 4    # progress notifications per layer
GUARD_S = 0.5                # safety margin either side of a predicted window
MERGE_GAP_S = 2 * GUARD_S    # windows closer than this are one conflict, not two


@dataclass
class Event:
    kind: str             # "wait" | "notify" | "detour"
    head: str              # toolhead this is inserted into
    move_index: int
    label: str
    params: dict = field(default_factory=dict)


def seam_events(tp_support, tp_dependent, buffer: int = SEAM_LAYER_BUFFER) -> list[Event]:
    """Wait/notify pairs enforcing the sloped-seam print order."""
    if not tp_support.layer_end_time or not tp_dependent.layer_end_time:
        return []

    by_layer_support: dict[int, list[int]] = {}
    for idx, mv in enumerate(tp_support.moves):
        by_layer_support.setdefault(mv.layer, []).append(idx)
    max_support_layer = max(by_layer_support)

    events: list[Event] = []
    checkpoint_id: dict[tuple[int, int], str] = {}
    notify_index = 1
    for layer, idxs in sorted(by_layer_support.items()):
        n = len(idxs)
        for step in range(1, SUBLAYER_CHECKPOINTS + 1):
            pos = (n * step) // SUBLAYER_CHECKPOINTS - 1
            if pos < 0:
                continue
            event_id = f"seam-{notify_index:04d}"
            events.append(Event("notify", "toolhead1", idxs[pos], event_id))
            checkpoint_id[(layer, step)] = event_id
            notify_index += 1

    by_layer_dep: dict[int, list[int]] = {}
    for idx, mv in enumerate(tp_dependent.moves):
        by_layer_dep.setdefault(mv.layer, []).append(idx)
    dep_layers = sorted(by_layer_dep)

    for layer in dep_layers[1:]:   # the first dependent layer sits on the bed
        required_layer = min(max_support_layer, layer - 1 + buffer)
        event_id = checkpoint_id.get((required_layer, SUBLAYER_CHECKPOINTS))
        if event_id is None:
            continue
        first_idx = by_layer_dep[layer][0]
        events.append(Event("wait", "toolhead2", first_idx, event_id, {"reason": "seam"}))
    return events


def _layer_bounds(tp: sim.Toolpath):
    """{layer: (first move index, last move index)}."""
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, m in enumerate(tp.moves):
        if m.layer >= 0:
            last[m.layer] = i
            first.setdefault(m.layer, i)
    return first, last


def alternating_events(tp1: sim.Toolpath, tp2: sim.Toolpath,
                        block_layers: int, lag_blocks: int = 0) -> list[Event]:
    f1, l1 = _layer_bounds(tp1)
    f2, l2 = _layer_bounds(tp2)
    layers = sorted(set(f1) & set(f2))
    if not layers:
        return []
    blocks = [layers[i:i + block_layers] for i in range(0, len(layers), block_layers)]

    events: list[Event] = []
    for j, blk in enumerate(blocks):
        # A_j : toolhead1 finished block j  ->  toolhead2 may start block j-lag
        events.append(Event("notify", "toolhead1", l1[blk[-1]], f"alt-a{j:05d}"))
        k = j - lag_blocks
        if 0 <= k < len(blocks):
            # The FIRST wait goes on move 0, not the first move of layer
            # 0: everything before ;LAYER:0 (home, Z lift, prime line) is
            # real motion, and a wait placed after it lets both heads run
            # their whole start-up sequence unguarded.
            at = 0 if k == 0 else f2[blocks[k][0]]
            events.append(Event("wait", "toolhead2", at, f"alt-a{j:05d}",
                                 {"reason": "alternate"}))
            # B_k : toolhead2 finished block k -> toolhead1 may start k+lag+1
            nxt = k + lag_blocks + 1
            if nxt < len(blocks):
                events.append(Event("notify", "toolhead2", l2[blocks[k][-1]],
                                     f"alt-b{k:05d}"))
                events.append(Event("wait", "toolhead1", f1[blocks[nxt][0]],
                                     f"alt-b{k:05d}", {"reason": "alternate"}))

    # Cheap structural check: on each head every notify must come before
    # any wait that the other head's release depends on. Ordering by move
    # index is what makes this a chain rather than a cycle.
    for head in ("toolhead1", "toolhead2"):
        idxs = [e.move_index for e in events if e.head == head]
        if idxs != sorted(idxs):
            raise RuntimeError(f"internal error: {head}'s alternating events "
                                f"are not in move order; that is a cycle waiting "
                                f"to happen")
    return events


def _yields(priority_head: str) -> str:
    return "toolhead2" if priority_head == "toolhead1" else "toolhead1"


def _move_index_before(tp: sim.Toolpath, t: float) -> int:
    """Last move that FINISHES at or before t -- used for a wait checkpoint,
    so the block takes effect before the danger zone is reached."""
    lo, hi, best = 0, len(tp.moves) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if tp.moves[mid].t1 <= t:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def _move_index_after(tp: sim.Toolpath, t: float) -> int:
    lo, hi, best = 0, len(tp.moves) - 1, len(tp.moves) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if tp.moves[mid].t0 >= t:
            best, hi = mid, mid - 1
        else:
            lo = mid + 1
    return best


def _raw_windows(tp1: sim.Toolpath, tp2: sim.Toolpath, clearance: float) -> list[tuple]:
    th1, th2 = TOOLHEADS["toolhead1"], TOOLHEADS["toolhead2"]
    total = max(tp1.total_time, tp2.total_time)
    windows = []
    in_conflict = False
    start = 0.0
    reason = ""
    t = 0.0
    while t <= total:
        p1, p2 = sim.position_at(tp1, t), sim.position_at(tp2, t)
        hit, why = sim.collision(th1, p1, th2, p2, clearance)
        if hit and not in_conflict:
            in_conflict, start, reason = True, t, why
        elif not hit and in_conflict:
            windows.append((start, t, reason))
            in_conflict = False
        t += DT
    if in_conflict:
        windows.append((start, total, reason))
    return windows


def _merge_windows(windows: list[tuple], gap: float = MERGE_GAP_S) -> list[tuple]:
    if not windows:
        return []
    windows = sorted(windows)
    merged = [list(windows[0])]
    for start, end, reason in windows[1:]:
        last = merged[-1]
        if start - last[1] <= gap:
            last[1] = max(last[1], end)
            if not reason.startswith("beam"):
                last[2] = reason   # a merged window is only a detour candidate
        else:                      # if EVERY sub-window in it was beam-only
            merged.append([start, end, reason])
    return [tuple(w) for w in merged]


def clearance_events(tp1: sim.Toolpath, tp2: sim.Toolpath, clearance: float = DEFAULT_CLEARANCE) -> list[Event]:
    raw = _raw_windows(tp1, tp2, clearance)
    events: list[Event] = []
    for event_id, (start, end, reason) in enumerate(_merge_windows(raw), start=1):
        guarded_start = max(0.0, start - GUARD_S)
        guarded_end = end + GUARD_S
        duration = guarded_end - guarded_start
        yielder = _yields("toolhead1")
        tp_yield = tp1 if yielder == "toolhead1" else tp2
        idx = _move_index_before(tp_yield, guarded_start)
        if reason.startswith("beam") and duration <= DETOUR_MAX_DURATION:
            events.append(Event("detour", yielder, idx, f"clr-{event_id:04d}",
                                 {"offset_y": DETOUR_OFFSET, "duration_s": round(duration, 2)}))
        else:
            wid = f"clr-{event_id:04d}"
            notifier = "toolhead1" if yielder == "toolhead2" else "toolhead2"
            notify_idx = _move_index_after(tp1 if notifier == "toolhead1" else tp2, guarded_end)
            events.append(Event("notify", notifier, notify_idx, wid))
            events.append(Event("wait", yielder, idx, wid, {"reason": "clearance"}))
    return events


def _merge_collision_windows(collisions: list[dict]) -> list[tuple]:
    """Group raw replay collision samples (already time-ordered) into
    windows, keeping the first and last sample of each so both heads'
    progress at the window's boundaries is available."""
    if not collisions:
        return []
    windows = []
    start, prev, reason = collisions[0], collisions[0], collisions[0]["reason"]
    for c in collisions[1:]:
        if c["t"] - prev["t"] <= MERGE_GAP_S:
            prev = c
            if not c["reason"].startswith("beam"):
                reason = c["reason"]
        else:
            windows.append((start, prev, reason))
            start, prev, reason = c, c, c["reason"]
    windows.append((start, prev, reason))
    return windows


def _events_for_residual(tp: dict, collisions: list[dict], start_id: int) -> list[Event]:
    """Turn collisions surviving a replay of the plan-so-far into more
    events, using each head's own PROGRESS at the window boundary (not
    the shared replay clock -- a head already delayed by an earlier wait
    is not at the same point in its toolpath as the wall clock would
    suggest, and using wall-clock time here would silently re-introduce
    the same before/after mixup already fixed once in clearance_events)."""
    events = []
    other = {"toolhead1": "toolhead2", "toolhead2": "toolhead1"}
    for i, (start, end, reason) in enumerate(_merge_collision_windows(collisions), start=start_id):
        yielder = _yields("toolhead1")
        notifier = other[yielder]
        if start[yielder]["waiting"] and end[yielder]["waiting"]:
            raise RuntimeError(
                f"toolhead2 collided ({reason}) while already parked, from "
                f"t={start['t']}s to t={end['t']}s. This is a geometry "
                f"problem (the park pose does not clear the other head's "
                f"envelope), not a scheduling one -- see geometry.py's "
                f"_check_park_clears_bed."
            )
        guarded_start = max(0.0, start[yielder]["progress"] - GUARD_S)
        guarded_end = end[notifier]["progress"] + GUARD_S
        idx = _move_index_before(tp[yielder], guarded_start)
        duration = guarded_end - guarded_start
        if reason.startswith("beam") and duration <= DETOUR_MAX_DURATION:
            events.append(Event("detour", yielder, idx, f"fix-{i:04d}",
                                 {"offset_y": DETOUR_OFFSET, "duration_s": round(duration, 2)}))
        else:
            wid = f"fix-{i:04d}"
            notify_idx = _move_index_after(tp[notifier], guarded_end)
            events.append(Event("notify", notifier, notify_idx, wid))
            events.append(Event("wait", yielder, idx, wid, {"reason": "residual"}))
    return events


def _finalize(events: list[Event]) -> dict:
    """Assign the integer EVENT_INDEX Klipper actually compares (a cheap
    ">=" against a per-toolhead counter), scoped per notifying head since
    SYNC_STATE.completed_event lives on that head, not globally. String
    labels stay in the JSON for readability but never reach the macros."""
    notify_index: dict[str, int] = {}
    counters = {"toolhead1": 0, "toolhead2": 0}
    for e in sorted(events, key=lambda e: (e.head, e.move_index)):
        if e.kind == "notify":
            counters[e.head] += 1
            e.params["event_index"] = counters[e.head]
            notify_index[e.label] = counters[e.head]
    for e in events:
        if e.kind == "wait":
            if e.label not in notify_index:
                raise RuntimeError(f"wait '{e.label}' on {e.head} has no matching notify")
            e.params["event_index"] = notify_index[e.label]
    events = _consolidate_waits(events)
    events.sort(key=lambda e: (e.head, e.move_index))
    return {"protocol": "coop-plan-v2", "events": [e.__dict__ for e in events]}


def _consolidate_waits(events: list[Event]) -> list[Event]:
    out = [e for e in events if e.kind != "wait"]
    by_head: dict[str, list[Event]] = {}
    for e in events:
        if e.kind == "wait":
            by_head.setdefault(e.head, []).append(e)

    for head, waits in by_head.items():
        waits.sort(key=lambda e: (e.move_index, e.params["event_index"]))
        merged: dict[int, Event] = {}
        for w in waits:
            prev = merged.get(w.move_index)
            if prev is None or w.params["event_index"] > prev.params["event_index"]:
                merged[w.move_index] = w
        highest = 0
        for move_index in sorted(merged):
            w = merged[move_index]
            if w.params["event_index"] <= highest:
                continue          # already guaranteed by an earlier wait
            highest = w.params["event_index"]
            out.append(w)
    return out


def plan(gcode1: Path, gcode2: Path, max_iterations: int = MAX_PLAN_ITERATIONS,
         on_progress=None) -> dict:
    def _say(msg):
        # A real print is ~100k moves and takes a couple of minutes to
        # plan. Silence for that long is indistinguishable from a hang,
        # so the caller gets told what is happening.
        if on_progress:
            on_progress(msg)

    tp1, tp2 = sim.parse_toolpath(gcode1), sim.parse_toolpath(gcode2)
    tp = {"toolhead1": tp1, "toolhead2": tp2}
    _say(f"seam dependencies across {len(tp1.layer_end_time)} layers...")
    events = seam_events(tp1, tp2)
    _say(f"  {len(events)} seam events; sweeping for clearance conflicts...")
    events += clearance_events(tp1, tp2)
    _say(f"  {len(events)} events total before verification")

    report = None
    best = None
    stalled = 0
    for iteration in range(max_iterations):
        _say(f"verifying (pass {iteration + 1}/{max_iterations})...")
        manifest = {"protocol": "coop-plan-v2", "events": [e.__dict__ for e in events]}
        report = replay.replay(tp1, tp2, manifest)
        if not report["collisions"]:
            events.sort(key=lambda e: (e.head, e.move_index))
            final = _finalize(events)
            # _finalize drops provably-redundant waits (see
            # _consolidate_waits). "Provably" still gets checked: re-run
            # the same replay against the reduced plan before returning
            # it, so a reasoning error there can never leave the library
            # handing back a plan that collides.
            recheck = replay.replay(tp1, tp2, final)
            if recheck["collisions"]:
                raise RuntimeError(
                    f"internal error: wait consolidation introduced "
                    f"{len(recheck['collisions'])} collisions. This is a bug in "
                    f"_consolidate_waits, not a property of your G-code.")
            return final
        n = len(report["collisions"])
        fixes = _events_for_residual(tp, report["collisions"], start_id=len(events) + 1)
        _say(f"  {n} colliding samples remain -> {len(fixes)} more events")
        if not fixes:
            break
        # Patching per conflict only converges when the conflicts are
        # isolated. When the two chunks genuinely cannot share the
        # machine, each wait just moves the conflict and the count
        # wanders instead of falling -- so stop early rather than
        # grinding out every pass to reach the same conclusion.
        if best is None or n < best * 0.95:
            best, stalled = n, 0
        else:
            stalled += 1
            if stalled >= 3:
                _say("  not improving; per-conflict patching cannot resolve this")
                break
        events += fixes

    # ---- fallback: stop trying to interleave, hand off instead ------
    _say("per-conflict patching did not converge; "
         "falling back to strict hand-off (no concurrency)")
    for block in (1, 2, 4, 8, 16, 32, 64, 128, 10 ** 9):
        # lag=1 keeps toolhead1 a block ahead, so the material
        # toolhead2's sloped overhang lands on already exists.
        events = alternating_events(tp1, tp2, block, lag_blocks=1)
        if not events:
            break
        candidate = _finalize(list(events))
        check = replay.replay(tp1, tp2, candidate)
        label = ("whole print" if block >= 10 ** 9 else f"every {block} layer(s)")
        _say(f"  hand-off {label}: {len(check['collisions'])} collisions")
        if not check["collisions"]:
            candidate["strategy"] = ("alternating"
                                     if block < 10 ** 9 else "serialised")
            candidate["block_layers"] = block
            return candidate

    raise RuntimeError(
        f"no collision-free plan exists for these two files, not even "
        f"printing one chunk completely before the other. That is a "
        f"GEOMETRY result, not a scheduling one: the park pose itself does "
        f"not clear the other toolhead's working area. Check "
        f"MOUNT_WIDTH_X / MOUNT_DEPTH_Y / PARK_CLEARANCE in geometry.py "
        f"against the real machine -- a parked head currently keeps "
        f"{geometry_guarantee():.1f}mm from one printing at the near bed edge."
    )


def geometry_guarantee() -> float:
    return geometry.parked_clearance_guarantee()
