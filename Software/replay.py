from __future__ import annotations

from dataclasses import dataclass, field

from geometry import TOOLHEADS, DEFAULT_CLEARANCE
import sim

def _move_index_at(tp: sim.Toolpath, t: float) -> int:
    lo, hi, best = 0, len(tp.moves) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if tp.moves[mid].t1 <= t:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def _fires_at(tp: sim.Toolpath, ev: dict) -> float:
    m = tp.moves[ev["move_index"]]
    return m.t0 if ev["kind"] in ("wait", "detour") else m.t1


@dataclass
class _HeadState:
    progress: float = 0.0
    next_event: int = 0
    waiting_for: str | None = None
    notified: set = field(default_factory=set)
    detour_until: float | None = None
    detour_offset: float = 0.0


def replay(tp1: sim.Toolpath, tp2: sim.Toolpath, manifest: dict,
           clearance: float = DEFAULT_CLEARANCE, dt: float = 0.1,
           record_frames: bool = False):
    """record_frames=True additionally returns every sampled frame (not
    just collisions) under report["frames"] -- used by demo_gen.py to
    animate a run; the planner never needs it, so it defaults off and
    costs nothing for the convergence loop."""
    th = {"toolhead1": TOOLHEADS["toolhead1"], "toolhead2": TOOLHEADS["toolhead2"]}
    tp = {"toolhead1": tp1, "toolhead2": tp2}
    events = {"toolhead1": [], "toolhead2": []}
    for e in manifest["events"]:
        events[e["head"]].append(e)
    for h in events:
        events[h].sort(key=lambda e: e["move_index"])

    state = {"toolhead1": _HeadState(), "toolhead2": _HeadState()}
    other = {"toolhead1": "toolhead2", "toolhead2": "toolhead1"}
    collisions = []
    frames = []
    t = 0.0
    deadline = (tp1.total_time + tp2.total_time) * 2.0 + 600.0

    while t <= deadline:
        for h in ("toolhead1", "toolhead2"):
            st = state[h]
            evs = events[h]
            while (st.next_event < len(evs)
                   and _fires_at(tp[h], evs[st.next_event]) <= st.progress):
                ev = evs[st.next_event]
                if ev["kind"] == "notify":
                    st.notified.add(ev["label"])
                elif ev["kind"] == "wait":
                    if ev["label"] not in state[other[h]].notified:
                        st.waiting_for = ev["label"]
                elif ev["kind"] == "detour":
                    st.detour_until = st.progress + ev["params"]["duration_s"]
                    st.detour_offset = ev["params"]["offset_y"]
                st.next_event += 1
            if st.waiting_for and st.waiting_for in state[other[h]].notified:
                st.waiting_for = None
            if not st.waiting_for:
                st.progress = min(tp[h].total_time, st.progress + dt)

        positions = {}
        for h in ("toolhead1", "toolhead2"):
            st = state[h]
            x, y, z = sim.position_at(tp[h], st.progress)
            if st.waiting_for or st.progress >= tp[h].total_time:
                x, y = th[h].park_pose_local
            elif st.detour_until and st.progress < st.detour_until:
                y += st.detour_offset
            else:
                st.detour_until = None
            positions[h] = (x, y, z)

        hit, reason = sim.collision(th["toolhead1"], positions["toolhead1"],
                                     th["toolhead2"], positions["toolhead2"], clearance)
        if hit:
            collisions.append({
                "t": round(t, 3), "reason": reason,
                "toolhead1": {"pos": positions["toolhead1"], "progress": state["toolhead1"].progress,
                              "waiting": bool(state["toolhead1"].waiting_for)},
                "toolhead2": {"pos": positions["toolhead2"], "progress": state["toolhead2"].progress,
                              "waiting": bool(state["toolhead2"].waiting_for)},
            })

        if record_frames:
            def _state_label(h):
                st = state[h]
                if st.waiting_for:
                    return "parked"
                if st.detour_until and st.progress < st.detour_until:
                    return "detour"
                if st.progress >= tp[h].total_time:
                    return "done"
                return "printing"
            frames.append({
                "t": round(t, 3),
                "toolhead1": {"pos": [round(v, 3) for v in positions["toolhead1"]],
                              "state": _state_label("toolhead1"),
                              "progress": round(state["toolhead1"].progress, 3)},
                "toolhead2": {"pos": [round(v, 3) for v in positions["toolhead2"]],
                              "state": _state_label("toolhead2"),
                              "progress": round(state["toolhead2"].progress, 3)},
                "collision": hit,
            })

        
        movable = any(not state[h].waiting_for
                      and state[h].progress < tp[h].total_time
                      for h in ("toolhead1", "toolhead2"))
        blocked = [h for h in ("toolhead1", "toolhead2") if state[h].waiting_for]
        
        releasable = any(state[h].waiting_for in state[other[h]].notified
                         for h in blocked)
       
        pending = any(
            state[h].next_event < len(events[h])
            and _fires_at(tp[h], events[h][state[h].next_event]) <= state[h].progress
            for h in ("toolhead1", "toolhead2"))
        if blocked and not movable and not releasable and not pending:
            detail = "; ".join(
                f"{h} waiting for '{state[h].waiting_for}'" for h in blocked)
            others = {h: ("finished" if state[other[h]].progress
                          >= tp[other[h]].total_time else "also waiting")
                      for h in blocked}
            why = "; ".join(f"{other[h]} is {others[h]}" for h in blocked)
            raise RuntimeError(
                f"deadlock at t={t:.1f}s: {detail}. {why}. Nothing can "
                f"release it, so the plan has a dependency cycle or a wait "
                f"whose notify never fires.")

        done = all(state[h].progress >= tp[h].total_time and not state[h].waiting_for
                   for h in ("toolhead1", "toolhead2"))
        if done:
            break
        t += dt
    else:
        raise RuntimeError(
            f"replay ran past {deadline:.0f}s of modelled time without "
            f"finishing. Not a dependency cycle (that is detected exactly), "
            f"so this is a plan that keeps making progress but never "
            f"completes -- report it as a bug.")

    report = {"complete": True, "duration_s": round(t, 2), "collisions": collisions}
    if record_frames:
        report["frames"] = frames
    return report
