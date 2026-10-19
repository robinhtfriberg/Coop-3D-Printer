from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

import geometry
import sim


@dataclass
class Endpoint:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class MoonrakerClient:
    def __init__(self, endpoint: Endpoint, timeout: float = 10.0):
        self.endpoint = endpoint
        self.timeout = timeout

    def _get(self, path: str, **kw):
        r = requests.get(f"{self.endpoint.base_url}{path}", timeout=self.timeout, **kw)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, **kw):
        r = requests.post(f"{self.endpoint.base_url}{path}", timeout=self.timeout, **kw)
        r.raise_for_status()
        return r.json()

    def is_ready(self) -> bool:
        try:
            return self._get("/printer/info")["result"]["state"] == "ready"
        except requests.RequestException:
            return False

    def local_position(self) -> tuple[float, float, float]:
        result = self._get("/printer/objects/query", params={"toolhead": "position"})
        x, y, z, _e = result["result"]["status"]["toolhead"]["position"]
        return (x, y, z)

    def print_state(self) -> str:
        result = self._get("/printer/objects/query", params={"print_stats": "state"})
        return result["result"]["status"]["print_stats"]["state"]

    def completed_event(self) -> int:
        result = self._get("/printer/objects/query",
                            params={"gcode_macro SYNC_STATE": "completed_event"})
        return int(result["result"]["status"]["gcode_macro SYNC_STATE"]["completed_event"])

    def set_completed_event(self, value: int):
        script = f"SET_GCODE_VARIABLE MACRO=SYNC_STATE VARIABLE=completed_event VALUE={int(value)}"
        return self._post("/printer/gcode/script", params={"script": script})

    def upload_gcode(self, path: Path):
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            r = requests.post(f"{self.endpoint.base_url}/server/files/upload",
                               files=files, data={"print": "false"}, timeout=60.0)
            r.raise_for_status()
            return r.json()

    def start_print(self, filename: str):
        return self._post("/printer/print/start", params={"filename": filename})

    def pause(self):
        return self._post("/printer/print/pause")

    def resume(self):
        return self._post("/printer/print/resume")

    def run_script(self, script: str):
        """Run raw G-code. Used by step4_live_guard.py to drive the
        secondary toolhead out of the way; kept here so there is one
        Moonraker client rather than two that can drift apart."""
        return self._post("/printer/gcode/script", params={"script": script})

    def emergency_stop(self):
        return self._post("/printer/emergency_stop")


class Orchestrator:
    def __init__(self, host: str, port1: int, port2: int):
        self.clients = {
            "toolhead1": MoonrakerClient(Endpoint(host, port1)),
            "toolhead2": MoonrakerClient(Endpoint(host, port2)),
        }
        self.toolheads = geometry.TOOLHEADS

    def check_ready(self) -> bool:
        ok = True
        for name, client in self.clients.items():
            ready = client.is_ready()
            print(f"  {name}: {'ready' if ready else 'NOT READY'}")
            ok = ok and ready
        return ok

    def upload_all(self, gcode_paths: dict[str, Path]):
        for name, path in gcode_paths.items():
            print(f"Uploading {path.name} to {name}...")
            self.clients[name].upload_gcode(path)

    def world_positions(self) -> dict[str, tuple]:
        return {
            name: self.toolheads[name].to_world(*client.local_position())
            for name, client in self.clients.items()
        }

    def watchdog_check(self, clearance: float = geometry.DEFAULT_CLEARANCE) -> tuple[bool, str]:
        """One poll of the backstop safety check. Returns (collision, reason)."""
        pos = {name: client.local_position() for name, client in self.clients.items()}
        return sim.collision(self.toolheads["toolhead1"], pos["toolhead1"],
                              self.toolheads["toolhead2"], pos["toolhead2"], clearance)

    def _pause_all(self):
        for client in self.clients.values():
            try:
                client.pause()
            except requests.RequestException:
                pass

    def run(self, gcode_filenames: dict[str, str], poll_interval: float = 0.3):
        """Start both prints and relay toolhead1's completed-event counter
        into toolhead2, with the watchdog as a backstop on every poll."""
        self.clients["toolhead1"].start_print(gcode_filenames["toolhead1"])
        self.clients["toolhead2"].start_print(gcode_filenames["toolhead2"])
        relayed = 0
        failure_states = {"error", "cancelled", "shutdown"}

        while True:
            try:
                states = {name: c.print_state() for name, c in self.clients.items()}
                for name, state in states.items():
                    if state in failure_states:
                        raise RuntimeError(f"{name} stopped in unsafe state: {state}")

                hit, reason = self.watchdog_check()
                if hit:
                    self._pause_all()
                    raise RuntimeError(
                        f"safety watchdog observed an actual collision ({reason}) -- "
                        f"paused both toolheads. This means real hardware diverged from "
                        f"the offline plan (timing drift, a wrong macro, or bad geometry "
                        f"constants) -- investigate before resuming."
                    )

                completed = self.clients["toolhead1"].completed_event()
                if completed > relayed:
                    self.clients["toolhead2"].set_completed_event(completed)
                    relayed = completed

                if all(s == "complete" for s in states.values()):
                    print("Both toolheads complete.")
                    return

                time.sleep(poll_interval)
            except requests.RequestException as exc:
                self._pause_all()
                raise RuntimeError("Moonraker communication failed -- paused both toolheads") from exc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="raspberrypi.local")
    ap.add_argument("--port1", type=int, default=7125)
    ap.add_argument("--port2", type=int, default=7126)
    ap.add_argument("--gcode1", type=Path, required=True)
    ap.add_argument("--gcode2", type=Path, required=True)
    args = ap.parse_args()

    orch = Orchestrator(args.host, args.port1, args.port2)
    print("Checking both instances are ready...")
    if not orch.check_ready():
        print("ERROR: not all toolheads report ready.", file=sys.stderr)
        sys.exit(1)

    orch.upload_all({"toolhead1": args.gcode1, "toolhead2": args.gcode2})
    orch.run({"toolhead1": args.gcode1.name, "toolhead2": args.gcode2.name})


if __name__ == "__main__":
    main()
