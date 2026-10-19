# Coop-3D-Printer
A cooperative 3D printer prototype with 2 toolheads independent in x, y and z axis. They share a single bed and are controlled by 2 SKR pico V2 motherboards and a raspberry pi.

## Pipeline summary

See PIPELINE.md for actually running. In short:

```
1. step1_split_stl.py     part.stl -> chunk_toolhead1.stl, chunk_toolhead2.stl
2. Ultimaker Cura         slice each chunk -> toolhead1.gcode, toolhead2.gcode
3. step3_deconflict.py    the pair -> ready/*_coop.gcode + plan.json + report
4. step4_live_guard.py    live clearance guard, runs for the whole print
5. step5_visualise.py     matplotlib: paths + deposited material
```

Supporting modules: `geometry.py` (Source of all dimensional info),
 `gcode.py` (Cura-tolerant G-code parser),
`sim.py` (motion + collision model), `planner.py` (offline planner),
`replay.py` (verifier -- used inside `planner.py` to check its own plan,
and standalone to sanity-check a `plan.json` before it goes near
hardware), `annotate.py` (inserts the macro calls),
`orchestrator.py` (Moonraker client), `klipper/` (macros + generated
travel limits).

## Known limits 
- Need to combine add in the MACRO.cfg file to both printer.cfg files. Add in
  [include macros.cfg] at the top of config file.
  
- Footprint is partly measured, partly assumed. `MOUNT_WIDTH_X` and
  `MOUNT_DEPTH_Y` need to be more accurately measured with calipers
   (dimensions were changed last minute during assembly so just double check).
   `PARK_CLEARANCE` up to 95mm: at an 80mm-deep mount, a parked toolhead and one printing at the
  near bed edge each reach 45mm (half-depth + safety) toward each other,
  so anything less does not actually clear. Everything downstream
  including step 4's alarm thresholds is derived from these, so
  re-check the hardware and measure the park zones.

- The toolheads are modelled as boxes. A fan duct or a
  cable chain that sticks out beyond `MOUNT_*` is invisible.

- Concurrency needs the two heads to have a larger seperation
  With a seam across the middle of the bed, toolhead1's mount reaches
  40mm past its own material and toolhead2's reaches 40mm back, so the
  two conflict almost everywhere and the planner falls back to a strict
  layer-by-layer hand-off. That is safe and prints the part, but the
  time saving of a second toolhead only appears when the seam gives each
  head its own region of the bed.

- Timing model. Both the planner and replay use straight-line
  distance/feedrate timing (no acceleration/jerk/look-ahead).
  Real print timing will differ, which is why checkpoints are
  progress-indexed rather than wall-clock-scheduled but the predicted
  conflict windows the planner reasons about are still only as good as
  that timing estimate.

- Orchestrator untested against live Moonraker,
 Built against the API, not tested
 against real hardware in this environment.
