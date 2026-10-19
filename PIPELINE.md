# Cooperative print pipeline

Five stages. Each is a separate command, each writes files the next one
reads, so you can stop and inspect anything in between.

```
  part.stl
     │
     │  1. step1_split_stl.py          split along a sloped seam
     ▼
  chunk_toolhead1.stl  chunk_toolhead2.stl
     │
     │  2. Ultimaker Cura              slice each chunk
     ▼
  toolhead1.gcode  toolhead2.gcode
     │
     │  3. step3_deconflict.py         predict + remove collisions,
     ▼                                 park the secondary where needed
  *_coop.gcode  plan.json  report.txt
     │
     ├──  4. step4_live_guard.py       live clearance guard, whole print
     │
     └──  5. step5_visualise.py        paths + deposited material
```

## Install

```
pip install -r requirements.txt        # numpy, matplotlib, requests
```

## Step 1 — split the STL

```
python3 step1_split_stl.py part.stl --slope-angle 12 --center --outdir chunks/
```

| `--slope-angle` | degrees off vertical. `0` is a flat cut.|
| `--seam-y` | put the seam at a fixed Y instead of searching.
| `--center` | move the part to the middle of the shared bed first. Without it the part is split where it already sits. |

Also writes `chunks/placement.json`, recording where each chunk belongs.
Step 3 needs it to undo Cura's re-centring — keep it with the STLs.

Prints the volume balance and checks both chunks are watertight and
touching the build plate.

**`chunk_toolhead2.stl` comes out pre-mirrored.** toolhead2 is mounted
facing the other way, so its file is written in its own frame. Slice it
in Cura like any ordinary part. Do not mirror it yourself.

## Step 2 — slice both chunks in Cura

Slice each chunk on its own, and get one G-code file per chunk.

Let Cura place them wherever it wants. step 3 puts
them back using `chunks/placement.json`.
(If you skip `--placement`, step 3 will tell you the two chunks cover the
same region of the bed)

Turn adhesion (brim/raft) OFF for these chunks, a skirt works.

Two other settings still matter:

1. Set the machine's bed to match `geometry.py`** — 280 × 280 mm, origin
   at the front-left corner. Cura's *Machine Settings* →
   Build plate shape: Rectangular, Origin at center: OFF.
2. Leave layer comments on. Cura emits `;LAYER:n` by default.
 Step 3 warns if they are missing.

## Step 3 — de-conflict

```
python3 step3_deconflict.py toolhead1.gcode toolhead2.gcode \
    --placement chunks/placement.json --outdir ready/
```

Outputs into `ready/`:

| file | --- |
|---|---|
| `*_placed.gcode` | your slices, moved back into position |
| `*_coop.gcode` | the files you actually print |
| `plan.json` | the schedule, for steps 4 and 5 |
| `deconfliction_report.txt` | every change, and why |

It refuses to write files it cannot prove clean — the plan is
replayed against the same collision model afterwards, and any surviving
collision is an error.

`--clearance` sets the safety margin around each envelope (default
`geometry.DEFAULT_CLEARANCE`).

## Step 4 — live guard (untested)

Run this for the whole print, alongside it:

```
python3 step4_live_guard.py --host1 printer1.local --host2 printer2.local
```

Polls both machines, converts positions into the shared world frame, and
escalates on the measured gap:

| gap | response |
|---|---|
| ≤ warn (12 mm) | alert — console, bell, optional `--webhook` |
| ≤ retreat (7.5 mm) | pause toolhead2 and drive it to its park pose |
| ≤ critical (3.8 mm) | `M112` emergency stop on both machines |
| no reply for 3 s | retreat and stop — it cannot verify clearance, so it does not continue |

Thresholds are derived from `geometry.py`: a correctly
parked toolhead sits `parked_clearance_guarantee()` = 15 mm from one
printing at the near bed edge, so anything above that would alarm through
every normal park. Override with `--warn/--retreat/--critical`

```
python3 step4_live_guard.py --simulate ready/plan.json \
    ready/toolhead1_placed.gcode ready/toolhead2_placed.gcode
python3 step4_live_guard.py --host1 ... --host2 ... --dry-run
```

`--simulate` replays a plan through the identical decision logic with no
hardware. `--dry-run` uses real positions but never sends a command.

## Step 5 — watch it

```
python3 step5_visualise.py ready/toolhead1_placed.gcode ready/toolhead2_placed.gcode \
    --plan ready/plan.json --compare
python3 step5_visualise.py ready/toolhead1_placed.gcode ready/toolhead2_placed.gcode \
    --plan ready/plan.json --save run.mp4
```

Use the `_placed.gcode` files 
 `--compare` puts the coordinated and
uncoordinated runs side by side. `--stride N` skips frames to render
faster.