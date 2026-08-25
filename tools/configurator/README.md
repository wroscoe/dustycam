# DustyCam Configurator

A single-file static webapp for weighing power vs. computation vs. optics across
candidate DustyCam setups: compute platform × camera × lens × battery × solar panel.

## Run it

No build step, no dependencies:

```
xdg-open index.html            # or just open the file in a browser
# or, if you prefer a server:
python3 -m http.server -d .    # → http://localhost:8000
```

## What it computes

For the selected setup it shows live stat tiles, a solar power-balance meter, and
three verdicts (power sustains? / optics resolve the task at distance? / compute
fast enough to track?). "Save setup to comparison" snapshots the numbers into a
table (persisted in localStorage) so setups can be compared side by side.

- **Power** — average draw at a duty cycle, ×1.25 peak for regulator sizing,
  battery autonomy at 80% usable depth, panel wattage needed at 15% average
  harvest and 85% conversion efficiency.
- **Optics** — horizontal FOV, pixels-per-meter at the target distance, and the
  max ranges for vehicle detection (≥25 px/m) and plate reading (≥150 px/m).
- **Compute** — ballpark small-model inference throughput per platform and
  whether that supports frame-to-frame tracking.

## Where the numbers come from

The math and constants are a JS port of the runnable calculators in the dustycam
repo — keep them in sync if either changes:

- `../dustycam/hardware/power/budget.py` — draws, duty-cycle budget, margins,
  solar derate, battery depth.
- `../dustycam/hardware/optics/fov.py` — sensor tables, thin-lens FOV /
  pixels-on-target math, px/m task thresholds.

All power draws and fps figures are **estimates until measured** on the real
pipeline (see the open items in `hardware/power/README.md`). The OpenMV N6 and
XIAO ESP32-S3 entries in particular are rough guesses.
