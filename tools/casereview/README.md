# casereview

Browse the camera-enclosure renders and pin comments directly onto them.

Click anywhere on a drawing to drop a numbered pin and type a note. Everything
you mark up lands in `comments.json` beside this file — which is the point of
the tool: that file lives in the repo next to the CAD source, versions with it,
and Claude reads it directly to see what you flagged and where.

## Run

```bash
python3 tools/casereview/serve.py                 # http://localhost:8102
python3 tools/casereview/serve.py --port 8103     # 8102 taken
python3 tools/casereview/serve.py --host 0.0.0.0  # reachable on the tailnet
```

**Stdlib only — nothing to install.** No Flask, no venv, no `pip`. It is
`http.server` plus one HTML file, which is the whole reason it can sit in the
repo and still run in five years.

## What you see

The sidebar lists every part, grouped by design, with a badge counting open
notes. Each part has six views, labelled by what the face means **once the case
is stood up in its deployed pose** — so "FLOOR" is the wall the USB ports open
through, not whichever face happens to be down in the CAD frame. That mapping
differs per board (GOOUUU stands on board -X, Waveshare on -Y), which is exactly
the kind of thing that is easy to get backwards when reviewing.

Notes can be resolved rather than deleted — resolved ones grey out and hide
until you tick "show resolved", so a review pass leaves a record of what was
raised and settled, not just what is still outstanding.

## Adding designs

Renders are discovered, not configured. Anything matching

```
cameras/<camera>/hardware/case/renders/<target>.<view>.png
```

appears automatically. Regenerate the set with the render loop in
`cameras/esp32_s3_cam/hardware/case/README.md`; the OpenMV N6 case will show up
here on its own once it has renders.

## comments.json

```json
{"target": "goouuu_cam_case_body", "view": "bottom",
 "x": 0.5, "y": 0.33, "text": "...", "status": "open", "n": 1}
```

`x`/`y` are fractions of the image, so pins survive re-rendering at a different
resolution — but **not** a change of camera angle. If you change a view's
camera, its pins will point at the wrong place; re-render with the same angles
or move the pins.

Safe to hand-edit or delete. Writes are atomic (temp file + rename), so a crash
mid-save cannot truncate it, and a malformed file is copied aside to
`comments.json.bad` rather than silently discarded.
