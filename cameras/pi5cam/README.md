# pi5cam

A Raspberry Pi 5 (or Pi Zero 2 W) camera running the DustyCam node/pipeline
runtime, with a CSI camera on the ribbon and optional battery + solar. Full
CPython and OpenCV on-device; models run as TFLite.

## Layout

| Path | Contents |
|---|---|
| [`hardware/`](hardware/) | Case CAD, carrier PCB, and the optics/power sizing math. |
| [`software/`](software/) | The `pi5cam` Python package and the Pi provisioning scripts. |
| [`tests/`](tests/) | Test suite. Pi-only tests auto-skip when picamera2 is missing. |

Build guides and workflow docs live in [`../../docs/`](../../docs/).

## Software

The runtime is a node graph — sources → detectors → processors → sinks — driven
by a `Runner` or a `PipelineManager`:

```python
from pi5cam import Runner
from pi5cam.nodes.sources import create_source
from pi5cam.nodes.detectors.yolo import YoloNode
from pi5cam.nodes.sinks.web import WebSink
```

`create_source()` auto-selects Picamera2 on a Pi and OpenCV elsewhere, so the
same graph runs on a desktop for development.

Install from this directory (it holds the `pyproject.toml`; the repo root is
not a Python package):

```bash
pip install -e "cameras/pi5cam[pi]"    # from the repo root: base + picamera2
```

Or provision a fresh Pi end to end:

```bash
./software/install-on-pi.sh          # apt deps, venv in /opt/dusty, clone, install
./software/install-service.sh        # dusty.service (motion capture) via systemd
make -C software status              # start / stop / restart / logs / status
```

## Tests

```bash
pytest cameras/pi5cam/tests
```

## Standard mapping (docs/camera_standard.md, 2026-09-02)

**Decision 2026-09-02: leave as is until the Pi is on the bench.** Nothing
here can be verified without the device plugged in, and the deployed unit
must not be touched blind. When the Pi is plugged in, do this, in order:

1. `ssh dusty@dusty.local`, `systemctl status dusty`, `journalctl -u dusty -n 200`.
   Expect a crash loop: `tests/test_motion.py` references `images_folder_path`
   outside `main()` (NameError on the first motion event) and `Restart=always`
   restarts it. Confirm before changing anything.
2. Give the service a real entry point (`software/app/` per the standard) instead
   of a file in `tests/`; keep `test_motion.py`'s lores/still switching as the
   Watch/Capture pair.
3. Implement the standard in `cameras/common/cpython/dusty/`: Deliver to the blob
   gate with the standard meta, Report telemetry, Serve `/status` + config pull
   + git-based update, Record with a spool on the SD card, setup mode = the
   existing `WebSink` MJPEG preview plus the focus score, on `:8266`, with the
   dashboard's unauthenticated `/api/restart` removed.
4. Detection: `YoloNode._load_tflite_model()` is a stub that raises; the TFLite
   exports that used to sit at the repo root now live on homegpu at
   `/hd2/models/dustycam/yolov8n/` (`yolov8n_saved_model/*.tflite` plus the
   calibration `.npy`; `scripts/export_yolo.py` regenerates them). Copy what
   the Pi needs with `scp`/`rsync`; models stay out of git. Implement or
   mark Judge not applicable.
5. Move `~/.dusty` generation onto `dustygen` (an `.env` for the service);
   register the device in `devices.json` with `expect`.
6. Fix the path split (`/opt/dusty/env` venv, `~/dustycam` checkout, unit
   hardcoding `/home/dusty`) so deploy works for any user; make the two
   `platform.machine().startswith("aarch64")` checks one helper.

| Stage / feature | Today | Gap |
|---|---|---|
| Boot / Connect / Announce | systemd `Restart=always`; no config, no secrets | all |
| Sense | none | — |
| Watch | 320×240 frame diff (`test_motion.py`) | `why` |
| Capture | 4056×3040 stills ×3 to `~/dustycam_images` | one frame + meta |
| Judge | `YoloNode` (PyTorch on desktop; TFLite stub raises on Pi) | TFLite on Pi |
| Record / Deliver / Report | files on SD, rsync by hand; nothing uploaded | everything |
| Serve | web dashboard :8000 (no auth, `/api/restart` open) | control plane on :8266 |
| Setup mode | `WebSink` MJPEG + property editor; `test_cam_config.py` PyQt focus slider | setup page with focus score |
| Layout | `software/pi5cam/` node graph + scripts; production entry point is a test file | `software/app|host` |
| Tests | 2 real unit tests, 2 need `yolov8n.pt`, 2 are apps guarded by `importorskip` | separate apps from tests |
