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

The same package holds the model-building toolchain behind the `dustycam` CLI
(`pi5cam.commands`, `pi5cam.utils`):

```bash
dustycam make "A camera model that recognizes Wyoming big game at 5 fps on a Pi 5."
```

Install from the repo root, which holds the `pyproject.toml`:

```bash
pip install -e ".[pi]"    # base + picamera2
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
