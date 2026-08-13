# esp32_s3_cam

Microcontroller-class camera: a **Waveshare ESP32-S3-CAM**. Two firmware
tracks live here — a MicroPython capture/uplink logger, and an ESP-IDF
person-detection app running a quantized TFLite-micro model on-device.

Moved in from `~/code/wavesharecam_sandbox`. Bring-up lessons (the USB-cable
trap, esptool v5 bug, board identification order) are in
[`../../docs/esp32_s3_cam_lessons.md`](../../docs/esp32_s3_cam_lessons.md) —
read that before touching a new board.

## Layout

| Path | Contents |
|---|---|
| `software/src/` | MicroPython modules deployed to board flash: `main.py` (safe-mode autostart shim), `app.py`, `boardcam.py`, `uplink.py`, `storage.py`. |
| `software/tools/` | Host-side tooling: `mp` (mpremote wrapper), `findport.py`, `monitor.py`, `server.py` (image ingest), `autolabel.py`, `train.py`. |
| `software/persondet_app/` | ESP-IDF person-detection application (`build/` excluded — regenerate with `idf.py`). |
| `software/classify/` | Host-side day-classification / fingerprinting scripts. |
| `software/server/` | `pump_server.py` — the ingest endpoint. |
| `software/Makefile` | The whole workflow; `make help` lists targets. |
| `tests/` | Host-side unit tests (no board needed). |
| `hardware/` | Empty — no board-specific design work yet. |

## Workflow

```bash
cd software
make setup            # venv + tools
make test             # host-side unit tests, no board
make dev              # deploy src/ to flash + run (the edit loop)
make server           # image collection server
make autolabel        # YOLO teacher labels dataset samples  (containerized)
make train            # fine-tune the tiny person model      (containerized)
make ota-deploy       # build detector firmware, publish for OTA
```

## Data and credentials

The training dataset is **not** in this repo — it lives at
`/hd2/datasets/wavesharecam/` (`samples/`, `labels.csv`, `model/`,
`incoming/`). The tools default to that path and honor a `DATASET_ROOT`
override:

```bash
DATASET_ROOT=/somewhere/else make train
```

Copy `software/src/secrets_example.py` to `software/src/secrets.py` and fill
it in; the ESP-IDF app reads `software/persondet_app/sdkconfig.secrets`. Both
are gitignored, as is the generated `sdkconfig`. Keep the master values in
`~/.dusty/` — see the [repo README](../../README.md).

`make deploy` copies `src/secrets.py` to flash if it exists. Note
`boot_cam.py` starts WebREPL only when `WEBREPL_PASS` is set, so an
incomplete secrets.py silently leaves WebREPL off.
