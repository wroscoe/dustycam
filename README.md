# DustyCam

Open source AI camera to extract quantified information. 


This project is focused on making it very easy to create special purpose AI cameras. Everything from choosing the hardware, creating the AI model, communicating the results, to deploying the camera. All the camera software is open source and available on GitHub.

The project aims to provide reasonable defaults to accomplish the users goals while also allowing for customization and extension.


## The One Shot Workflow — the goal, not yet a command

The intended end state: describe what the camera should detect, and the
project handles the rest.

1. Define what to detect (wildlife, people, vehicles, license plates, …).
2. Generate training data for those subjects and scenes.
3. Finetune a model on the generated dataset.
4. Define and test a pipeline for the camera — when to shoot, how to
   process, where to store.
5. Quantize the model for the target board (Raspberry Pi, or an AI-capable
   microcontroller).

See [docs/one_shot_workflow.md](docs/one_shot_workflow.md) for the full
design. **There is no `dustycam` command today** — an earlier prototype CLI
was removed; steps 2-3 currently run as the per-camera tooling under
`cameras/<camera>/software/tools/` (see the ESP32-S3 teacher-student loop).


## Repository layout

| Path | Contents |
|---|---|
| [`docs/`](docs/) | All documentation: build guides, the one-shot workflow, architecture notes and plans. **Start with [`docs/camera_standard.md`](docs/camera_standard.md)** (pipeline, setup/live modes, device↔sensorhub contract, layout) and [`docs/camera_recipe.md`](docs/camera_recipe.md) (the brief for writing a new camera). |
| [`cameras/`](cameras/) | One directory per camera. Each owns its `hardware/`, `software/`, and `tests/`, a `camera.toml` manifest, and a "Standard mapping" section in its README saying how it meets the standard and where it does not. |
| [`sensors/`](sensors/) | Non-camera devices: `miclogger/` (XIAO S3 Sense continuous mic), `espnowbridge/` + `espnowmeter/` (its ESP-NOW repeater and signal meter), `plantlogger/` (FeatherS3 soil sensor). Status table in [`sensors/README.md`](sensors/README.md). |
| [`mesh/`](mesh/) | LoRa / MeshCore device side: radio firmware images + hardware notes (T114, Heltec V4). |
| [`tools/`](tools/) | `dustygen` (the one config/secrets/bundle/stage generator for every camera, standard §5), `casereview/` (markup of CAD renders), `configurator/` (static webapp weighing power × compute × optics × battery). |
| [`server/`](server/) | Base station / ingest side — the thing cameras report *to*. In practice this is **sensorhub** (`~/code/sensorhub`, MQTT + blob ingest + pages UI); this dir holds only design notes. |
| [`STATUS.md`](STATUS.md) | Every device with its status and the date it was last proven. |

**Ownership rule (2026-08-25):** every *device* — camera, sensor, or mesh
radio: firmware, hardware notes, deploy scripts — lives here. Everything
*server-side* (ingest, storage, pages, OTA staging) lives in sensorhub. The
deploy scripts stage OTA firmware into `/hd2/sensorhub/firmware/`, which
sensorhub's ingest serves to the boards.

### Cameras

| Camera | Board | Software |
|---|---|---|
| [`openmv_rt1062/`](cameras/openmv_rt1062/) | OpenMV Cam RT1062 (R6) | **Live** on the shared MicroPython runtime (2.0.8-rt): motion-gated uploader, full-res 2592x1944 capture, phone setup page with focus score, config + firmware pull with rollback; the reference for the standard. Printed case in `case/` |
| [`openmv_n6/`](cameras/openmv_n6/) | OpenMV N6 | **Live** on the same runtime (2.0.10-n6) since 2026-09-03: 1280x800 frames, battery/charge telemetry, `wifi_linger_s` low-power setting |
| [`esp32_s3_cam/`](cameras/esp32_s3_cam/) | GOOUUU ESP32-S3-CAM (goouuu1) | `software/camlogger/` ESP-IDF wake-cycle firmware: thumbnail motion diff, TFLite-micro animal gate, pull config + OTA. Silent since 2026-08-18, Docker build path broken (phase 4). Waveshare-era tracks in `archive/` |
| [`pi5cam/`](cameras/pi5cam/) | Raspberry Pi 5 / Pi Zero 2 W | Linux + CPython node/pipeline prototype; the deployed unit uploads nothing. Parked until the Pi is on the bench; its README lists the bench steps in order (phase 6) |
| [`n6_speedcam/`](cameras/n6_speedcam/) | OpenMV N6 + HLK-LD2415H 24 GHz speed radar, solar | Radar-triggered capture with vehicle speed; tscircuit carrier, DFR0535 solar/LiPo power, radome enclosure (designed 2026-09-02, unbuilt) |

Each owns its `hardware/`, `software/`, and `tests/`. A new camera means a new
directory under `cameras/`, not a fork of an existing one.

Bulk data stays out of the repo: the ESP32-S3 training set lives at
`/hd2/datasets/wavesharecam/`.


## Configuration and secrets — `~/.dusty/`

Nothing sensitive lives in this repo. Settings and credentials live in a
hidden folder in your home directory:

```
~/.dusty/                 (0700)
├── config.toml           non-secret settings: server host, dataset paths,
│                         per-camera tuning knobs
└── secrets.toml   (0600) WiFi, MQTT, OTA tokens, Google API key
```

`~/.dusty/` is the master copy you maintain by hand — one place to look when
a password changes. Keep `secrets.toml` mode 0600.

**Microcontrollers cannot read `~/.dusty/`**, so each board needs its own copy
of the values it uses. These are gitignored; copy the matching `*_example.py`
and fill it in from `~/.dusty/`:

| Board file | From template | For |
|---|---|---|
| `cameras/openmv_rt1062/software/app/secrets.py` | `tools/dustygen cameras/openmv_rt1062 [--public] [--stage]` | MicroPython, USB copy to `/flash`; the same run stamps tuning, publishes the server config and bundles/stages the app (camera standard §5) |
| `cameras/openmv_n6/software/app/secrets.py` | `tools/dustygen cameras/openmv_n6 [--stage]` | same |
| `cameras/esp32_s3_cam/software/camlogger/sdkconfig.secrets` | hand-filled from `~/.dusty` (dustygen espidf: TODO, phase 4) | compiled into ESP-IDF firmware |
| `sensors/*/sdkconfig.secrets` | hand-filled from `~/.dusty` | compiled into ESP-IDF firmware |

Give each board only the credentials it uses — the ESP32 has no MQTT client,
so it has no business holding the MQTT password.

Two things to remember when a credential changes: the board copies do **not**
update themselves (re-deploy them), and the ESP-IDF `sdkconfig` is regenerated
from `sdkconfig.secrets` at build time, so it is gitignored too.

Docs are centralized in `docs/`; the only documentation that stays out of it is
the README next to each hardware design directory, which describes the files
sitting beside it.


## Build Your Own DustyCam

The Raspberry Pi 5 build is the reference camera — BOM, case CAD, carrier PCB
and the optics/power sizing math are in
[`cameras/pi5cam/hardware/`](cameras/pi5cam/hardware/), and the step-by-step
walkthrough is [`docs/pi5_build_guide.md`](docs/pi5_build_guide.md).


## Tests on your computer

The repo root is not a Python package. The MicroPython cameras test on the
host with board stubs and need only pytest:

```bash
python -m pytest cameras/common/tests cameras/openmv_rt1062/tests cameras/openmv_n6/tests
```

The Pi camera is its own package (`cameras/pi5cam/pyproject.toml`):

```bash
pip install -e cameras/pi5cam          # add [pi] on the Pi for picamera2
python -m pytest cameras/pi5cam/tests   # Pi-only tests skip off the Pi
```


## Tips


* copy files from your pi to your computer: `rsync -avz dusty@dusty.local:~/dustycam_images ~/dustycam_images`
