# DustyCam

Open source AI camera to extract quantified information. 


This project is focused on making it very easy to create special purpose AI cameras. Everything from choosing the hardware, creating the AI model, communicating the results, to deploying the camera. All the camera software is open source and available on GitHub.

The project aims to provide reasonable defaults to accomplish the users goals while also allowing for customization and extension.


## The One Shot Workflow

Wildlife camera model.
```bash
dustycam make "A camera model that recognizes typical wyoming big game animals, people, vehicles, dogs, and cats. This model should be small enough to run at 5 frames per second on a Raspberry Pi 5."
``` 

License plate detector.
```bash
dustycam make "A camera model that reads license plates from passing cars on a city street. This model should be small enough to run at 5 frames per second on a Raspberry Pi 5."
``` 

## What DustyCam does behind the scenes. 

1. Defines what the user wants the camera to detect (ie wildlife, people, vehicles, license plates, etc).

2. Generates training data based on what the user wants to detect and scene.

3. Finetunes a model on the generated dataset.

4. Defines and tests a pipeline to run on the camera. This includes logic about when to take photos, how to process them, and how to store them.

5. Quantizes the model to run on the target camera (ie. to a Raspberry Pi or AI enabled microcontroller).

See a detailed description of the workflow in the [docs](docs/one_shot_workflow.md).


## Repository layout

| Path | Contents |
|---|---|
| [`docs/`](docs/) | All documentation: build guides, the one-shot workflow, architecture notes and plans. |
| [`cameras/`](cameras/) | One directory per camera. Each owns its `hardware/`, `software/`, and `tests/`. |
| [`server/`](server/) | Base station / ingest side — the thing cameras report *to*. Not yet implemented. |
| `yolov8n_saved_model/`, `*.npy` | Project-level data: exported models and quantization calibration samples. |

### Cameras

| Camera | Board | Software |
|---|---|---|
| [`pi5cam/`](cameras/pi5cam/) | Raspberry Pi 5 / Pi Zero 2 W | Linux + CPython; the node/pipeline runtime and the `dustycam` CLI |
| [`esp32_s3_cam/`](cameras/esp32_s3_cam/) | Waveshare ESP32-S3-CAM | MicroPython logger + an ESP-IDF person-detection app (TFLite-micro) |
| [`openmv_n6/`](cameras/openmv_n6/) | OpenMV N6 | MicroPython uploader with motion gating, MQTT telemetry, WiFi OTA |

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

Create it, then fill in `secrets.toml`:

```bash
dusty init                    # write both files from templates
dusty show                    # print the merged config, secrets redacted
dusty generate --all          # write each board's credential files
```

Both files are merged into one view, with `secrets.toml` winning on conflict
and `[camera.<name>]` sections scoping settings to one camera. A
`DUSTY_<SECTION>_<KEY>` environment variable overrides both, and `DUSTY_HOME`
relocates the folder. `secrets.toml` must be mode 0600 — the loader refuses to
read it otherwise.

**Microcontrollers cannot read `~/.dusty/`**, so the files they *can* read are
generated from it by `dusty generate`:

| Generated file | For |
|---|---|
| `cameras/esp32_s3_cam/software/src/secrets.py` | MicroPython, copied to board flash |
| `cameras/esp32_s3_cam/software/persondet_app/sdkconfig.secrets` | compiled into ESP-IDF firmware |
| `cameras/openmv_n6/software/secrets.py` | MicroPython, copied to `/flash` |

All three are mode 0600 and gitignored. Treat them as build artifacts: edit
`~/.dusty/secrets.toml` and regenerate, never edit them directly. Each camera
receives only the credentials it needs — the ESP32 board has no MQTT client,
so it never gets the MQTT password.

Host-side code reads `~/.dusty/` directly (`from dusty.config import load`),
so the toolchain's `GOOGLE_API_KEY` and the dataset path come from the same
place. An existing `GOOGLE_API_KEY` environment variable still wins.

Docs are centralized in `docs/`; the only documentation that stays out of it is
the README next to each hardware design directory, which describes the files
sitting beside it.


## Build Your Own DustyCam

The Raspberry Pi 5 build is the reference camera — BOM, case CAD, carrier PCB
and the optics/power sizing math are in
[`cameras/pi5cam/hardware/`](cameras/pi5cam/hardware/), and the step-by-step
walkthrough is [`docs/pi5_build_guide.md`](docs/pi5_build_guide.md).


## Test software on your computer.
For desktop development, install only base deps (no Pi extras). The Pi-only tests will auto-skip when the dependency is missing.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .  # no [pi] extra
pip install pytest

pytest cameras/pi5cam/tests
```


## Tips


* copy files from your pi to your computer: `rsync -avz dusty@dusty.local:~/dustycam_images ~/dustycam_images`
