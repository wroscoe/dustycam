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

`~/.dusty/` is the master copy you maintain by hand — one place to look when
a password changes. Keep `secrets.toml` mode 0600.

**Microcontrollers cannot read `~/.dusty/`**, so each board needs its own copy
of the values it uses. These are gitignored; copy the matching `*_example.py`
and fill it in from `~/.dusty/`:

| Board file | From template | For |
|---|---|---|
| `cameras/esp32_s3_cam/software/src/secrets.py` | `secrets_example.py` | MicroPython, copied to board flash |
| `cameras/esp32_s3_cam/software/persondet_app/sdkconfig.secrets` | `sdkconfig.defaults` | compiled into ESP-IDF firmware |
| `cameras/openmv_n6/software/secrets.py` | `secrets_example.py` | MicroPython, copied to `/flash` |

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
