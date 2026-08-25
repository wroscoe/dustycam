# wavecam (Waveshare ESP32-S3-CAM) — firmware archive

The basement pump listener started as this board: an ESP32-S3-CAM running
MicroPython, streaming continuous mic PCM to `POST :8090/ingest`. It has
been quiet since ~2026-08-11; the live audio source is now **xiaomic1**
(XIAO S3 Sense, `firmware/miclogger/`) feeding the same server through
`pump/xiaomic_pumpbridge.py`. The pump server still accepts this board's
ingest and OTA endpoints unchanged, so it can be dropped back in.

## What's here

| file | what it is |
|---|---|
| `micropython_camera_firmware.bin` | MicroPython + camera build flashed to the board |
| `factory_firmware_restore.bin` | 16 MB dump of the as-shipped flash — the only copy, not re-downloadable |
| `person_detection.bin` | built ESP-IDF person-detect app (the experiment that was abandoned) |
| `model/person_detect.tflite`, `model/*.cc` | the tflite-micro model that app embedded |
| `wavecam_stream.py` | the on-device streaming script |

Not archived, because it is all reproducible: the ESP-IDF build trees
(`persondet_app/build`, `managed_components/`) and the upstream
`esp-tflite-micro` component that `persondet_build/` was a checkout of.

## OTA (the only safe update path)

The board pulls new code from the pump server, one shot per flag:

1. write the new script to `/hd2/pumpaudio/fw_stage.py`
2. `touch /hd2/pumpaudio/update.flag`
3. the board's next `GET /fw` gets the file, and the flag is consumed

**Never use WebREPL** for this board — it wedges. If USB enumeration is
wedged, recover with `raw_deploy.py` (in the dustycam tree, below).

## Related trees

- `~/code/dusty/dustycam/cameras/esp32_s3_cam/software/` holds this board's
  working tooling: `raw_deploy.py`, `deploy_out.py`, `boot_cam.py`,
  `wavecam_main.py`. It also carries **copies of `pump_server.py` and
  `classify/`** — as of 2026-08-24 the sensorhub copies here are the live
  ones and dustycam's are a stale mirror. Edit them here, not there.
- The original `wavesharecam_sandbox/LESSONS.md` (34 lessons, referenced by
  `../../LESSONS.md`) was lost before this merge; no copy exists on this
  machine. `../../LESSONS.md` is the surviving companion.
