# Device status

One line per device, updated when something is proven or breaks. The
vocabulary is `camera.toml`'s: **designed** → **built** → **live** →
**archived**. "Proven" means exercised on the real device, not read from
the code.

| Device | Dir | Status | Last proven | Notes |
|---|---|---|---|---|
| owlcam `rt1062cam` | `cameras/openmv_rt1062` | **live** 2.0.8-rt | 2026-09-03 | pull OTA + rollback, config pull, setup mode; no SD card (spool off) |
| `n6cam` | `cameras/openmv_n6` | **live** 2.0.10-n6 | 2026-09-03 | same runtime; radio cycling (`wifi_linger_s`) proven; button press and battery divider unverified; no SD card |
| `goouuu1` | `cameras/esp32_s3_cam` | built, silent since 2026-08-18 | 2026-08-18 | camlogger; Docker build path broken — phase 4 |
| pi5cam `dusty` | `cameras/pi5cam` | built, uploads nothing | — | crash-looping test file as service; phase 6, bench steps in its README |
| `speedcam1` | `cameras/n6_speedcam` | designed | — | radar parser tested on the host; carrier/case/power designed 2026-09-02, unbuilt — phase 5 |
| `xiaomic1` | `sensors/miclogger` | built | 2026-08-26 | continuous audio to sensorhub; not on the standard |
| espnowbridge | `sensors/espnowbridge` | built | 2026-08-26 | ESP-NOW repeater for the miclogger |
| espnowmeter | `sensors/espnowmeter` | built | 2026-08 | bench RSSI meter |
| plantlogger | `sensors/plantlogger` | built, deployed | 2026-07-21 | FeatherS3 soil logger → plantlog :8087 |
| CornsnowBase | `mesh/` | live | 2026-09-01 | Heltec V4 base station node (serial owned by sensorhub's meshbase.service) |
| DuckBisonRepeater | `mesh/` | live | 2026-09-01 | RAK4631 MeshCore repeater 1.17.1 |

Server side for all of them: `~/code/sensorhub` (ingest, blob gate, MQTT,
pages, OTA staging in `/hd2/sensorhub/`).

## The standard

`docs/camera_standard.md` is in force. Phases (from
`docs/camera_standard_proposal.md` §9): 0 docs, 1 server gate, 2 shared
MicroPython runtime + `tools/dustygen`, 3 N6 — **done 2026-09-03**;
4 esp32 camlogger, 5 speedcam from the recipe, 6 pi5cam — open.
