# sensors — non-camera devices

Device side only; the server side (ingest, MQTT, pages) is sensorhub.
Status as of 2026-09-03; none of these are on the camera standard yet
(decision: cameras first, sensors follow once the recipe is proven).

| Device | Board | What | Status |
|---|---|---|---|
| [`miclogger/`](miclogger/) | Seeed XIAO ESP32S3 Sense | Continuous PDM mic → WAV chunks POSTed to sensorhub `/blob/xiaomic1/audio`; pull OTA. ESP-IDF, Docker build (`make build/flash/ota-deploy`). | built; fw v20260826 |
| [`espnowbridge/`](espnowbridge/) | ESP32-S3 | Standalone ESP-NOW → WiFi repeater for the miclogger's audio, pinned to the router's channel. ESP-IDF. | built; fw v20260826 |
| [`espnowmeter/`](espnowmeter/) | Heltec ESP32-S3 | Roving ESP-NOW signal-strength meter (promiscuous sniffer, 5 s RSSI windows). ESP-IDF, bench tool. | built |
| [`plantlogger/`](plantlogger/) | Unexpected Maker FeatherS3 | Soil moisture + light + MAX17048 fuel gauge, hourly POST to the plantlog server :8087 with an offline queue. MicroPython; `HARDWARE.md` is the reference. | built; deployed |

`build/` directories are ESP-IDF outputs (ignored, ~150 MB each); regenerate
with each project's `make build`. `sdkconfig.secrets` files are gitignored
and hand-filled from `~/.dusty/`.
