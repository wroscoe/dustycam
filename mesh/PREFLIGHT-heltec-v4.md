# Pre-flight — heltecv4sandbox (2026-08-13)

Stack searched: heltec, meshcore, esp32-s3, usb-serial-jtag, sx1262, flash/serial/usb intents.

Board on the bench enumerates as `303a:1001 Espressif USB JTAG/serial debug unit`
→ `/dev/ttyACM0` (ESP32-S3 native USB, no external UART bridge).

## Bring-up / physical layer
- **Cable first**: short known-good USB-C data cable, watch `dmesg -w` while plugging;
  any `error -71`/`error -110` = fix physical layer before trusting anything —
  `marginal-usb-cable-error-71-esptool-stalls-look-like-bricked-firmware` (working)
- **Stop ModemManager before serial work**: `sudo systemctl stop ModemManager` —
  `stop-modemmanager-before-esp32-ttyacm-serial-work` (working)
- **Identify before flashing**: `lsusb` → `esptool chip-id` → boot log →
  vendor repo for restore path —
  `protocol-first-bring-up-order-for-a-new-esp32-s3-board` (working)

## Flashing (ESP32-S3 over USB-Serial/JTAG)
- **Large writes can stall even with a good cable** (esptool issue #1155). Recipe:
  `ESPTOOL_STUB_VERSION=2`, erase whole chip first, and if a one-shot write stalls,
  64 KB-chunked writes with retries (`split -b 65536`, offset = index × 0x10000) —
  `esptool-large-flash-writes-reads-stall-on-esp32-s3-usb-serial-jtag` (working)
- **Exit download mode with `--after watchdog-reset`**, never plain reset —
  `exit-esp32-s3-download-mode-with-esptool-after-watchdog-reset` (working)
- **Don't back up flash over USB-Serial/JTAG** — reads stall; keep a restore path
  from vendor/MeshCore release binaries instead —
  `waveshare-esp32-s3-cam-factory-firmware-restore-path` (working)
- **Expect the USB identity to change after flashing** an app that reconfigures USB;
  absence of `303a:1001` afterward is not a brick —
  `esp32-s3-usb-pid-changes-303a-1001-to-303a-4001-after-flashing-micropython` (working)
- **pyserial opens can bounce the board into download mode** — open ports with
  DTR/RTS pre-set False —
  `pyserial-open-dtr-rts-reboots-esp32-s3-into-download-mode` (working)

## MeshCore specifics (from T114 session, applies to config phase)
- **Firmware boots on EU frequency default** — set the North American preset
  (910.525 MHz / BW 62.5 kHz / SF7 / CR5) before expecting traffic —
  `t114-meshcore-i2c-sensor-scans-wire1-not-primary-sda-scl` (provisional, related note)
- **`uvx meshcore-cli -s /dev/ttyACM0 …`** is the known-working CLI path for
  companion_radio_usb builds (reboot, self_telemetry, settings).

## Not covered (first time on record — capture as you go)
- Heltec WiFi LoRa 32 **V4** board itself — measured envelope + STEP for `amz-heltec-lora-v4` were kept in the parts warehouse (no longer on this machine)
- MeshCore firmware build/variant for Heltec V4, its flash offsets
- MeshCore serial companion protocol → webserver integration
