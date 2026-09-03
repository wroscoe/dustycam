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

## Outcome (2026-09-01)
- Flashed `heltec_v4_companion_radio_usb-v1.17.0-merged.bin` (sha256 94abbf9e…, identical to
  flasher.meshcore.io) with `ESPTOOL_STUB_VERSION=2 uvx esptool --port <by-id> erase-flash`
  then `write-flash 0x0 <bin> --after watchdog-reset`. Wrote + verified in 6 s, no chunking needed.
- Board then vanished from USB for good on the hub port (1-1.1.4): descriptor read -110 / -71,
  RST didn't help, OLED showed MeshCore so the app was running. **Fix: rear motherboard port,
  no hub** — enumerated as 303a:1001 immediately (firmware uses HW USB-Serial/JTAG, same PID).
- Config: `uvx meshcore-cli -s <by-id> set radio 910.525,62.5,7,5 set name CornsnowBase` (renamed from wroscoe-v4 same day).
- meshbase.service repointed at the by-id path; base station connected, publishes home/mesh/#.
- The GitHub issue #2734 / PR #3006 (ARDUINO_USB_MODE=0 on heltec_v4) is already fixed in v1.17.0 —
  not the cause here.
- Follow-up: the same board on a **powered** hub (Realtek 0bda:5411, one level deep, bus 3)
  enumerated within seconds → the failing path was the Genesys hub with its power adapter NOT plugged in → 05e3:0610 → Terminus
  1a40:0101 chain (100 mA budget, both falsely claim self-powered). No battery attached.
  Sarg lesson recorded (private): heltec-v4-running-meshcore-companion-radio-usb-never.
