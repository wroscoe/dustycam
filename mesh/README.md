# mesh — LoRa / MeshCore devices

Device side of the MeshCore link: radio firmware images and hardware notes
for the nodes. The **server side** (the base station that owns the USB serial
port, publishes `home/mesh/#`, and backs the `/p/mesh` page) lives in
`~/code/sensorhub/basestation/`.

- `firmware/` — MeshCore companion-radio images as flashed (Heltec T114
  v1.16.0, Heltec V4 v1.17.0-merged) + the T114 datasheet. Binary images are
  kept on disk but not tracked in git (re-downloadable from MeshCore
  releases); this README is the version record.
- `PREFLIGHT-heltec-v4.md` — flash/bring-up checklist for the Heltec V4.

Current hardware: Heltec V4 (**base station node since 2026-09-01**, running
`heltec_v4_companion_radio_usb-v1.17.0-merged.bin`, node name `CornsnowBase`,
US preset 910.525 MHz / BW 62.5 / SF7 / CR5) and a Heltec T114 (spare, was
the base station until Aug 2026).

**Heltec V4 must be on a rear motherboard USB port or a powered hub** (since
2026-09-01 it lives on the powered Realtek hub, bus 3). Through the unpowered
Genesys→Terminus hub chain it never enumerates after boot (dmesg: `device
descriptor read/64, error -110` then `device not accepting address, error
-71`) even though the OLED shows MeshCore and the same cable/hub flashed it
fine. The by-id path
`/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_90:70:69:85:0F:18-if00`
is what meshbase.service uses. Serial access: always use
`/dev/serial/by-id/` paths, and check `fuser -v /dev/ttyACM*` before
debugging a "flaky" board.

## Repeater — RAK4631 "DuckBisonRepeater" (updated 2026-09-01)

RAKwireless WisCore RAK4631 (nRF52840 + SX1262) repeater, pubkey `DDC738D4…`,
US preset, TX 22 dBm, on the powered dock (by-id
`usb-RAKwireless_WisCore_RAK4631_Board_6CC4C6743BE6D7F3-if00`). Updated
1.11.0 → **1.17.1** over USB serial DFU, no bootloader change, all settings +
identity preserved:

    uvx adafruit-nrfutil --verbose dfu serial \
        --package firmware/RAK_4631_repeater-v1.17.1-d929643.zip \
        -p /dev/ttyACM1 -b 115200 --singlebank --touch 1200      # ~27 s

The zip's `softdevice_req` is 182 (= S140 6.1.1), which the stock RAK4631
bootloader already has. The OTAFIX bootloader is only for BLE OTA updates.
Console: `uvx meshcore-cli -s <by-id> -r` (commands: `ver`, `get name`,
`get radio`, `time <epoch>` …). Its clock was years off; set with
`time $(date +%s)`. Pre-update settings dump:
`repeater-DuckBisonRepeater-config-2026-09-01.txt`.
