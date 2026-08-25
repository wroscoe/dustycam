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

Current hardware: Heltec T114 (base station node — plugged into this
machine's USB when active) and a Heltec V4. Serial access: always use
`/dev/serial/by-id/` paths, and check `fuser -v /dev/ttyACM*` before
debugging a "flaky" board.
