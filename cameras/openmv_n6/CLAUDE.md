# openmv_n6 — working notes

The README is the reference; this file is what a session on this board
needs first.

- **Ask sarg first** (`sarg ask "OpenMV N6 <symptom>"`): the fw 5.x API
  differences and the CSI hang list are recorded there.
- **Bench:** USB serial `/dev/serial/by-id/usb-MicroPython_Pyboard_Virtual_Comm_Port_in_FS_Mode_31001c00025043364d343000-if00`
  (bare `/dev/ttyACM*` numbering moves — never hardcode it), mass storage
  `/media/wroscoe/PYBFLASH`. `fuser -v` the port before blaming the board;
  `udisksctl unmount` the drive before anything that resets it. The board
  re-enumerates on every reset, so a serial reader must reopen with retries.
- **Ship code:** bump `APP_VERSION` in `software/app/board.py`, run
  `tools/dustygen cameras/openmv_n6 --stage`, `POST /refresh`; `/status`
  shows the new version within ~10 s and `fw_pending` clears after the first
  upload. A crash at import rolls back and blacklists — a wrong build costs
  one boot, not a USB session.
- **Change tuning:** edit `~/.dusty/config.toml` `[camera.openmv_n6]`, run
  `tools/dustygen cameras/openmv_n6 --no-bundle`, `POST /refresh`. Keys
  must exist in `camera.toml [tuning]` (firmware first, then config).
- **Watch the stream with a socket client, not curl:** `curl -N` on this
  host drained the socket so slowly the board hit write timeouts
  (`viewer dropped OSError(110)`) and the stream looked broken at 1 fps; a
  plain Python socket reader gets ~33 fps / 600 kB/s.
- The interrupted app (`mpremote exec`, REPL) must be followed by
  `mpremote reset` so the loader starts the app again.
- `batt_v` uses an inferred divider (1.5). First thing with a meter: read
  the pack, compare, fix `BATT_DIVIDER` in `board.py`.
