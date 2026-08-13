# Waveshare ESP32-S3-CAM-OVxxxx — capture an image over USB (lessons learned)

## Next new board: do it in this order (15 min, not hours)

What made the first time slow was (a) a marginal USB cable producing two
unrelated-looking failures, (b) a real esptool v5 bug stacked on top of it,
and (c) discovering documented facts too late. The order below front-loads
the cheap checks:

1. **Cable first.** Short, known-good USB-C *data* cable, direct rear
   motherboard port, no hubs. Before anything else: `sudo dmesg -w` in a
   terminal while plugging in — any `device descriptor read, error -71` /
   `error -110` means fix the physical layer NOW; nothing software-side will
   be trustworthy until it's clean. `sudo systemctl stop ModemManager`.
2. **Identify before flashing** (5 min): `lsusb` for the USB identity;
   `esptool chip-id` for chip/PSRAM type (octal vs quad decides the firmware
   variant!); read the boot log over serial (open port with DTR/RTS pre-set
   False) to see what the stock firmware says; find the vendor's GitHub repo —
   it usually contains the factory firmware (your restore path — skip trying
   to back up flash over USB-Serial/JTAG, reads stall) and the real pin map
   in demo code. Don't trust "compatible with board X" claims — diff the pins.
3. **Research early, not late.** Fire a web search for
   "<board> esptool site:github.com issues" and the vendor wiki *before* the
   first flash attempt. Every blocker we hit (esptool #1155 stub regression,
   watchdog-reset requirement, sensor capabilities, pin map) was already
   documented somewhere.
4. **Flashing recipe that works:** `ESPTOOL_STUB_VERSION=2`; erase whole chip
   first (fast, and an empty chip always falls back to download mode — no
   more BOOT-button dances); if a one-shot write stalls, go straight to
   64 KB-chunked writes with retries (`split -b 65536`, one esptool session
   per chunk, offset = index × 0x10000). Exit download mode with
   `--after watchdog-reset`, never a plain reset.
5. **Expect the USB identity to change after flashing MicroPython**
   (ROM `303a:1001` → app `303a:4001`). Absence of the old device is not a
   brick; absence of *any* device after a clean cable is (then: dmesg).
6. **Capture:** adapt `capture.py` (this dir) — it already encodes every
   serial-protocol trap (DTR/RTS-safe open, soft-reboot-before-raw-REPL,
   chunked code upload, blocking-print avoidance, offset-indexed transfer,
   stdlib PNG). Only the pin map and sensor format should need changing.

Everything below was learned the hard way on 2026-08-07 getting a first image
out of this board on the Linux workstation (`homegpu`). Following the Quick
Path should make it work the first time.

## The board (this specific unit)

- **Waveshare ESP32-S3-CAM-OVxxxx** — ESP32-S3 (QFN56 rev v0.2), **8 MB octal
  PSRAM** (AP_3v3, embedded), **16 MB QIO flash**, MAC `a4:cb:8f:d7:81:90`.
- **Camera sensor: GC0308** (identified via probe). Max 640×480 (VGA).
  **No JPEG output** — RGB565 / YUV422 / grayscale only. This is why the stock
  firmware printed `JPEG format is not supported on this sensor` and crashed:
  Waveshare ships one demo binary for four possible sensors (OV5640, OV3660,
  GC2145, GC0308) and parts of it assume JPEG.
- It is **not a USB webcam** out of the box: it enumerates as a serial device.
  (The vendor repo's `06_usb_host_uvc` ESP-IDF example — actually a UVC
  *device* demo — could make it one, but **only for OV5640/OV3660**, not our
  GC0308.)
- Camera pin map (from Waveshare's own demo code; they hijacked the
  `CAMERA_MODEL_ESP_EYE` slot but the pins are NOT the real ESP32-S3-EYE map):
  `XCLK=38 SIOD=8 SIOC=7 VSYNC=17 HREF=18 PCLK=41 D0–D7=45,47,48,46,42,40,39,21`
  (PWDN/RESET not connected).
- Buttons: **BOOT** and **PWR** only — **no reset button**. To force the ROM
  bootloader: hold BOOT while plugging in.
- USB identities you will see:
  - `303a:1001` "USB JTAG/serial debug unit" → ROM / stock IDF firmware
  - `303a:4001` "Espressif Device" → MicroPython's own USB-CDC (TinyUSB)

## Hardware / host gotchas (these cost the most time)

1. **The USB cable is everything.** With a marginal USB-C cable the board
   enumerates and chats fine, but: sustained esptool transfers die mid-stream
   (`Serial data stream stopped`, `Packet content transfer stopped`), and
   MicroPython's USB-CDC **never enumerates at all** (kernel log:
   `device descriptor read, error -71`, `error -110`,
   `Device not responding to setup address`). It looks exactly like bricked
   firmware. Use a short, known-good data cable, plugged **directly into a
   rear motherboard port — no hubs**. `sudo dmesg` is the diagnostic:
   -71/-110 enumeration errors = electrical, not software.
2. **Stop ModemManager** before serial work: `sudo systemctl stop ModemManager`
   (it probes fresh ttyACM devices mid-session). *It was left stopped; it
   returns on reboot or `sudo systemctl start ModemManager`.*
3. **pyserial opens reset the chip.** Opening `/dev/ttyACM0` naively asserts
   DTR/RTS, which the USB-JTAG interface maps to EN/BOOT — you can reboot the
   chip into download mode just by opening the port. Always pre-set lines
   before opening:
   ```python
   s = serial.Serial(); s.port='/dev/ttyACM0'; s.baudrate=115200
   s.dtr = False; s.rts = False   # BEFORE open()
   s.open()
   ```

## esptool gotchas (esptool v5.3.1)

4. **Large single writes/reads stall** on this board/host even with a good
   cable (~known ESP32-S3 USB-Serial/JTAG stub regression, esptool issue
   #1155). `ESPTOOL_STUB_VERSION=2` fixed *erase* (16 MB in 6 s) but not
   sustained writes. **Reliable recipe: write in 64 KB chunks, one esptool
   session per chunk, retry failures** (~10% of chunks fail transiently;
   each write is hash-verified by esptool, so a 0 exit = that chunk is good):
   ```bash
   split -b 65536 -d -a 3 --additional-suffix=.bin firmware.bin part_
   # part N goes to offset N*0x10000:
   esptool --port /dev/ttyACM0 --after no-reset write-flash 0x0 part_000.bin
   esptool --port /dev/ttyACM0 --after no-reset write-flash 0x10000 part_001.bin
   # ... retry any nonzero exits; repeat until all 0
   ```
5. **Exiting download mode:** if the chip entered the bootloader manually
   (BOOT held) — or you're unsure — a plain esptool reset will NOT boot the
   app. Use:
   ```bash
   esptool --port /dev/ttyACM0 --no-stub --before no-reset --after watchdog-reset chip-id
   ```
6. **Entering download mode from MicroPython** (no buttons needed):
   `import machine; machine.bootloader()`. Quirk: the CDC device may linger as
   a zombie on the host for a bit; it clears itself (or
   `USBDEVFS_RESET` ioctl with sudo). Then flash with `--before no-reset`.

## Firmware choice

7. Use **micropython-camera-API prebuilt** (cnadler86), release v0.6.2,
   **`ESP32_GENERIC_S3-SPIRAM_OCT`** zip (MicroPython v1.27 + esp32-camera +
   mp_jpeg). Octal-PSRAM variant is mandatory (this chip is S3R8).
   - Do **NOT** use the `ESP32S3_EYE` zip: its baked-in default pins are the
     real S3-EYE's, which don't match this board. The generic build with
     explicit pins is correct.
   - Flash at offset **0x0** (chunked, per #4).
   - Local copies: `micropython_camera_firmware.bin` (this dir).
8. **Restore path:** Waveshare's factory image is
   `factory_firmware_restore.bin` (this dir; from
   `github.com/waveshareteam/ESP32-S3-CAM-OVxxxx` → `Firmware/`,
   full 16 MB image, flash at 0x0). No device backup needed — full flash
   *reads* stall anyway (see #4; chunked 1 MB reads also failed pre-cable-fix).

## Capturing over the REPL

9. After boot MicroPython shows up as `303a:4001` → `/dev/ttyACM0`
   (~1–2 s after reset; the 1001 device disappearing and 4001 appearing is
   *normal*, not a brick).
10. Camera init (raw REPL, or mpremote once installed):
    ```python
    from camera import Camera, PixelFormat, FrameSize
    cam = Camera(
        data_pins=[45,47,48,46,42,40,39,21],
        vsync_pin=17, href_pin=18, sda_pin=8, scl_pin=7,
        pclk_pin=41, xclk_pin=38, xclk_freq=20000000,
        pixel_format=PixelFormat.RGB565,   # GC0308: no JPEG!
        frame_size=FrameSize.VGA,          # sensor max
        init=False)
    cam.init()
    for _ in range(4): img = cam.capture()   # warm-up for AE/AWB
    # len(img) == 614400 == 640*480*2  (RGB565)
    ```
11. Pull the frame as base64 lines over the REPL; **decode each line
    separately** (each `b2a_base64` chunk carries its own padding —
    concatenating the text then decoding silently truncates at the first
    `==`).
12. **Keep the host reading the port the whole time.** If the host stops
    reading, the device's CDC buffer fills and its `print()` blocks forever —
    the board looks hung but is just waiting.
13. **Soft-reboot (Ctrl-D) before re-initializing the camera.** Re-running
    `Camera(...)` in a dirty session hard-hangs the interpreter (wedged
    SCCB/I²C) — REPL dead, only a power cycle (replug) recovers it. This is
    the one failure that needs hands on the hardware, so avoid it.
14. RGB565 from esp32-camera is big-endian per pixel; if colors are wrong
    on conversion, swap bytes before unpacking (`>u2` vs `<u2` in numpy).

15. **Raw REPL is unreliable on the first session after a boot** — the first
    exec attempt reliably fails with `SyntaxError: invalid syntax` at line 1.
    Fix: do a soft reboot first (Ctrl-B, Ctrl-D, wait ~2 s), then enter raw
    REPL. Also feed the code in ≤128-byte pieces with small delays (the CDC
    link has no flow control on code upload).
16. **Expect a few lost 4 KB chunks per ~600 KB serial transfer** (~2–3%).
    Prefix every chunk with its byte offset (`print(str(i)+':'+b64)`) and
    reassemble by offset on the host — losses become thin black stripes
    instead of shearing the whole image.

## Dev-workflow lessons (the wedge family, discovered building the make flow)

Three ways to hard-hang this board so only a physical replug recovers it —
the dev tooling is designed so none can happen:

17. **Never deploy camera code as `main.py`.** It autostarts at boot with no
    host reading the CDC and wedges before any tool can connect — and does it
    again every boot. App code is `app.py`, run explicitly. (Escape hatch if
    it ever happens: BOOT-held replug → `esptool erase-region 0x200000
    0x100000` wipes the littlefs filesystem; MicroPython reformats on boot.)
18. **Never use `mpremote mount`.** Mounted imports ride the lossy CDC; a
    stall mid-import strands the interpreter with the camera live. Code is
    always copied to flash (`mpremote cp` — short, retryable transfers).
19. **Hard-reset (machine.reset / `mpremote reset`) before every run.**
    Clears wedged sensor state from any crashed prior run; soft reset does
    not. This is what makes `make run` self-recovering after crashes.
20. **The CDC needs ~5 s of quiet between tool sessions** or raw-REPL entry
    fails ("could not enter raw repl"); `tools/mp` wraps mpremote with
    retry+pause. And mpremote's `exec` can hang forever when the CDC drops
    the end-of-program marker — `tools/run_app.py` streams with our
    loss-tolerant transport (sentinel + idle-timeout) instead.

## Status / working end-to-end flows

**Photo capture:** `capture.py` — soft-reboot → raw REPL → camera init with
the Waveshare pins → warm-up frames → VGA RGB565 → offset-indexed base64 over
serial → pure-stdlib PNG. ~80 s (transfer-bound). First image:
`first_capture.png` (2026-08-08).

**Dev workflow (proven 2026-08-08, incl. the brightness→"LED" demo):**

```
make test    # host pytest on pure logic (src/brightness.py) — no board
make run     # deploy src/ to flash → hard reset → run app.py, stream output
make repl    # interactive REPL
```

- Layout: `src/brightness.py` (pure logic, host-testable), `src/boardcam.py`
  (camera singleton + pins + release()), `src/app.py` (demo loop,
  try/finally release; NEVER named main.py — lesson #17).
- Plumbing that makes it reliable: `tools/mp` (mpremote + retry/pause,
  lesson #20) and `tools/run_app.py` (loss-tolerant output streaming with
  APP_DONE sentinel + idle-timeout, soft-reboot-first, 3 attempts).
- Acceptance-tested: repeated runs, and a run killed at a random moment is
  fully recovered by the next `make run` (hard-reset-first, lesson #19).
  No remaining failure mode needs a replug.
- Demo verified live: covering the lens → luma 10 → `DARK ** LED ON **`.

ModemManager is still stopped on the host (see #2).

## Person detector (ESP-IDF + TFLite Micro) — built & running 2026-08-08

The esp-tflite-micro `person_detection` example runs on this board:
continuous ~several-fps person/no-person inference on 96×96 grayscale frames,
scores on the USB-Serial/JTAG console (`tools/monitor.py`). Project lives in
`persondet_app/` (adapted copy) + `persondet_build/` (esp-tflite-micro clone).

Build lessons:

21. **Build hermetically in Docker** — `espressif/idf:release-v5.5` image,
    no host installs: `docker run --rm -v $PWD:/work -w /work/persondet_app
    espressif/idf:release-v5.5 bash -c "idf.py build"`. (Anything the
    container writes is root-owned — rm/clean via docker too.)
22. **The example can't build in-tree.** IDF fails to resolve the
    `override_path: ../../../` managed dependency ("Failed to resolve
    component ... unknown name", IDF 5.3 and 5.5). Fix: copy the example out
    of the repo, drop the managed dep from `main/idf_component.yml`, symlink
    the repo as `components/esp-tflite-micro`, add it to main's
    PRIV_REQUIRES.
23. **The example's Kconfig pin ranges are written for the original ESP32**
    (`range 0 33` / `0 39`) and *silently* discard S3 GPIOs 38-48 — the
    build succeeds with default (WROVER) pins mixed in. Widen to
    `range -1 48` in `main/Kconfig.projbuild` and verify EVERY
    `CONFIG_CAMERA_PIN_*` in the generated sdkconfig before flashing.
24. Also: base `sdkconfig.defaults` hardcodes `CAMERA_MODULE_ESP_EYE=y` —
    remove it or it fights the target-specific CUSTOM choice; set
    `CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y` or scores go to UART0 pins;
    XCLK_FREQ_HZ in app_camera_esp.h 15→20 MHz (proven with this sensor).
25. **An IDF app with wrong camera pins wedges the chip** so hard that even
    esptool's USB-JTAG reset fails — BOOT-held replug required. Verify pins
    (#23) before flashing to avoid the trip.
26. **The tty node can hop** (ttyACM0→ttyACM1) when re-enumeration races the
    old node's release. All tooling uses the stable
    `/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_...` path.

Flash layout (chunked recipe per #4): bootloader @0x0, partition table
@0xc000, ota_data @0x13000, `person_detection.bin` @0x20000.
Restore MicroPython: flash `micropython_camera_firmware.bin` @0x0.

## Sample pipeline + the noise discovery (2026-08-08)

The detector now uploads every (96×96 frame, person/no-person score) pair to
the server (`persondet_app/main/uploader.c` → `tools/server.py /sample` →
`dataset/samples/*.png` + matching `.json`). First real catch:

27. **THE big one — never read a TFLite Micro input tensor after Invoke().**
    The memory planner reuses the input tensor's arena as scratch during
    inference, so post-Invoke reads return model activations, not your
    image. Uploading `input->data.int8` after inference produced
    light-correlated "noise" that perfectly impersonated a broken camera
    driver and burned five debugging iterations (framesize theories, YUV
    byte-order theories, driver bump to 2.1.7, cache-config theories — all
    red herrings chasing this one bug). Snapshot the input with memcpy
    BEFORE Invoke if you need it afterward.
    - Corollary: the "GC0308 96x96/QQVGA modes are broken" claims from that
      debugging session are UNVERIFIED — the evidence was contaminated.
      What IS proven working now: QVGA RGB565 + software luma extraction +
      center-crop 192² + 2×2 average → 96×96 (and the incidental cleanups:
      zero-initialized camera_config_t — real bug, the example leaves
      grab_mode/conv_mode as stack garbage — and esp32-camera 2.1.7).
28. **The model scored the garbage as "person 79–94%"** — a detector can
    look perfectly healthy on the console while its input is trash. Never
    trust on-device confidence until you've LOOKED at what the model sees;
    the image+JSON sample pipeline is the tool for that.
    Noise-vs-dark-scene test (limited: can't distinguish a broken pipeline
    from a genuinely dark room): mean |neighbor diff| ≈ |random-pair diff|
    ⇒ uncorrelated; a lit-scene test is definitive.
29. **Secrets→sdkconfig generation**: greedy regex `'(.*)'` ate an
    apostrophe in a trailing comment and shipped garbage as the server
    host (uploads silently failed). Use non-greedy + verify the generated
    `sdkconfig.secrets` values.
30. The zombie CDC device (after `machine.bootloader()`) **self-clears in
    ~4–5 min** — "No such device" from the USBDEVFS_RESET ioctl means it
    was already gone. Patience is a valid strategy; poll for 303a:1001.

## OTA updates (WiFi flashing) — working 2026-08-08

`make ota-deploy` = Docker build + publish; the board self-updates within
~30 s and reboots. Proved itself immediately: five firmware iterations in
~40 minutes during the tensor-bug hunt, zero cables touched.

- Server: `GET /firmware/version` (parsed from esp_app_desc_t at offset 32
  of the .bin) + `GET /firmware.bin`; board's ota task polls every 30 s,
  runs esp_https_ota on mismatch (`persondet_app/main/uploader.c`).
- Version = `persondet_app/version.txt` (auto-stamped by ota-deploy).
- Anti-brick: `CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE=y`; the new image
  calls `esp_ota_mark_app_valid_cancel_rollback()` only after its first
  successful sample upload. **Do NOT open the serial port while an update
  is pending-verify** — attaching the monitor resets the chip (open →
  DTR/RTS glitch), and a reset before mark-valid rolls the update back.
  Observe OTA cycles via the server log only.
- Limits: OTA covers IDF-app slots only (MicroPython ↔ IDF swaps stay
  USB); a wedged board can't OTA.

## Fine-tuning loop (teacher-student) — closed 2026-08-09

Full cycle proven: board collects frames+scores → `make autolabel` (YOLOv8s
teacher in a container; 4127 person / 7636 no-person from night one; stock
model agreed with teacher only 37%) → `make train` (MobileNetV1-0.25 96×96
int8, ~83% val acc, emits drop-in `person_detect_model_data.cc`) →
`make ota-deploy`. Lessons:

31. **Register every op the converter emits, not the ops you think you
    used.** A Keras `Dense` = FULLY_CONNECTED, and `Flatten` sneaks in
    SHAPE + STRIDED_SLICE + PACK (dynamic shape computation). Missing ops
    crash at model load → OTA crash-loop: old slot boots → downloads →
    new slot crashes → rollback → repeat (the board did ~1700 cycles
    overnight; rollback made it survivable, which is the point of #4-style
    safety). Diagnose by dumping ops from the .tflite with the TF Lite
    interpreter — no board needed.
32. **"Intermittent grainy frames" = sensor warm-up frames.** GC0308's
    first ~5 frames after init are AGC/AWB-unsettled garbage; a
    crash-looping board re-inits constantly, so they show up "every couple
    frames". Fixed: discard 5 frames at camera init (app_camera_esp.c).
33. The board once dropped OFF USB spontaneously (enumeration gone, power
    retained) and kept running on WiFi for 9+ hours — collection, OTA
    crash-loop and all. Everything except first-flash now works with no
    USB data connection. Cause of the drop unknown; replug restores it.
34. A model trained on one lighting condition (one evening) misjudges
    others (daylight) — keep collecting across conditions and retrain;
    the loop is 3 commands.

## Image logger + WiFi upload (dataset collection) — working 2026-08-08

Store-and-forward field camera: capture VGA grayscale → JPEG on device
(~21 KB) → internal flash (/imgs/pending, ~650-image buffer) → when WiFi
connects, POST to `tools/server.py` on the workstation (~2 images/s), move
to /imgs/sent. Nothing deletes until the server confirmed receipt.

- `make logger` (interactive run) · `make server` (receiver →
  dataset/incoming/) · `make deploy-autostart` (adds main.py = autonomous
  on boot; plain `make deploy` never ships main.py).
- WiFi creds in `src/secrets.py`, generated from `~/.dusty/secrets.toml` by `dusty generate esp32_s3_cam` (never committed).
- **microSD is NOT usable from stock MicroPython builds**: the slot is
  SDMMC-only (CLK=16 CMD=43 D0=44 — Waveshare BSP: "SPI mode not supported
  by HW!") and prebuilt firmware can't remap SDMMC pins (fixed defaults, no
  kwargs). Needs a custom MicroPython build to enable → until then, internal
  flash (14 MB) is the buffer.
- `jpeg.Encoder` accepts only `GRAY` and `RGB888` pixel formats — capture
  GRAYSCALE from the sensor (native on GC0308); RGB565 must be converted
  before encoding.
- The by-id serial path CHANGES with USB mode (JTAG "debug unit" vs
  MicroPython "Espressif Device") — tools resolve it at runtime via
  `tools/findport.py` (glob /dev/serial/by-id/usb-Espressif*).
- Autostart safety (src/main.py): 6 s boot grace watching the BOOT button
  (GPIO0) + `/disable.txt` kill-file → REPL instead of app; watchdog fed
  from inside the capture loop; echo=False always (prints wedge, log to
  /imgs/log.txt instead).

## Audio: dual mics verified working (2026-08-09)

Both onboard mics work. Verified with `mictest.py` (device script) run via
`mic_runner.py` (host runner reusing capture.py's DTR/RTS-safe protocol —
mpremote wedges this board, same as before).

- **Chips on I2C(1, sda=8, scl=7):** ES8311 DAC @ 0x18, CH32V003 IO expander
  @ 0x24, **ES7210 mic ADC @ 0x40** (chip-id reg 0x3D reads 0x72).
- **I2S pins (ESP32 perspective, from vendor 06_esp_sr.ino):** MCLK=10,
  SCLK=11, WS=12, DIN=13. The wiki interface map labels pins from the
  *codec's* perspective — its "I2S_DOUT GPIO13" is the ESP32's data-in.
- **This firmware's `machine.I2S` (MP v1.27 camera build) has no `mck` arg** —
  generate MCLK with LEDC PWM on GPIO10 at 256*fs (4.096 MHz nominal, actual
  4.0895 MHz, works). ES7210 runs I2S slave, 16 kHz/16-bit stereo, MIC1+MIC2,
  PGA 30 dB (init sequence ported from esp-adf es7210.c, in mictest.py).
- Result: ~7 RMS noise floor, claps hit RMS ~14 / peak ~76 on both channels
  simultaneously with distinct values — both mics live. Levels are low;
  raise PGA to 37.5 dB and/or ADC digital volume for real use.

## Serial port contention (2026-08-09)

Most of today's "flaky USB" was not the board: `~/code/openmvsandbox/fb_webui.py`
had been holding `/dev/ttyACM0` open for 2 days, silently stealing bytes from
every esptool/mpremote/pyserial session (symptoms: intermittent "could not
enter raw repl", stalled flash reads, writes that hang after open()). Before
blaming this board's USB, run `fuser -v /dev/ttyACM0`. Any long-running tool
that opens a serial port should use its stable `/dev/serial/by-id/...` path,
never bare `ttyACM0` — device numbers get reshuffled by replugs.

If the CDC endpoint is wedged (open OK, write hangs): replug the board, or
`sudo` a USBDEVFS_RESET ioctl on `/dev/bus/usb/<bus>/<dev>`.

## Wi-Fi + WebREPL (2026-08-09)

`boot.py` (deployed; copy in repo as `boot_cam.py` if re-flashing) joins the
configured Wi-Fi network, sets hostname `wavecam`, and starts WebREPL on
`ws://192.168.86.210:8266`. Credentials — the Wi-Fi SSID/password and the
WebREPL password — live in `~/.dusty/secrets.toml`, never here. Push scripts over Wi-Fi with
webrepl_cli / the browser client at http://micropython.org/webrepl — no USB
needed after boot. NOTE: a soft reset now spends up to 15 s in the Wi-Fi
connect loop before the REPL answers — serial tools with fixed timing
(mic_runner.py included) must wait that out.

## Sound recorder web app (2026-08-09)

`wavecam_main.py` (deployed to the board as `main.py`) serves a recorder UI at
http://192.168.86.210/ — record button (3/5/10/15 s) -> stereo 16 kHz WAV in
`/recs` on flash, listed with inline <audio> players + delete. Mic PGA at
34.5 dB. Deploy updates with `make_deploy2.py <file> <target> reset` +
`mic_runner.py deploy_out.py` over serial, or WebREPL over Wi-Fi. The server
polls accept() with a 2 s timeout so ctrl-C/WebREPL can still interrupt it.

## Continuous pump listener (2026-08-09)

Replaced the button-recorder: the board now streams continuously.
- **Board** (`wavecam_stream.py` = device `main.py`): 8 kHz mono (ES7210 8k
  coeffs: reg02=0x41, lrck=0x0200, MCLK still 4.096 MHz = 512fs via PWM),
  POSTs 5 s PCM chunks to homegpu, 25-chunk (~2 min) PSRAM ring for outages.
- **Server** (`server/pump_server.py`, systemd user unit `pumpaudio.service`,
  port 8090): appends per-minute .pcm under /hd2/pumpaudio/audio/<date>/,
  1 Hz RMS log per day (86400 u16 slots, 0xFFFF = gap), 30-day rotation.
- **UI http://192.168.86.26:8090/**: day chart, threshold slider -> pump
  on/off event list (client-side, retunable over old data), click chart or
  event to hear that minute. Chunk->wall-clock mapping is seq-based per boot
  so late retries land at the right time.

**8 kHz capture gotcha (2026-08-09):** `I2S.MONO` at 8 kHz with the ES7210
512fs coeff row captures pure zeros on this build — looked "working" as a
dead-flat RMS on the server. Fix: board always captures the proven 16 kHz
STEREO / 256fs config; the server downmixes to 8 kHz mono at ingest
(`downmix()` in pump_server.py). Rule: watch the level jitter — real audio
is never perfectly constant; a flat RMS means silence, not quiet.

**Hardened streamer (2026-08-09, deployed):** field failure taught: long
Wi-Fi outage -> requests leaks sockets -> LWIP exhausted -> main.py dies,
TCP wedges (ping OK, all connects time out), needs battery-cycle. Hardened
main.py: reconnects Wi-Fi every ~60 s while down, gc.collect() after failed
posts, machine.reset() after ~8 min unreachable and on any uncaught crash.
Also learned: WebREPL OTA cannot complete a handshake while main.py busy-loops
on a weak link — deploy over serial, or add a server-side pull-updater later.

## Basement deployment campaign (2026-08-09 afternoon)

**Wi-Fi at the pump spot is the bottleneck, not the mic.**
- Coverage collapsed 98% -> 8-25% the moment the board went downstairs, with
  ~8% = exactly one 5 s chunk/minute surviving each brief reconnect. All pump
  observations that day went through that straw.
- A human body next to the board crushes the link further (89 s walk test ->
  10 s of data). Don't judge the link while standing next to it.
- Google Wifi mesh: 4 nodes broadcast the same SSID on 2.4 GHz. ESP32 default
  connect is fast-scan (first node that answers), NOT strongest, and it never
  roams. Rebooting the AP it's stuck on (Google Home app) forces a reconnect
  but is a coin flip — coverage briefly hit ~50%, then decayed right back.
- Fix deployed: scan-before-connect in boot.py + main.py — scan, filter SSID,
  connect with bssid= of the strongest node, at boot and at every ~60 s
  reconnect. Each reconnect also POSTs an AP survey (bssid ch dBm list) to
  the server -> /hd2/pumpaudio/scans.log. Desk reference: two nodes at
  -63 dBm. Rough guide: >=-75 good, -80..-85 gappy, <=-88 hopeless (extender).
- Remote reset hook: server answers HTTP 418 to an ingest when
  /hd2/pumpaudio/reset.flag exists (then deletes it) -> board machine.reset()s.
  Exists because a wedged/limping board previously had NO remote lever at all:
  a ~10% link keeps resetting the fail counter, so the 8-min self-reboot never
  fires, and weak-but-associated Wi-Fi never triggers the reconnect path.

**Pump acoustics (through the gappy link, so provisional):**
- Quiet basement floor: RMS ~10-13. Pump running: ~35-46 sustained with
  startup spikes 300-1180. Only ~3x the floor — marginal for detection;
  gain bump to 37.5 dB and/or closer placement would help.
- Footsteps/door closes are LOUDER than the pump (RMS 138-678) and are ALSO
  low-frequency (floor thumps), so RMS level, 40-250 Hz band energy, and
  crest factor all FAIL to separate person-noise from pump. The promising
  discriminator is tonal steadiness (pump holds one narrow motor line for
  the whole run; footsteps are chaotic second-to-second) — needs a clean
  pump cycle (nobody in the basement) to tune. Retrospective review is
  already robust regardless: click listen in the UI and your ears decide.
- pump_watch.py (host-side Monitor): adaptive threshold = max(28, 2.5x 25th
  pctile of last 10 min), ON when >=10 of last 30 s loud. Fires on people
  walking near the sensor — accepted for now, see above.

**UI:** outage strip (red = no data, orange = <60% of that pixel's seconds)
along the chart top + day coverage %. Future hours render neutral, not lost.

**USB handling rules (learned the hard way):**
- When the board visits USB: FLASH FIRST, probe later. Running
  `esptool chip-id` against a board that's actively streaming over its
  USB-JTAG port can wedge the CDC endpoint (opens fail with Errno 71 on the
  DTR ioctl; raw termios hangs too). Only a physical replug recovers it
  without root.
- After any replug/reset, wait a beat — deploying in the same second as
  enumeration races udev and fails with "No such file or directory".
- With both boards plugged in: Waveshare cam = 303a:4001 on ttyACM0, generic
  clone = CH340 on ttyUSB0. Prefer /dev/serial/by-id/ paths
  (usb-Espressif_..._a4cb8fd781900000-if00 = the wavecam).

**Scan-while-connected hang (2026-08-09, cost one battery-cycle trip):**
`sta.scan()` while ASSOCIATED can hang forever in C on ESP32 — main.py's
startup survey did exactly that after a remote reset: board pings, webrepl
port SYN-ACKs but nothing answers (REPL starved), crash-catcher useless
(it's a hang, not an exception), self-reboot counter frozen. Rules:
only scan when disconnected (boot.py pre-connect scan and the
disconnect->scan->connect reconnect path are safe), and main() now arms
machine.WDT(120000), fed once per chunk loop — ANY hang self-recovers in
2 min. WDT side effect: an interrupted REPL session (mic_runner/webrepl)
resets the board after 2 idle minutes — deploy fast or expect a reboot.

**Mesh RSSI is location-specific — never extrapolate:** node 3cba read
-80 dBm at the desk but -48 dBm in the basement (it's the closest node down
there); the desk-best node 3b2b reads -82 dBm in the basement. The board
clinging to 3b2b was the entire "basement dead zone" story. Basement
coverage went 8-25% (old firmware) -> 40-50% (auto-roam, still on the
desk-chosen node) -> expect ~full on 3cba after next boot.

**RSSI isn't throughput (2026-08-09 evening):** after auto-roam got the board
onto the basement-near node at -48 dBm, coverage only reached ~40-45%. Ping:
0% loss but 9-340 ms jitter = the satellite node's wireless BACKHAUL across
the house is the bottleneck, not the local hop. 320 KB chunk POSTs blow the
4 s timeout ~60% of the time. Fix staged for next USB visit: capture 8 kHz
mono on-board (the 256fs MONO config that tested live in miccheck8k) -> 80 KB
chunks; bundle with the scan-hang fix + WDT. Server should auto-detect chunk
format by size so either firmware works during the transition.

## USB CDC wedge: DTR ioctl times out even after replug (2026-08-09)

Symptom: pyserial `open()` dies with `TimeoutError: [Errno 110]` in
`_update_dtr_state()` — even on a freshly re-enumerated device. The CDC
*modem-control* channel is wedged, but the *data* path is fine.

Fix that works (see `raw_deploy.py` in this repo, now the preferred fallback):
- `os.open("/dev/serial/by-id/...", O_RDWR|O_NOCTTY|O_NONBLOCK)` — no pyserial
- raw termios (`CS8|CREAD|CLOCAL`, all else 0) — tcsetattr never touches DTR/RTS
- `\x03\x03` interrupts the streaming loop: KeyboardInterrupt is a
  BaseException, so the firmware's `except Exception: machine.reset()`
  catch-all does NOT swallow it — you land at a live REPL
- then the normal raw-REPL dance (`\r\x01`, 128 B chunks, `\x04`)

So: if mic_runner.py fails with Errno 110/71 on open, don't fight pyserial —
use the raw-fd deploy. Replug fixes enumeration but NOT this wedge.

## OTA: WebREPL cannot update a streaming board — use the /fw channel (2026-08-09)

WebREPL OTA fails every way you try it: while main.py streams, the busy loop
(blocking I2S readinto + requests.post) starves WebREPL's socket servicing —
TCP connects (SYN-ACK from lwIP) but the websocket handshake never completes,
even with 90 s patience, even racing the ~1 s boot window after a 418 reset.
A dangling half-open attempt then blocks WebREPL's single client slot.

The firmware now has a native OTA path instead (needs >= the 2026-08-09
ibuf-480k build):
1. stage:  cp wavecam_stream.py /hd2/pumpaudio/fw_stage.py
           touch /hd2/pumpaudio/update.flag
2. next /ingest POST gets HTTP 210 (a 200-equivalent: chunk is accepted)
3. board GETs /fw, syntax-checks with compile(), writes main_new.py,
   os.rename -> main.py, reboots. /fw consumes update.flag (one-shot) so a
   failed install can't boot-loop; re-touch the flag to retry.
Never set update.flag while a pre-210 firmware is on the board: old code
treats 210 as an error and retries the same chunk forever.
