# Recipe: writing the software for a new camera

Paste this as the brief for a session that builds a new camera. It assumes
[`camera_standard.md`](camera_standard.md) is in force.

---

**You are writing the software for a new dustycam camera.** Read
`docs/camera_standard.md` first; it defines the pipeline, the modes, the
contract with sensorhub, the configuration tiers and the folder layout.
Follow these steps in order. Do not skip a step; if a step does not apply,
write why in the README's "Standard mapping" section.

**1. Establish the facts before writing code.** Query sarg for the board,
its sensor, its firmware version, and every part in the design. Read the
closest existing camera (same runtime) and its README. Write `camera.toml`:
id, board, runtime, power model, sensors, capabilities. Record the board
facts you will need: pins for the button and LED, the camera framesize and
pixel-format names, the memory budget (frame buffer, heap), storage (SD or
flash), radios, and any calls known to hang the board. Put unknowns in a
list and resolve them on the bench before step 4.

**2. Decide the trigger and the power model.** Always-on with motion diff,
or wake-cycle with a sensor interrupt or an interval. State the expected
frame rate to the server and the expected daily energy. If there is an
on-device model, name its input size and where its weights live.

**3. Lay out the folder** exactly as the standard says. Copy nothing from
another camera's app; import from `cameras/common/<runtime>/` and list the
modules in `camera.toml`. Write the README skeleton with the "Standard
mapping" table filled in with your intentions.

**4. Implement live mode stage by stage**, in this order, testing each on
the bench before the next:

1. Boot: secrets, tuning defaults, stored config, recovery check.
2. Connect: WiFi, NTP, IP noted.
3. Deliver: one hard-coded frame reaches sensorhub with the standard meta.
4. Report: telemetry appears on the device page.
5. Serve: `/status` answers; config pull works and `cfg` changes; firmware
   pull installs and a bad build rolls back.
6. Watch: preview and trigger, every trigger with its `why`.
7. Capture: full quality.
8. Judge: the model, if any, with audit frames.
9. Record: spool and drain, verified by unplugging the network.
10. Rest: the idle loop keeps the control plane responsive, or deep sleep
    with state persisted and a wake that resumes cleanly.

**5. Implement setup mode**: button entry, `/setup` entry, server config
entry; the preview stream with the focus score; `/shoot`; the phone page
with its watchdog; timeout back to live; the LED. Verify from a phone on the
LAN that the page reconnects after the screen locks.

**6. Write the host tools**: `dustygen` support for this camera's secrets
format if it is new, a `stage-firmware` target, a bench script that prints
the board's status. Add the device to `/hd2/sensorhub/pages/devices.json`
with `expect`.

**7. Write host tests** for every piece of pure logic (parsers, trigger
math, meta builder, config merge, spool naming). They must pass without a
board.

**8. Prove it in the field**: one full day of live mode with a heartbeat
every 5 minutes; then a config change pulled without reflash; then a
firmware update with a deliberate bad build that rolls back. Record what you
learned in sarg. Update the README status line and `camera.toml` `status`.

**Laws** (from the standard): heartbeat is the safety net; firmware before
config; `ts`, `ip`, `v`, `why` in every meta; telemetry over the gate, never
raw MQTT; no TLS while a viewer is attached; the spool is capped; the
control plane is polled from everywhere the loop waits; recovery mode is
never updated over the air.

**Before you finish**, go through the "Definition of done" checklist at the
end of the standard and leave any unchecked item, with its reason, in the
README.
