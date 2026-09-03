"""board: facts and stamped defaults for the OpenMV N6 (STM32N657).
Bundled first (see camera.toml [bundle]); the shared modules read these
names at call time.

Firmware 5.0.0 / MicroPython 1.28 (v1.28.0-49): the legacy `sensor` API
still works; `machine.LED`, `ssl.SSLContext`, `json` present; pins
`SW` (user button, active-low), `CHG` (active-low charging), `BAT_ADC`,
`LED_RED/GREEN/BLUE`. Sensor: CSI camera, native HD 1280x800 (VGA is
640x400, QVGA 320x200 — 16:10, not 4:3); the CSI **rejects JPEG and
YUV422 pixformats**, so captures are RGB565 at HD + software JPEG
(CAPTURE_MODE = 'rgb565'; a 1280x800 RGB565 frame is 2 MB in the fb).
Framebuffer pool 32 MB (20 MiB usable), fb cost = w*h*2 for every format
(sarg: sargbench1 lessons, 2026-08).

Deny list (hard-hangs the MCU, bypasses rollback): pyb.ADCAll /
read_core_temp while streaming; sensor.set_frame_callback(); cpufreq
unproven — none of these are called.
BATT_DIVIDER 1.5 is inferred, not measured: batt_v is indicative only.
"""
import machine

APP_VERSION = '2.0.10-n6'

# --- tuning: stamped by tools/dustygen from camera.toml [tuning] overridden by
# ~/.dusty/config.toml [camera.openmv_n6]; served at /config/n6cam and pulled
# at runtime (config.py).
TUNING = {'period_s': 10, 'diff_min_frac': 0.005, 'diff_l_thresh': 8, 'heartbeat_s': 300, 'telemetry_s': 60, 'capture_framesize': 'HD', 'capture_settle_ms': 400, 'setup_secs': 240, 'wifi_linger_s': 0}
# --- end tuning

PREVIEW_FRAMESIZE = 'VGA'         # 640x400 on this sensor
JPEG_QUALITY = 85
CAPTURE_MODE = 'rgb565'           # CSI has no JPEG output: RGB565 at HD + to_jpeg
BUTTON_NAMES = ('SW',)
LED_NAME = 'LED_BLUE'
MAX_PENDING = 2000
WIFI_RETRY_S = 30
BATT_DIVIDER = 1.5                # inferred; verify against a real pack

try:
    _bat_adc = machine.ADC(machine.Pin.board.BAT_ADC)
except (AttributeError, ValueError, OSError):
    _bat_adc = None
try:
    _chg_pin = machine.Pin(machine.Pin.board.CHG, machine.Pin.IN, machine.Pin.PULL_UP)
except (AttributeError, ValueError, OSError):
    _chg_pin = None


def board_sensors():
    """Sense stage for this board: battery volts (8-sample mean) + charging."""
    vals = {}
    if _bat_adc is not None:
        try:
            # BAT_ADC reads ~550 or the real value (~55000) in alternating bursts a
            # few hundred ms long (the divider is switched under the ADC, REPL-verified
            # 2026-09-03): take the max of samples spread over ~0.6 s.
            import time
            raw = 0
            for _ in range(5):
                raw = max(raw, _bat_adc.read_u16())
                time.sleep_ms(120)
            vals['batt_v'] = round(raw / 65535 * 3.3 * BATT_DIVIDER, 3)
        except (OSError, ValueError):
            pass
    if _chg_pin is not None:
        try:
            vals['charging'] = 0 if _chg_pin.value() else 1
        except (OSError, ValueError):
            pass
    return vals
