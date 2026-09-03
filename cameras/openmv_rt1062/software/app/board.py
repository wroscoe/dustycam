"""board: facts and stamped defaults for the OpenMV Cam RT1062 (R6).
Bundled first (see camera.toml [bundle]); the shared modules read these
names at call time.

Firmware 4.8.1 / MicroPython 1.26: legacy `sensor` API, `machine.LED`,
user button `machine.Pin.board.SW` (active-low), OV5640 native WQXGA2
2592x1944 with on-sensor JPEG. Frame buffer 13 MB, fb_alloc 10 MB, GC heap
8 MB (boards/OPENMV_RT1060/omv_boardconfig.h @ v4.8.1).
Deny list: none known on this board (the N6's ADCAll/core-temp hang does
not apply; there is no `cpufreq`, `imu` or `BAT_ADC`).
"""
import machine

APP_VERSION = '2.0.8-rt'

# --- tuning: stamped by tools/dustygen from camera.toml [tuning] overridden by
# ~/.dusty/config.toml [camera.openmv_rt1062]; the same keys are served by
# sensorhub at /config/rt1062cam and pulled at runtime (config.py).
TUNING = {'period_s': 10, 'diff_min_frac': 0.04, 'diff_l_thresh': 20, 'heartbeat_s': 300, 'telemetry_s': 60, 'capture_framesize': 'WQXGA2', 'capture_settle_ms': 400, 'setup_secs': 300, 'wifi_linger_s': 0}
# --- end tuning

PREVIEW_FRAMESIZE = 'VGA'         # the Watch stream and the setup stream
JPEG_QUALITY = 85
BUTTON_NAMES = ('SW',)            # probed in order; first that resolves wins
LED_NAME = 'LED_BLUE'
CAPTURE_MODE = 'sensor_jpeg'      # OV5640 encodes on-chip
MAX_PENDING = 2000                # spool cap (files)
WIFI_RETRY_S = 30

try:    # R6 exposes CHG (active-low charge indicator), no BAT_ADC
    _chg_pin = machine.Pin(machine.Pin.board.CHG, machine.Pin.IN, machine.Pin.PULL_UP)
except (AttributeError, ValueError, OSError):
    _chg_pin = None


def board_sensors():
    """Sense stage for this board: charging only (no battery ADC on the R6)."""
    vals = {}
    if _chg_pin is not None:
        try:
            vals['charging'] = 0 if _chg_pin.value() else 1
        except (OSError, ValueError):
            pass
    return vals
