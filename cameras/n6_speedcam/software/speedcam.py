"""n6_speedcam app skeleton — radar-gated capture on the OpenMV N6.

Deployed as /flash/app.py under the openmv_n6 OTA bootstrap
(cameras/openmv_n6/software/ota_main.py): main.py brings up WiFi + OTA and
calls app.run(ota.poll). This file is the radar-specific part; frame upload,
MQTT and SD buffering are meant to come from sensorhub_cam.py once this loop
is proven on the bench.

Wiring (hardware/carrier/): radar UART on P4/P5 (UART3), WAKE on P11.

Two modes, chosen by SLEEP:
  SLEEP = False   the N6 stays up, watches the UART, and captures when a
                  Speed frame above SPEED_MIN arrives. Bench mode.
  SLEEP = True    after IDLE_S without a frame the N6 deep-sleeps; the radar's
                  trigger output (command 0x04) pulls P11 low and wakes it.
                  This is the mode the power budget assumes -- UNTESTED on
                  OpenMV 5.0 firmware; the pinout promises P11 = WKUP3.
"""
import time
import sensor
import machine
from machine import Pin

from ld2415h import Radar

SLEEP = False
SPEED_MIN = 15.0          # km/h that counts as a vehicle worth a frame
IDLE_S = 30               # no frames for this long -> deep sleep (SLEEP mode)
TRIGGER_HOLD_S = 2        # radar holds its trigger output this long
RADAR_SENSITIVITY = 5     # 1 (far, twitchy) .. 15 (near, calm); tune on site
RADAR_ANGLE_DEG = 10      # beam-to-road angle compensation


def setup_camera():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.VGA)
    sensor.skip_frames(time=500)


def capture(speed):
    """One frame tagged with the radar reading. Replace with the sensorhub upload."""
    img = sensor.snapshot()
    ts = time.time()
    tag = "%s%.1f" % ("+" if speed.approaching else "-", speed.value)
    try:
        img.save("/sdcard/speed_%d_%s.jpg" % (ts, tag))
    except OSError:
        pass
    print("capture", ts, tag)


def run(idle=None):
    radar = Radar(uart_id=3, tx="P4", rx="P5")
    radar.configure(min_kmh=int(SPEED_MIN), angle_deg=RADAR_ANGLE_DEG,
                    sensitivity=RADAR_SENSITIVITY, direction=Radar.BOTH, unit=Radar.KMH)
    radar.set_trigger(hold_s=TRIGGER_HOLD_S, threshold_kmh=int(SPEED_MIN))
    wake = Pin("P11", Pin.IN, Pin.PULL_UP)
    setup_camera()
    last_frame = time.ticks_ms()
    peak = None
    while True:
        for s in radar.poll():
            last_frame = time.ticks_ms()
            if s.value >= SPEED_MIN and (peak is None or s.value > peak.value):
                peak = s
        # a pass is over when the radar goes quiet for half a second: record its peak
        if peak is not None and time.ticks_diff(time.ticks_ms(), last_frame) > 500:
            capture(peak)
            peak = None
        if idle:
            idle()
        if SLEEP and time.ticks_diff(time.ticks_ms(), last_frame) > IDLE_S * 1000:
            print("idle -> deepsleep, wake on P11 (now %d)" % wake.value())
            machine.deepsleep()
        time.sleep_ms(20)


if __name__ == "__main__":
    run()
