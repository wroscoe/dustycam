# Diagnose framerate droop: auto-exposure vs memory vs thermal.
# Run: mpremote run bench_diagnose.py  (stop fb_webui.py first)

import gc
import sensor
import time

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_framerate(480)
sensor.skip_frames(time=2000)


def measure(n=200):
    t0 = time.ticks_ms()
    for _ in range(n):
        sensor.snapshot()
    dt = time.ticks_diff(time.ticks_ms(), t0)
    return n * 1000 / dt


print("phase 1: auto-exposure ON, sampling for ~30s")
for i in range(10):
    fps = measure()
    gc.collect()
    print("  t=%2ds fps=%.1f exposure_us=%d gc_free=%d"
          % (i * 3, fps, sensor.get_exposure_us(), gc.mem_free()))

print("phase 2: auto-exposure OFF, exposure=1500us, sampling for ~15s")
sensor.set_auto_exposure(False, exposure_us=1500)
sensor.skip_frames(time=500)
for i in range(5):
    fps = measure()
    print("  fps=%.1f exposure_us=%d" % (fps, sensor.get_exposure_us()))
