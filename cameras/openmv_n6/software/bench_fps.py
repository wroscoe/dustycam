# Camera-side FPS benchmark for the OpenMV N6.
# Run from the host with: mpremote run bench_fps.py
# Measures pure sensor-loop speed (no host streaming attached).

import sensor
import time

SIZE = 60


def measure(n=80):
    t0 = time.ticks_ms()
    for _ in range(n):
        sensor.snapshot()
    dt = time.ticks_diff(time.ticks_ms(), t0)
    return n * 1000 / dt


def measure_draw(n=80):
    t0 = time.ticks_ms()
    for _ in range(n):
        img = sensor.snapshot()
        x = (img.width() - SIZE) // 2
        y = (img.height() - SIZE) // 2
        img.draw_rectangle((x, y, SIZE, SIZE), color=(255, 0, 0), fill=True)
    dt = time.ticks_diff(time.ticks_ms(), t0)
    return n * 1000 / dt


def setup(pixformat, framesize):
    sensor.reset()
    sensor.set_pixformat(pixformat)
    sensor.set_framesize(framesize)
    sensor.skip_frames(time=1500)


print("=== OpenMV N6 sensor FPS benchmark ===")

setup(sensor.RGB565, sensor.QVGA)
print("RGB565 QVGA auto-exposure:      %.1f fps" % measure())
print("RGB565 QVGA + draw rectangle:   %.1f fps" % measure_draw())

# manual short exposures (dark frames are fine, we want the speed ceiling)
for us in (8000, 4000, 2000, 500):
    try:
        sensor.set_auto_exposure(False, exposure_us=us)
        sensor.skip_frames(time=500)
        print("RGB565 QVGA exposure=%dus:   %.1f fps" % (us, measure()))
    except Exception as e:
        print("exposure", us, "failed:", e)
sensor.set_auto_exposure(True)

setup(sensor.GRAYSCALE, sensor.QVGA)
print("GRAY   QVGA auto-exposure:      %.1f fps" % measure())

setup(sensor.RGB565, sensor.QQVGA)
print("RGB565 QQVGA auto-exposure:     %.1f fps" % measure())

setup(sensor.RGB565, sensor.VGA)
print("RGB565 VGA auto-exposure:       %.1f fps" % measure())

try:
    setup(sensor.RGB565, sensor.HD)
    print("RGB565 HD auto-exposure:        %.1f fps" % measure())
except Exception as e:
    print("HD failed:", e)

print("=== done ===")
