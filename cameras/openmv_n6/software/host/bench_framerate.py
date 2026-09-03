import sensor
import time

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=1000)

print("sensor id:", hex(sensor.get_id()))
print("has get_framerate:", hasattr(sensor, "get_framerate"))
print("has set_framerate:", hasattr(sensor, "set_framerate"))
try:
    print("current framerate:", sensor.get_framerate())
except Exception as e:
    print("get_framerate err:", e)


def measure(n=80):
    t0 = time.ticks_ms()
    for _ in range(n):
        sensor.snapshot()
    dt = time.ticks_diff(time.ticks_ms(), t0)
    return n * 1000 / dt


for fps_req in (240, 200, 150, 120):
    try:
        sensor.set_framerate(fps_req)
        sensor.skip_frames(time=500)
        print("requested %d -> measured %.1f" % (fps_req, measure()))
    except Exception as e:
        print("set_framerate(%d) err:" % fps_req, e)
