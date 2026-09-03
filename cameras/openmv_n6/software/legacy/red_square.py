# Draw a filled red square in the middle of every frame.
# View via fb_webui.py (http://localhost:8080) or the OpenMV IDE.
#
# sensor.set_framerate is the key to speed on the N6: the default capture
# mode idles at ~118 fps regardless of resolution/exposure; requesting more
# unlocks up to ~460 fps at QVGA (~235 fps at VGA). At 460 fps max exposure
# is ~2 ms, so expect dark frames unless the scene is bright.

import sensor
import time

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_framerate(480)  # clamps to the sensor ceiling (~460 fps)
sensor.skip_frames(time=2000)

SIZE = 60
n = 0
t_last = time.ticks_ms()

while True:
    img = sensor.snapshot()
    x = (img.width() - SIZE) // 2
    y = (img.height() - SIZE) // 2
    img.draw_rectangle((x, y, SIZE, SIZE), color=(255, 0, 0), fill=True)
    n += 1
    if n == 200:
        # windowed fps over the last 200 frames; NOTE clock.fps() is a
        # cumulative average since boot and *looks* like a slow decay
        now = time.ticks_ms()
        print("%.1f" % (200000 / time.ticks_diff(now, t_last)))
        t_last = now
        n = 0
