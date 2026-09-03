# Draw a filled red square in the middle of every frame — no printing.
# See red_square.py for the annotated version.

import sensor
import time

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_framerate(480)  # clamps to the sensor ceiling (~460 fps)
sensor.skip_frames(time=2000)

SIZE = 60

while True:
    img = sensor.snapshot()
    x = (img.width() - SIZE) // 2
    y = (img.height() - SIZE) // 2
    img.draw_rectangle((x, y, SIZE, SIZE), color=(255, 0, 0), fill=True)
