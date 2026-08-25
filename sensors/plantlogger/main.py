"""FeatherS3 blink: flash the blue LED and cycle the RGB NeoPixel."""
import time

import feathers3
import neopixel
from machine import Pin

# NeoPixel is powered from LDO2 - must be on to light it
feathers3.set_ldo2_power(True)
pixel = neopixel.NeoPixel(Pin(feathers3.RGB_DATA), 1)

hue = 0
led_on = False
last_led = time.ticks_ms()

while True:
    now = time.ticks_ms()

    # Blue LED: toggle every 500 ms
    if time.ticks_diff(now, last_led) >= 500:
        led_on = not led_on
        feathers3.led_set(led_on)
        last_led = now

    # NeoPixel: slow rainbow sweep
    pixel[0] = feathers3.rgb_color_wheel(hue)
    pixel.write()
    hue = (hue + 1) % 256

    time.sleep_ms(20)
