"""Standalone battery logger: sample VBAT + VBUS 4x/s to battlog.csv, blink LED."""
import time

import feathers3
from machine import ADC, Pin

adc = ADC(Pin(feathers3.VBAT_SENSE), atten=ADC.ATTN_11DB)
led = False
start = time.ticks_ms()

with open("battlog.csv", "w") as f:
    f.write("ms,vbus,pin_v\n")
    while True:
        v = adc.read_uv() / 1e6
        f.write(
            "%d,%d,%.3f\n"
            % (time.ticks_diff(time.ticks_ms(), start), feathers3.get_vbus_present(), v)
        )
        f.flush()
        led = not led
        feathers3.led_set(led)
        time.sleep_ms(250)
