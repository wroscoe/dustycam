"""Read the Adafruit STEMMA soil sensor (seesaw @ 0x36): moisture + temperature."""
import time

import feathers3
from machine import I2C, Pin

ADDR = 0x36

# Sensor lives on the FeatherS3D's second STEMMA port (I2C2, LDO2-powered).
# Keep it off the I2C1 port: the onboard MAX17048 fuel gauge also uses 0x36 there.
feathers3.set_ldo2_power(True)
time.sleep_ms(100)

i2c = I2C(1, sda=Pin(16), scl=Pin(15))


def read_moisture():
    # seesaw touch module (0x0F), channel 0 (0x10); raw range ~200 (dry) - 2000 (wet)
    i2c.writeto(ADDR, bytes([0x0F, 0x10]))
    time.sleep_ms(5)
    raw = i2c.readfrom(ADDR, 2)
    return (raw[0] << 8) | raw[1]

def read_temperature():
    # seesaw status module (0x00), temp register (0x04); signed 16.16 fixed point
    i2c.writeto(ADDR, bytes([0x00, 0x04]))
    time.sleep_ms(5)
    raw = i2c.readfrom(ADDR, 4)
    val = (raw[0] << 24) | (raw[1] << 16) | (raw[2] << 8) | raw[3]
    if val & 0x80000000:
        val -= 0x100000000
    return val / 65536

for _ in range(5):
    print("moisture: %4d (raw, ~200 dry .. ~2000 wet)   temp: %.1f C" % (read_moisture(), read_temperature()))
    time.sleep(1)
