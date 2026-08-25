"""FeatherS3[D] battery status via the onboard MAX17048 fuel gauge (I2C1 @ 0x36).

Note: this board variant has NO analog VBAT divider on IO2 - the FeatherS3
helper's get_battery_voltage() reads a floating pin and must not be used.
"""
import struct

import feathers3
from machine import I2C, Pin

i2c = I2C(0, sda=Pin(8), scl=Pin(9))


def _reg(addr, signed=False):
    fmt = ">h" if signed else ">H"
    return struct.unpack(fmt, i2c.readfrom_mem(0x36, addr, 2))[0]


voltage = _reg(0x02) * 78.125e-6      # VCELL
soc = _reg(0x04) / 256                # state of charge %
rate = _reg(0x16, signed=True) * 0.208  # charge/discharge %/hr

print("battery: %.3f V   charge: %.1f %%   rate: %+.1f %%/hr   USB: %s"
      % (voltage, min(soc, 100.0), rate, feathers3.get_vbus_present()))
