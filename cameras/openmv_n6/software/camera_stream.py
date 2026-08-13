# OpenMV camera-side video streamer.
# Waits for a 4-byte "snap" command on the USB virtual comm port, then replies
# with a little-endian uint32 JPEG size followed by the JPEG bytes.
# Deployed to the camera as /flash/main.py so it runs on boot.

import sensor
import struct
from pyb import USB_VCP
from machine import LED

usb = USB_VCP()
led = LED("LED_GREEN")

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

while True:
    cmd = usb.recv(4, timeout=5000)
    if cmd == b"snap":
        led.on()
        img = sensor.snapshot().to_jpeg(quality=80)
        usb.send(struct.pack("<L", img.size()))
        usb.send(img)
        led.off()
