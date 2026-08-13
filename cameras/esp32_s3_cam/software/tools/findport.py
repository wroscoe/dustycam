#!/usr/bin/env python3
"""Print the board's serial device path, whichever USB mode it's in.

The by-id name embeds the USB product string, which differs between the
ROM/IDF JTAG mode ("USB_JTAG_serial_debug_unit") and MicroPython's CDC
("Espressif_Device") — so no single by-id path is stable across modes.
This resolves whichever is present (both contain 'Espressif').
"""
import glob
import sys

links = sorted(glob.glob('/dev/serial/by-id/usb-Espressif*'))
if not links:
    print('no Espressif serial device found', file=sys.stderr)
    sys.exit(1)
print(links[0])
