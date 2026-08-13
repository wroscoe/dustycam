#!/usr/bin/env python3
"""Watch the stream channel's available-size while reading frames slowly.
If size grows over time, the ring buffer is backlogging and reads slow down."""

import time

import omv_patches  # noqa: F401
from openmv.camera import Camera

cam = Camera(port="/dev/ttyACM0", timeout=2.0, ack=False)
cam.connect()
cam.reset()
time.sleep(4)

cam = Camera(port="/dev/ttyACM0", timeout=2.0, ack=False)
cam.connect()
cam.streaming(True)
stream_id = cam.get_channel(name="stream")

t_start = time.monotonic()
next_report = 0.0
reads = 0
while time.monotonic() - t_start < 45:
    t = time.monotonic() - t_start
    if t >= next_report:
        next_report = t + 5
        try:
            size = cam._channel_size(stream_id)
        except Exception as e:
            size = f"err {type(e).__name__}"
        print(f"t={t:5.1f}s reads={reads:4d} channel_size={size}", flush=True)
    try:
        r0 = time.monotonic()
        f = cam.read_frame()
        dt = time.monotonic() - r0
        if f:
            reads += 1
            if reads % 100 == 0:
                print(f"   read_frame took {dt*1000:.0f} ms, raw={f['raw_size']}",
                      flush=True)
    except Exception as e:
        print("  read err:", type(e).__name__, flush=True)
        time.sleep(0.1)
    time.sleep(0.03)  # deliberately slow host, like the throttled server

cam.disconnect()
print("done", flush=True)
