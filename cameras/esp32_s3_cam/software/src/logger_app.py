"""Autonomous image logger: capture -> JPEG -> flash -> upload when online.

Never prints in autonomous mode (console blocks with no host attached);
logs to /imgs/log.txt instead. Run interactively via make targets with
echo=True for visible output.
"""
import time

import jpeg
from camera import Camera, PixelFormat, FrameSize

import boardcam
import storage
import uplink

CAPTURE_INTERVAL_S = 2
UPLOAD_EVERY_N = 10        # try WiFi/upload every N captures

_W, _H = 640, 480


def run(count=None, echo=False, wdt=None):
    storage.init()

    def say(msg):
        storage.log(msg)
        if echo:
            print(msg)

    cam = boardcam.get_camera(frame_size=FrameSize.VGA,
                              pixel_format=PixelFormat.GRAYSCALE)
    enc = jpeg.Encoder(width=_W, height=_H, pixel_format='GRAY')
    say('logger start, free_kb=%d' % storage.free_kb())
    n = 0
    try:
        while count is None or n < count:
            if wdt:
                wdt.feed()
            frame = cam.capture()
            data = enc.encode(frame)
            path = storage.save_jpeg(data)
            say('cap %s %dB' % (path, len(data)))
            n += 1
            if n % UPLOAD_EVERY_N == 0:
                up = uplink.upload_pending()
                say('uploaded %d, %d pending' % (up, len(storage.pending())))
            time.sleep(CAPTURE_INTERVAL_S)
    finally:
        boardcam.release()
        up = uplink.upload_pending()
        say('final upload %d, %d pending' % (up, len(storage.pending())))
