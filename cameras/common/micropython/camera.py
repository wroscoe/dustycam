"""camera: preview stream + full-quality capture. Standard §2 stages 2-3.

The loop watches on a small RGB565 stream. A recorded frame switches the
sensor to CFG['capture_framesize'] in JPEG pixformat so the sensor's own
encoder produces the frame and the frame buffer holds only the JPEG. The
returned buffer ALIASES the frame buffer: upload it or write it to SD
before the next snapshot()/mode switch, then call restore_preview().
Falls back to a software JPEG of the preview if the switch fails.

Board facts used: PREVIEW_FRAMESIZE (sensor.<NAME>), CAPTURE_MODE
('sensor_jpeg' = the sensor encodes, OV5640; 'rgb565' = RGB565 at the
capture size + software to_jpeg, e.g. the N6 whose CSI rejects JPEG),
CFG['capture_framesize'], CFG['capture_settle_ms'], JPEG_QUALITY.
"""
import gc

import sensor

from config import *

_fullres = False
LAST_CAPTURE = ['']          # 'fullres WxH nB' | 'failed ... -> fallback ...', for /status


def preview_init():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(getattr(sensor, PREVIEW_FRAMESIZE))
    sensor.skip_frames(time=1500)
    return sensor.width(), sensor.height()


def restore_preview():
    global _fullres
    if not _fullres:
        return
    _fullres = False
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(getattr(sensor, PREVIEW_FRAMESIZE))
    sensor.skip_frames(time=CFG['capture_settle_ms'])


def capture():
    """-> (jpeg bytes-like, w, h)."""
    global _fullres
    name = CFG.get('capture_framesize', '')
    fs = getattr(sensor, name, None) if name else None
    mode = globals().get('CAPTURE_MODE', 'sensor_jpeg')
    if fs is not None:
        try:
            gc.collect()
            if mode == 'rgb565':
                # RGB565 at the capture size, encoded in software (needs the
                # frame in the fb: 1280x800x2 = 2 MB on the N6)
                sensor.set_pixformat(sensor.RGB565)
                sensor.set_framesize(fs)
                _fullres = True
                sensor.skip_frames(time=CFG['capture_settle_ms'])
                img = sensor.snapshot()
                w, h = img.width(), img.height()
                jpg = img.to_jpeg(quality=JPEG_QUALITY)
                data = jpg.bytearray()
            else:
                sensor.set_pixformat(sensor.JPEG)
                sensor.set_framesize(fs)
                _fullres = True
                if hasattr(sensor, 'set_quality'):
                    sensor.set_quality(JPEG_QUALITY)
                sensor.skip_frames(time=CFG['capture_settle_ms'])
                img = sensor.snapshot()
                data = img.bytearray()
                w, h = img.width(), img.height()
            if len(data) > 4096 and w * h > 640 * 400:
                LAST_CAPTURE[0] = 'fullres %dx%d %dB' % (w, h, len(data))
                return data, w, h
            LAST_CAPTURE[0] = 'rejected %dx%d %dB' % (w, h, len(data))
            print('full-res capture rejected: %dx%d %d bytes' % (w, h, len(data)))
        except Exception as e:
            LAST_CAPTURE[0] = 'failed %r' % e
            print('full-res capture failed:', repr(e))
        restore_preview()
    img = sensor.snapshot()
    jpg = img.to_jpeg(quality=JPEG_QUALITY)
    data = jpg.bytearray()
    LAST_CAPTURE[0] += ' -> fallback %dx%d %dB' % (jpg.width(), jpg.height(), len(data))
    return data, jpg.width(), jpg.height()


def grayscale_preview():
    """Setup mode's stream format: 1 byte/pixel, cheap to score."""
    global _fullres
    _fullres = True                     # so restore_preview() knows to switch back
    sensor.set_pixformat(sensor.GRAYSCALE)
    sensor.set_framesize(getattr(sensor, PREVIEW_FRAMESIZE))
    sensor.skip_frames(time=300)
