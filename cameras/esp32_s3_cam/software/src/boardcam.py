"""Camera wrapper for the Waveshare ESP32-S3-CAM (GC0308 sensor).

Owns the one rule that must never be broken on this board: the camera is
initialized exactly once per boot. Re-initializing in a dirty session
hard-hangs the interpreter (see LESSONS.md) - always soft-reset first.
"""
from camera import Camera, PixelFormat, FrameSize

# Waveshare ESP32-S3-CAM pin map (NOT the real ESP32-S3-EYE map)
PINS = dict(
    data_pins=[45, 47, 48, 46, 42, 40, 39, 21],
    vsync_pin=17, href_pin=18,
    sda_pin=8, scl_pin=7,
    pclk_pin=41, xclk_pin=38,
    xclk_freq=20_000_000,
)

_cam = None


def get_camera(frame_size=FrameSize.QVGA, pixel_format=PixelFormat.RGB565):
    """Return the singleton camera, initializing it on first call."""
    global _cam
    if _cam is None:
        _cam = Camera(pixel_format=pixel_format,
                      frame_size=frame_size, init=False, **PINS)
        _cam.init()
        for _ in range(3):      # warm-up frames for auto-exposure
            _cam.capture()
    return _cam


def release():
    """Deinit the camera. MUST be called before the session ends (try/finally)
    or the next init will hang the board until a power cycle."""
    global _cam
    if _cam is not None:
        try:
            _cam.deinit()
        finally:
            _cam = None
