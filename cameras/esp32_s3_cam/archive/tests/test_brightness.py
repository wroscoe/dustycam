"""Host-side tests for the pure brightness logic - no board needed."""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import brightness


def rgb565(r8, g8, b8):
    """Pack 8-bit RGB into a big-endian RGB565 pixel."""
    v = ((r8 >> 3) << 11) | ((g8 >> 2) << 5) | (b8 >> 3)
    return struct.pack(">H", v)


def frame_of(r8, g8, b8, npix=1000):
    return rgb565(r8, g8, b8) * npix


def test_black_is_zero():
    assert brightness.mean_luma(frame_of(0, 0, 0)) == 0


def test_white_is_bright():
    assert brightness.mean_luma(frame_of(255, 255, 255)) > 250


def test_mid_gray():
    luma = brightness.mean_luma(frame_of(128, 128, 128))
    assert 110 < luma < 140


def test_green_brighter_than_blue():
    # luma weights: green dominates
    assert (brightness.mean_luma(frame_of(0, 255, 0))
            > brightness.mean_luma(frame_of(0, 0, 255)))


def test_is_dark_threshold():
    assert brightness.is_dark(frame_of(10, 10, 10), threshold=40)
    assert not brightness.is_dark(frame_of(200, 200, 200), threshold=40)


def test_empty_buffer():
    assert brightness.mean_luma(b"") == 0


def test_real_frame_if_present():
    """If a captured frame exists, sanity-check it end to end."""
    path = os.path.join(os.path.dirname(__file__), "..", "frame_rgb565.bin")
    if not os.path.exists(path):
        return
    raw = open(path, "rb").read()
    luma = brightness.mean_luma(raw)
    assert 0 <= luma <= 255
