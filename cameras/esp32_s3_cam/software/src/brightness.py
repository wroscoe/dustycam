"""Pure logic: brightness from an RGB565 frame buffer.

No hardware imports — runs identically on MicroPython (board) and CPython
(host tests).
"""


def mean_luma(buf, samples=1024):
    """Mean luma (0-255) of a big-endian RGB565 buffer, sampled sparsely.

    Samples ~`samples` pixels evenly across the buffer instead of touching
    all of them - plenty for a brightness decision and fast enough for a
    MicroPython loop.
    """
    npix = len(buf) // 2
    if npix == 0:
        return 0
    step = max(1, npix // samples)
    total = 0
    count = 0
    for i in range(0, npix, step):
        j = i * 2
        v = (buf[j] << 8) | buf[j + 1]
        r = (v >> 11) & 0x1F
        g = (v >> 5) & 0x3F
        b = v & 0x1F
        # ITU-R 601 luma weights on 8-bit-expanded channels
        total += (299 * ((r * 255) // 31)
                  + 587 * ((g * 255) // 63)
                  + 114 * ((b * 255) // 31)) // 1000
        count += 1
    return total // count


def is_dark(buf, threshold=40, samples=1024):
    return mean_luma(buf, samples) < threshold
