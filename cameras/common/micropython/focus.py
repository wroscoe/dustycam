"""focus: sharpness score + on-frame annotation for the setup stream.

Score = stdev of a x4 Laplacian over the centre ROI of a grayscale frame,
computed from the histogram (get_statistics() is integer-valued on fw 5.x).
Scene-dependent: judge by the peak, in daylight.
"""
import math

FOCUS_ROI = (200, 120, 240, 240)       # centre of VGA


def sharpness(img, roi=FOCUS_ROI):
    crop = img.copy(roi=roi)
    crop.laplacian(1, mul=4.0)
    # stdev of the laplacian, computed from the histogram: get_statistics()
    # returns whole numbers on fw 5.x (N6), too coarse to see focus change.
    h = crop.get_histogram()
    bins = h.bins() if hasattr(h, 'bins') else h.l_bins()
    mean = 0.0
    for i, p in enumerate(bins):
        mean += i * p
    var = 0.0
    for i, p in enumerate(bins):
        var += p * (i - mean) * (i - mean)
    return math.sqrt(var)


_DRAW_XY_TUPLE = None   # fw 5.x wants draw_string((x, y), txt); fw 4.x wants (x, y, txt)


def _draw_string(img, x, y, txt, **kw):
    global _DRAW_XY_TUPLE
    if _DRAW_XY_TUPLE is not False:
        try:
            img.draw_string((x, y), txt, **kw)
            _DRAW_XY_TUPLE = True
            return
        except (TypeError, ValueError):
            _DRAW_XY_TUPLE = False
    img.draw_string(x, y, txt, **kw)


def annotate(img, score, best, remaining, roi=FOCUS_ROI, quality=60):
    """Draw score/best/time + ROI box + bar; -> JPEG bytes (aliases the fb)."""
    img.draw_rectangle(roi, color=255, thickness=2)
    txt = 'focus %5.1f   best %5.1f   %3ds' % (score, best, remaining)
    _draw_string(img, 10, 10, txt, color=0, scale=4, mono_space=False)
    _draw_string(img, 8, 8, txt, color=255, scale=4, mono_space=False)
    bar = int(min(score, 100) / 100 * (img.width() - 16))
    img.draw_rectangle((8, img.height() - 24, bar, 16), color=255, fill=True)  # tuple form: fw 5.x (N6) rejects bare ints
    return img.to_jpeg(quality=quality).bytearray()
