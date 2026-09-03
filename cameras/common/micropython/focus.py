"""focus: sharpness score + on-frame annotation for the setup stream.

Score = stdev of a x4 Laplacian over the centre ROI of a grayscale frame.
get_statistics().l_stdev() on grayscale is integer-valued, hence the x4.
Scene-dependent: judge by the peak, in daylight.
"""
FOCUS_ROI = (200, 120, 240, 240)       # centre of VGA


def sharpness(img, roi=FOCUS_ROI):
    crop = img.copy(roi=roi)
    crop.laplacian(1, mul=4.0)
    st = crop.get_statistics()
    return st.l_stdev() if hasattr(st, 'l_stdev') else st.stdev()


def annotate(img, score, best, remaining, roi=FOCUS_ROI, quality=60):
    """Draw score/best/time + ROI box + bar; -> JPEG bytes (aliases the fb)."""
    img.draw_rectangle(roi, color=255, thickness=2)
    txt = 'focus %5.1f   best %5.1f   %3ds' % (score, best, remaining)
    img.draw_string(10, 10, txt, color=0, scale=4, mono_space=False)
    img.draw_string(8, 8, txt, color=255, scale=4, mono_space=False)
    bar = int(min(score, 100) / 100 * (img.width() - 16))
    img.draw_rectangle(8, img.height() - 24, bar, 16, color=255, fill=True)
    return img.to_jpeg(quality=quality).bytearray()
