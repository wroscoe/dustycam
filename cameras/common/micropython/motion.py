"""motion: the Watch stage's trigger. Standard §2 stage 2.

Frame difference against the last *recorded* frame (not the previous
preview), so a slow change accumulates until it crosses the gate. Two heap
image.Image buffers (not alloc_extra_fb, which fw 5.0 dropped).
"""
import image
import sensor


class MotionGate:
    def __init__(self, w, h):
        self.ref = image.Image(w, h, sensor.RGB565)    # last recorded frame
        self.work = image.Image(w, h, sensor.RGB565)   # scratch for the diff
        self.have_ref = False
        self.win_max = 0.0                             # stats over a telemetry window
        self.win_sum = 0.0
        self.win_n = 0

    def diff(self, img, l_thresh):
        """Fraction of pixels whose luminance changed by >= l_thresh
        (1.0 when there is no reference yet)."""
        if not self.have_ref:
            return 1.0
        self.work.replace(img)
        self.work.difference(self.ref)
        frac = sum(self.work.get_histogram().l_bins()[l_thresh:])
        if frac > self.win_max:
            self.win_max = frac
        self.win_sum += frac
        self.win_n += 1
        return frac

    def commit(self, img):
        self.ref.replace(img)
        self.have_ref = True

    def reset(self):
        """After setup mode or any exposure change: the next frame records."""
        self.have_ref = False

    def window(self):
        """(max, mean) since the last call; resets the window."""
        out = (round(self.win_max, 4),
               round(self.win_sum / self.win_n, 4) if self.win_n else 0)
        self.win_max = self.win_sum = 0.0
        self.win_n = 0
        return out
