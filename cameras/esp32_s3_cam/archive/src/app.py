"""Demo app: report DARK/BRIGHT from live camera luma.

The board has no ESP32-controllable LED (schematic: LEDs are power/charge
indicators on the helper MCU), so the "LED" is the printed status line.

Deliberately named app.py, NOT main.py: a main.py that touches the camera
autostarts at boot with no host reading the CDC and wedges the board
(replug-only recovery). Never deploy camera code as main.py.

Run from the host:  make run   (deploy + hard-reset + bounded demo)
"""
import time
import boardcam
import brightness

THRESHOLD = 40      # luma 0-255; below this counts as dark
PERIOD_S = 0.5


def run(iterations=None):
    # deinit ALWAYS: a camera left initialized + a later re-init on dirty
    # hardware hard-hangs the board (LESSONS.md #13) — only a replug recovers.
    cam = boardcam.get_camera()
    try:
        n = 0
        while iterations is None or n < iterations:
            frame = cam.capture()
            luma = brightness.mean_luma(frame)
            dark = luma < THRESHOLD
            state = "DARK  ** LED ON **" if dark else "bright   led off"
            print('{"luma": %3d, "dark": %s}  %s' % (luma, dark, state))
            time.sleep(PERIOD_S)
            n += 1
    finally:
        boardcam.release()


if __name__ == "__main__":
    run(iterations=20)
