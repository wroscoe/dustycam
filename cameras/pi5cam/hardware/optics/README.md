# Optics

Sensor + lens selection math: how far away can this camera actually resolve
what the model needs to see?

- `fov.py` — runnable calculator with presets for the supported Pi cameras.
  `python3 cameras/pi5cam/hardware/optics/fov.py`

## The one number that matters: pixels on target

Detection quality is set by how many pixels land on the object, which follows
from three choices: **sensor** (size + resolution), **lens focal length**, and
**distance**. Rules of thumb (horizontal pixels per meter of scene):

| Task | px/m needed |
|---|---|
| Detect a vehicle is present | ~25 |
| Classify vehicle type | ~60 |
| Read a license plate (good light, low motion blur) | ~150–200 |

So for the license-plate use case, run `fov.py`, find the distance at which
your sensor/lens combo drops below ~150 px/m, and that's your working range.

## Supported sensors

| Camera | Sensor | Resolution | Pixel size | Lens |
|---|---|---|---|---|
| Pi Camera Module HQ | IMX477 | 4056×3040 | 1.55 µm | C/CS-mount, interchangeable |
| Pi Camera Module 3 | IMX708 | 4608×2592 | 1.4 µm | fixed, ~4.74 mm |
| Pi Global Shutter | IMX296 | 1456×1088 | 3.45 µm | C/CS-mount; global shutter kills motion-blur skew — best for fast cars |

Not yet modeled here: motion blur vs. exposure time at target speed, and IR
cutoff / low-light behavior. Add them to `fov.py` as they come up.
