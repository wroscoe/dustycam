"""Field-of-view and pixels-on-target math for camera + lens selection.

Thin-lens geometry, objects at distance >> focal length:
    scene width covered at distance d = d * sensor_width / focal_length
Run: python3 hardware/optics/fov.py
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Sensor:
    name: str
    width_px: int
    height_px: int
    pixel_um: float  # pixel pitch in micrometers

    @property
    def width_mm(self) -> float:
        return self.width_px * self.pixel_um / 1000

    @property
    def height_mm(self) -> float:
        return self.height_px * self.pixel_um / 1000


# The Pi cameras the build guides support.
IMX477_HQ = Sensor("Pi HQ (IMX477)", 4056, 3040, 1.55)
IMX708_MODULE3 = Sensor("Pi Module 3 (IMX708)", 4608, 2592, 1.4)
IMX296_GS = Sensor("Pi Global Shutter (IMX296)", 1456, 1088, 3.45)

# XIAO ESP32-S3 Sense camera modules (fixed lens, no M12/C-mount).
OV2640_XIAO = Sensor("XIAO OV2640", 1600, 1200, 2.2)
OV5640_XIAO = Sensor("XIAO OV5640 (AF swap)", 2592, 1944, 1.4)

# Pi AI Camera: IMX500 with on-sensor NPU; detection runs on the sensor and
# full-res stills are still available to the host (10 fps at 12 MP).
IMX500_AI = Sensor("Pi AI Camera (IMX500)", 4056, 3040, 1.55)

# OpenMV cams: M12 interchangeable lenses (H7 Plus rolling shutter 5MP;
# N6/AE3 ship the PAG7936 1MP global shutter, N6 has a 600 GOPS NPU).
OV5640_H7PLUS = Sensor("OpenMV H7 Plus (OV5640)", 2592, 1944, 1.4)
PAG7936_N6 = Sensor("OpenMV N6/AE3 (PAG7936)", 1288, 808, 3.0)

MODULE3_FOCAL_MM = 4.74  # Module 3 lens is fixed
OV2640_FOCAL_MM = 2.7    # stock ~66 deg HFOV
OV5640_FOCAL_MM = 3.4    # typical fixed-lens FPC module
IMX500_FOCAL_MM = 4.74   # fixed, manual-focus; 78 deg diagonal FOV

# Horizontal pixels-per-meter rules of thumb (see README).
PXM_DETECT_VEHICLE = 25
PXM_READ_PLATE = 150


def hfov_deg(sensor: Sensor, focal_mm: float) -> float:
    return math.degrees(2 * math.atan(sensor.width_mm / (2 * focal_mm)))


def scene_width_m(sensor: Sensor, focal_mm: float, distance_m: float) -> float:
    return distance_m * sensor.width_mm / focal_mm


def pixels_per_meter(sensor: Sensor, focal_mm: float, distance_m: float) -> float:
    return sensor.width_px / scene_width_m(sensor, focal_mm, distance_m)


def max_distance_m(sensor: Sensor, focal_mm: float, required_pxm: float) -> float:
    """Farthest distance at which the combo still delivers required_pxm."""
    return sensor.width_px * focal_mm / (required_pxm * sensor.width_mm)


def min_focal_mm(sensor: Sensor, distance_m: float, required_pxm: float) -> float:
    """Shortest lens that delivers required_pxm at distance_m."""
    return required_pxm * distance_m * sensor.width_mm / sensor.width_px


if __name__ == "__main__":
    combos = [
        (IMX477_HQ, 6.0),
        (IMX477_HQ, 16.0),
        (IMX708_MODULE3, MODULE3_FOCAL_MM),
        (IMX296_GS, 6.0),
        (IMX296_GS, 16.0),
        (OV2640_XIAO, OV2640_FOCAL_MM),
        (OV5640_XIAO, OV5640_FOCAL_MM),
        (IMX500_AI, IMX500_FOCAL_MM),
        (OV5640_H7PLUS, 8.0),
        (PAG7936_N6, 8.0),
        (PAG7936_N6, 16.0),
    ]
    print(f"{'camera + lens':38} {'HFOV':>6} {'px/m @30m':>10} "
          f"{'plate range':>12} {'vehicle range':>14}")
    for sensor, f in combos:
        print(f"{sensor.name + f' + {f:g}mm':38} "
              f"{hfov_deg(sensor, f):5.1f}° "
              f"{pixels_per_meter(sensor, f, 30):10.0f} "
              f"{max_distance_m(sensor, f, PXM_READ_PLATE):10.1f} m "
              f"{max_distance_m(sensor, f, PXM_DETECT_VEHICLE):12.1f} m")
    print(f"\nplate range = distance where px/m falls below {PXM_READ_PLATE} "
          f"(license plate readable)")
    print(f"vehicle range = distance where px/m falls below {PXM_DETECT_VEHICLE} "
          f"(vehicle detectable)")
