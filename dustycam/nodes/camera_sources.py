from __future__ import annotations

import cv2
import numpy as np
import time
from typing import Optional

class OpenCVSource:
    """Webcam capture wrapper."""
    def __init__(self, camera_index: int = 0):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
             print(f"Warning: Could not open camera {camera_index}")

    def read(self) -> Optional[np.ndarray]:
        if not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        if self._cap:
            self._cap.release()

class Picamera2Source:
    """Picamera2 wrapper."""
    def __init__(self, size=(320, 240)):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise RuntimeError("Picamera2 not found")
            
        self._picam2 = Picamera2()
        high_res = (2028, 1520)
        config = self._picam2.create_video_configuration(
            main={"size": high_res, "format": "RGB888"},
            lores={"size": size, "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()

    def read(self) -> Optional[np.ndarray]:
        return self._picam2.capture_array("lores")

    def take_photo(self) -> Optional[np.ndarray]:
        try:
            return self._picam2.capture_array("main")
        except:
            return None

    def close(self):
        if self._picam2:
            self._picam2.stop()
