from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

from dustycam.frame import FramePacket
from dustycam.node import SourceNode


class BaseSource(SourceNode):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self._frame_id = 0

    def forward(self, packet: FramePacket) -> FramePacket:
        """Sources just pass packets downstream after caching."""
        self._set_cache(packet)
        return packet

    def _next_frame_id(self) -> int:
        fid = self._frame_id
        self._frame_id += 1
        return fid


class OpenCVSource(BaseSource):
    """Webcam capture using cv2.VideoCapture."""

    def __init__(self, camera_index: int = 0, name: Optional[str] = None):
        super().__init__(name=name or "OpenCVSource")
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

    def next_packet(self) -> Optional[FramePacket]:
        ok, frame = self._cap.read()
        if not ok:
            return None
        return FramePacket(
            frame_id=self._next_frame_id(),
            timestamp=time.time(),
            image=frame,
        )


class FileSource(BaseSource):
    """Reads images from a directory in sorted order."""

    def __init__(self, directory: Path | str, loop: bool = False, name: Optional[str] = None):
        super().__init__(name=name or "FileSource")
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.directory}")
        self.loop = loop
        self._files = self._list_images(self.directory)
        self._idx = 0

    def _list_images(self, directory: Path) -> List[Path]:
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files: List[Path] = []
        for pattern in patterns:
            files.extend(directory.glob(pattern))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"No images found in {directory}")
        return files

    def next_packet(self) -> Optional[FramePacket]:
        if self._idx >= len(self._files):
            if not self.loop:
                return None
            self._idx = 0
        path = self._files[self._idx]
        self._idx += 1
        frame = cv2.imread(str(path))
        if frame is None:
            # Skip unreadable files
            return self.next_packet()
        return FramePacket(
            frame_id=self._next_frame_id(),
            timestamp=time.time(),
            image=frame,
        )


class Picamera2Source(BaseSource):
    """Capture from Raspberry Pi camera using Picamera2."""

    def __init__(self, size: tuple[int, int] = (640, 480), name: Optional[str] = None):
        super().__init__(name=name or "Picamera2Source")
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError("Picamera2 is not available on this system") from exc
        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(main={"size": size})
        self._picam2.configure(config)
        self._picam2.start()

    def next_packet(self) -> Optional[FramePacket]:
        frame = self._picam2.capture_array()
        return FramePacket(
            frame_id=self._next_frame_id(),
            timestamp=time.time(),
            image=frame,
        )



class BoxWithTextSource(BaseSource):
    """
    Synthetic source that generates frames with a moving box containing text.
    Useful for testing detection without a physical camera.
    """
    def __init__(self, text: str = "DustyCam", size: tuple[int, int] = (640, 480), 
                 box_size: tuple[int, int] = (100, 50), velocity: tuple[int, int] = (5, 5),
                 name: Optional[str] = None):
        super().__init__(name=name or "BoxWithTextSource")
        self.text = text
        self.width, self.height = size
        self.box_w, self.box_h = box_size
        self.vx, self.vy = velocity
        
        # Initial position (center)
        self.x = (self.width - self.box_w) // 2
        self.y = (self.height - self.box_h) // 2

    def next_packet(self) -> Optional[FramePacket]:
        # Create black background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off edges
        if self.x <= 0 or self.x + self.box_w >= self.width:
            self.vx *= -1
            self.x = max(0, min(self.x, self.width - self.box_w))
            
        if self.y <= 0 or self.y + self.box_h >= self.height:
            self.vy *= -1
            self.y = max(0, min(self.y, self.height - self.box_h))
            
        # Draw white box
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.box_w, self.y + self.box_h), (255, 255, 255), -1)
        
        # Draw text centered in box
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(self.text, font, text_scale, thickness)
        
        text_x = self.x + (self.box_w - text_w) // 2
        text_y = self.y + (self.box_h + text_h) // 2
        
        cv2.putText(frame, self.text, (text_x, text_y), font, text_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
        return FramePacket(
            frame_id=self._next_frame_id(),
            timestamp=time.time(),
            image=frame,
        )


class BouncingBallSource(BaseSource):
    """
    Synthetic source that generates frames with a moving ball.
    """
    def __init__(self, radius: int = 30, color: tuple[int, int, int] = (0, 0, 255),
                 size: tuple[int, int] = (640, 480), velocity: tuple[int, int] = (6, 6),
                 name: Optional[str] = None):
        super().__init__(name=name or "BouncingBallSource")
        self.radius = radius
        self.color = color  # BGR
        self.width, self.height = size
        self.vx, self.vy = velocity
        
        self.x = self.width // 2
        self.y = self.height // 2

    def get_properties(self) -> Dict:
        return {
            "radius": self.radius,
            "color": list(self.color),
            "vx": self.vx,
            "vy": self.vy
        }

    def set_properties(self, **kwargs):
        if "radius" in kwargs:
            self.radius = int(kwargs["radius"])
        if "color" in kwargs:
            self.color = tuple(kwargs["color"])
        if "vx" in kwargs:
            self.vx = int(kwargs["vx"])
        if "vy" in kwargs:
            self.vy = int(kwargs["vy"])

    def next_packet(self) -> Optional[FramePacket]:
        # Create black background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off edges (considering radius)
        if self.x - self.radius <= 0 or self.x + self.radius >= self.width:
            self.vx *= -1
            self.x = max(self.radius, min(self.x, self.width - self.radius))
            
        if self.y - self.radius <= 0 or self.y + self.radius >= self.height:
            self.vy *= -1
            self.y = max(self.radius, min(self.y, self.height - self.radius))
            
        # Draw filled circle
        cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)
        
        return FramePacket(
            frame_id=self._next_frame_id(),
            timestamp=time.time(),
            image=frame,
        )


def _is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    return machine.startswith("arm") or machine.startswith("aarch64")


def create_source(
    preferred: Optional[str] = None,
    file_dir: Optional[Path | str] = None,
    camera_index: int = 0,
    picam_size: tuple[int, int] = (640, 480),
) -> BaseSource:
    """
    Factory that chooses a source based on availability and preference.
    preferred: 'picamera2', 'opencv', or 'file'
    """
    preference_order: List[str] = []
    if preferred:
        preference_order.append(preferred.lower())
    if not preference_order:
        if _is_raspberry_pi():
            preference_order.append("picamera2")
        if file_dir:
            preference_order.append("file")
        preference_order.append("opencv")
    for pref in preference_order:
        try:
            if pref == "file" and file_dir:
                return FileSource(directory=file_dir)
            if pref == "picamera2":
                return Picamera2Source(size=picam_size)
            if pref == "opencv":
                return OpenCVSource(camera_index=camera_index)
        except Exception:
            continue
    raise RuntimeError(f"Could not create a source with preferences: {preference_order}")
