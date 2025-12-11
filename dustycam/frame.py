from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    # Bounding box in pixel coordinates: [x1, y1, x2, y2]
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: Optional[str] = None


@dataclass
class FramePacket:
    # Raw data and metadata
    frame_id: int
    timestamp: float
    image: np.ndarray

    # Analysis state
    detections: List[Detection] = field(default_factory=list)
    regions_of_interest: List[np.ndarray] = field(default_factory=list)
    ocr_results: List[str] = field(default_factory=list)

    # Control flags
    drop_frame: bool = False

    def copy_shallow(self) -> "FramePacket":
        # Zero-copy by default for image; shallow copy metadata lists
        return FramePacket(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            image=self.image,
            detections=list(self.detections),
            regions_of_interest=list(self.regions_of_interest),
            ocr_results=list(self.ocr_results),
            drop_frame=self.drop_frame,
        )
