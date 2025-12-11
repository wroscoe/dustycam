from __future__ import annotations

import logging
import platform
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np

from dustycam.frame import FramePacket, Detection
from dustycam.node import Node

# Configure logging
logger = logging.getLogger(__name__)

def _is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    return machine.startswith("arm") or machine.startswith("aarch64")

class YoloNode(Node):
    """
    Node for Object Detection using YOLO models.
    Automatically switches backend based on hardware:
    - Desktop: Ultralytics (PyTorch)
    - Raspberry Pi: TFLite Runtime
    """

    def __init__(self, model_name: str = "yolov8n", name: Optional[str] = None):
        super().__init__(name=name or "YoloNode")
        self.model_name = model_name
        self.backend = "unknown"
        self._model = None
        self._model = None
        self._load_model()

    def get_properties(self) -> Dict:
        return {"model_name": self.model_name}

    def set_properties(self, **kwargs):
        if "model_name" in kwargs and kwargs["model_name"] != self.model_name:
            self.model_name = kwargs["model_name"]
            self._load_model()

    def _load_model(self):
        """Loads the appropriate model based on the platform."""
        if _is_raspberry_pi():
            self._load_tflite_model()
        else:
            self._load_pytorch_model()

    def _load_pytorch_model(self):
        """Loads model using Ultralytics (Desktop/GPU)."""
        try:
            from ultralytics import YOLO
            self.backend = "ultralytics"
            # Ensure .pt extension
            model_file = f"{self.model_name}.pt" if not self.model_name.endswith(".pt") else self.model_name
            logger.info(f"Loading YOLO model {model_file} with Ultralytics backend...")
            self._model = YOLO(model_file)
        except ImportError:
            logger.error("Ultralytics not installed. Cannot run YOLO on desktop.")
            raise

    def _load_tflite_model(self):
        """Loads model using TFLite Runtime (Raspberry Pi)."""
        # TODO: Implement TFLite loading
        logger.warning("TFLite backend not yet implemented. Falling back to simple placeholder or error.")
        try:
             # Fallback to pytorch if available even on Pi for now/testing? 
             # Or just strictly fail if tflite is missing?
             # For this step, let's try importing tflite_runtime, if fail, maybe try ultralytics? 
             # The plan said: "Pi Behavior: Looks for my_v8_model.tflite... Uses tflite_runtime delegate."
             # Since this is PHASE 1 of YoloNode, I will just log generic error for now.
             raise NotImplementedError("TFLite backend not implemented yet.")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise

    def forward(self, packet: FramePacket) -> FramePacket:
        if self._model is None:
            return packet

        if self.backend == "ultralytics":
            self._inference_ultralytics(packet)
        
        # TFLite inference will go here

        return packet

    def _inference_ultralytics(self, packet: FramePacket):
        """Run inference using Ultralytics API."""
        # Run inference
        # stream=True might be faster but for single image 'results' is a list
        results = self._model(packet.image, verbose=False) 

        packet.detections = [] # modify in place, but good practice to clear regular list if any
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # box.xyxy is [x1, y1, x2, y2]
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                label = self._model.names[cls_id]

                detection = Detection(
                    bbox=tuple(coords),
                    confidence=conf,
                    class_id=cls_id,
                    label=label
                )
                packet.detections.append(detection)
