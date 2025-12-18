from __future__ import annotations

import logging
import platform
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import os
# Configure logging
logger = logging.getLogger(__name__)

def _is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    return machine.startswith("arm") or machine.startswith("aarch64")

def load_yolo_model(model_name: str = "yolov8n") -> Any:
    """Loads and returns the model object."""
    from ultralytics import YOLO
    if _is_raspberry_pi():
        # TODO: TFLite implementation
        logger.warning("Raspberry Pi detected. TFLite backend not implemented yet. Falling back to Ultralytics (slow).")


    try:
        if model_name.endswith('.tflite') or model_name.endswith('.pt'):
            model_file = model_name
        else:
            model_file = f"{model_name}.pt"


        logger.info(f"Loading YOLO model {model_file}...")
        return YOLO(model_file)
    except ImportError:
        logger.error("Ultralytics not installed.")
        return None

def detect_objects(image: np.ndarray, model: Any, confidence: float = 0.5, imgsz: int = 640) -> List[Dict]:
    """
    Runs detection on image.
    Returns list of dicts: {'bbox': (x1,y1,x2,y2), 'conf': float, 'label': str, 'cls': int}
    """
    if model is None or image is None:
        return []

    # Check if it's ultralytics model (basic check)
    if hasattr(model, 'names') and hasattr(model, 'predict'):
        results = model(image, verbose=False, conf=confidence, imgsz=imgsz)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                label = model.names[cls_id]
                
                detections.append({
                    'bbox': tuple(coords),
                    'conf': conf,
                    'cls': cls_id,
                    'label': label
                })
        return detections
        
    return []

