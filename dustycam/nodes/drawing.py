import cv2
import numpy as np
from typing import List, Dict

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draws bounding boxes and labels for any detections found.
    Detections should be a list of dicts with keys: 'bbox', 'conf', 'label', 'cls'.
    """
    if not detections or image is None:
        return image

    # We'll draw directly on the image (in-place modification is standard for opencv drawing)
    
    for det in detections:
        x1, y1, x2, y2 = det.get('bbox', (0,0,0,0))
        conf = det.get('conf', 0.0)
        label = det.get('label', 'Unknown')
        
        # Draw box
        # Green color (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background for text
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
    return image
