
import cv2
import numpy as np

from dustycam.frame import FramePacket
from dustycam.node import Node

class DrawDetectionsNode(Node):
    """
    Draws bounding boxes and labels for any detections found in the packet.
    """
    def __init__(self, name: str = "DrawDetectionsNode"):
        super().__init__(name=name)

    def forward(self, packet: FramePacket) -> FramePacket:
        if not packet.detections or packet.image is None:
            return packet

        # Make a copy if you don't want to modify the original image in the packet
        # But for pipeline efficiency, we usually modify in-place if no fan-out needs the clean original.
        # Let's assume in-place for now, or copy if safer.
        # packet.image is a numpy array.
        
        # We'll draw directly on the image.
        img = packet.image
        
        for det in packet.detections:
            x1, y1, x2, y2 = det.bbox
            conf = det.confidence
            cls = det.class_id
            label = det.label or f"Class {cls}"
            
            # Draw box
            # Green color (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 8)
            
            # Draw label
            text = f"{label} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size
            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background for text
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
            
        return packet
