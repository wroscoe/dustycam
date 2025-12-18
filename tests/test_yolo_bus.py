import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Ensure current dir is in path
sys.path.append(".")

from dustycam.nodes.yolo import load_yolo_model, detect_objects
from dustycam.nodes.drawing import draw_detections

def test_bus_detection():
    image_path = Path("bus.jpg")
    if not image_path.exists():
        print(f"Error: {image_path} not found.")
        sys.exit(1)
        
    print(f"Loading {image_path}...")
    image = cv2.imread(str(image_path))
    if image is None:
        print("Failed to read image.")
        sys.exit(1)
        
    print("Loading YOLO model...")
    model = load_yolo_model("yolov8n")
    if model is None:
        print("Failed to load model (ultralytics possibly missing or TFLite issue).")
        sys.exit(1)
        
    print("Running detection...")
    detections = detect_objects(image, model)
    
    print(f"Found {len(detections)} objects:")
    for d in detections:
        print(f" - {d['label']} ({d['conf']:.2f}) at {d['bbox']}")
        
    output_path = "bus_detected.jpg"
    print(f"Drawing detections and saving to {output_path}...")
    result_image = draw_detections(image, detections)
    cv2.imwrite(output_path, result_image)
    print("Done.")

if __name__ == "__main__":
    test_bus_detection()
