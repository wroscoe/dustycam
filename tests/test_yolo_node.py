
import os
import time
import pytest
import numpy as np
import cv2
from dustycam.frame import FramePacket
from dustycam.nodes.detectors.yolo import YoloNode

# Check if we are potentially on a CI/headless env where we might skip heavy inference tests if needed
# But for this task, we want to run it.

def get_test_image():
    # Attempt to load a real image, or create a dummy one if network/file missing
    img_path = "data/carandplate.jpg"
    if os.path.exists(img_path):
        return cv2.imread(img_path)
    
    # Fallback: create a dummy image with a square (simple shape might not be detected by YOLO but ensures code runs)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    return img

def test_yolo_node_desktop_run():
    """Verify YoloNode loads and runs on desktop without crashing."""
    try:
        from ultralytics import YOLO
    except ImportError:
        pytest.skip("Ultralytics not installed, skipping desktop YOLO test")

    node = YoloNode(model_name="yolov8n")
    
    # Create packet
    img = get_test_image()
    packet = FramePacket(
        frame_id=1,
        timestamp=time.time(),
        image=img
    )

    # Run inference
    result_packet = node.forward(packet)

    # Check that detections list is initialized (might be empty if dummy image)
    assert isinstance(result_packet.detections, list)
    
    # If we used the real car image, we expect some detections (car, license plate etc)
    # But strictly, we just verified it ran.
    # Let's inspect contents if possible
    print(f"Detections found: {len(result_packet.detections)}")
    for d in result_packet.detections:
        print(d)

if __name__ == "__main__":
    test_yolo_node_desktop_run()
