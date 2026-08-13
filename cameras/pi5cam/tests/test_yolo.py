from ultralytics import YOLO
import cv2
import os

def test_yolo():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    # Use one of the generated test images if available, otherwise use a standard image
    image_path = "data/carandplate.jpg"
    
    if not os.path.exists(image_path):
        print(f"Test image {image_path} not found. Using 'bus.jpg' from ultralytics assets.")
        image_path = "https://ultralytics.com/images/bus.jpg"

    print(f"Running inference on {image_path}...")
    results = model(image_path)  # predict on an image

    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        
        print(f"Detected {len(boxes)} objects.")
        
        for box in boxes:
            # get box coordinates in (top, left, bottom, right) format
            b = box.xyxy[0].tolist()
            c = box.conf.item()
            cls = int(box.cls.item())
            name = model.names[cls]
            print(f"Object: {name}, Confidence: {c:.2f}, Box: {b}")
        result.save(filename="yolo_result.jpg")  # save to disk
        print("Saved result to yolo_result.jpg")

if __name__ == "__main__":
    test_yolo()
