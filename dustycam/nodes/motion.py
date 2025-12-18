import cv2
import numpy as np

def detect_motion(image: np.ndarray, previous_gray: np.ndarray = None, threshold: int = 25, min_area: int = 500, max_area: int = None, blur_size: int = 21):
    """
    Detects motion in an image compared to a previous frame.
    Returns: (processed_image, motion_detected_bool, current_gray)
    """
    if image is None:
        return image, False, previous_gray

    
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    if previous_gray is None:
        return image, False, gray
        
    frame_delta = cv2.absdiff(previous_gray, gray)
    
    thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_with_detections = image.copy()

    motion_detected = False
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    if motion_detected:
        cv2.putText(image_with_detections, "Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    return image_with_detections, motion_detected, gray

