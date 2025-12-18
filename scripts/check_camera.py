
import cv2
import time
import platform
import sys

def check_picamera2():
    print("--- Checking Picamera2 ---")
    try:
        from picamera2 import Picamera2
        print("Import successful.")
        try:
            picam2 = Picamera2()
            print("Picamera2 instantiated.")
            config = picam2.create_video_configuration(main={"size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            print("Picamera2 started.")
            # frame = picam2.capture_array()
            # print(f"Captured frame shape: {frame.shape}")
            picam2.stop()
            print("Picamera2 stopped.")
        except Exception as e:
            print(f"Picamera2 init/start failed: {e}")
            import traceback
            traceback.print_exc()
    except ImportError as e:
        print(f"Picamera2 import failed: {e}")

def check_opencv():
    print("\n--- Checking OpenCV ---")
    verified_indices = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Index {i}: Opened.", end=" ")
            ret, frame = cap.read()
            if ret:
                print(f"Read success. Shape: {frame.shape}")
                verified_indices.append(i)
            else:
                print("Read failed.")
            cap.release()
        else:
            pass
            # print(f"Index {i}: Failed to open.")
            
    if not verified_indices:
        print("No working OpenCV camera indices found in range 0-9.")
    else:
        print(f"Working OpenCV indices: {verified_indices}")

if __name__ == "__main__":
    print(f"Platform: {platform.machine()}")
    check_picamera2()
    check_opencv()
