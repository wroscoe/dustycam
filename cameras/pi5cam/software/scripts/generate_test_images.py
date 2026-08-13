import cv2
import numpy as np
import os

output_dir = os.path.expanduser("~/dustycam_test_input")
os.makedirs(output_dir, exist_ok=True)

# Create a background
width, height = 640, 480
background = np.zeros((height, width, 3), dtype=np.uint8)
# Add some static noise/features
cv2.circle(background, (100, 100), 50, (255, 0, 0), -1)
cv2.rectangle(background, (400, 300), (500, 400), (0, 255, 0), -1)

# Generate frames with a moving object
num_frames = 10
for i in range(num_frames):
    frame = background.copy()
    # Moving circle
    x = 200 + i * 20
    y = 240
    cv2.circle(frame, (x, y), 30, (0, 0, 255), -1)
    
    filename = os.path.join(output_dir, f"frame_{i:03d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Generated {filename}")

print("Done generating test images.")
