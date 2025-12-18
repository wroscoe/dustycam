#!/usr/bin/env python3
import cv2
import time
import json
import os
import shutil
from datetime import datetime
try:
    from dustycam.nodes.sources import create_source
except ImportError:
    # Fallback for when running directly without package installation if needed
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dustycam.nodes.sources import create_source

# Defaults
motion_settings = {
    "active": True, 
    "threshold": 25,
    "min_area": 500,
    "blur_size": 21
}

SETTINGS_FILE = "motion_settings.json"

def load_settings():
    global motion_settings
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                motion_settings.update(saved)
            print(f"Loaded settings: {motion_settings}")
        except Exception as e:
            print(f"Error loading settings: {e}")
    else:
        print("Settings file not found, using defaults.")

def create_run_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("data", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save settings copy
    with open(os.path.join(run_dir, "settings.json"), 'w') as f:
        json.dump(motion_settings, f, indent=4)
        
    print(f"Created run directory: {run_dir}")
    return run_dir

def main():
    print("Starting Motion Recorder...")
    load_settings()
    run_dir = create_run_dir()
    
    try:
        source = create_source()
        print(f"Using camera source: {source}")
    except Exception as e:
        print(f"Failed to create source: {e}")
        return

    print("Camera source ready. Waiting for motion...")
    
    previous_gray = None
    frames_to_record = 0
    
    # GPIO Setup
    try:
        from gpiozero import Button
        enable_pin = Button(21)
        print("GPIO 21 initialized for conditional recording.")
        print("  - Switch CLOSED (Grounded) = ENABLED")
        print("  - Switch OPEN = DISABLED")
    except ImportError:
        print("gpiozero not found. Running in ALWAYS ENABLED mode.")
        enable_pin = None
    except Exception as e:
        print(f"GPIO Error: {e}. Running in ALWAYS ENABLED mode.")
        enable_pin = None

    try:
        while True:
            # Check GPIO Switch
            if enable_pin and not enable_pin.is_pressed:
                # Switch is Open (High) -> Disabled
                # throttle logs
                if time.time() % 5 < 0.1: 
                    print("Recording disabled (GPIO Switch OPEN)", end='\r')
                time.sleep(0.1)
                continue
            
            packet = source.next_packet()
            if packet is None:
                time.sleep(0.01)
                continue
                
            # If currently recording burst
            if frames_to_record > 0:
                filename = f"{datetime.now().strftime('%H%M%S_%f')}.jpg"
                filepath = os.path.join(run_dir, filename)
                cv2.imwrite(filepath, packet.image)
                print(f"Recorded frame: {filename}")
                frames_to_record -= 1
                if frames_to_record == 0:
                    print("Burst complete. Waiting for motion...")
                    previous_gray = None # Reset background to adapt to potentially new scene state
                continue

            # Motion Detection Logic
            try:
                gray = cv2.cvtColor(packet.image, cv2.COLOR_BGR2GRAY)
                bs = motion_settings.get("blur_size", 21)
                gray = cv2.GaussianBlur(gray, (bs, bs), 0)
                
                if previous_gray is None:
                    previous_gray = gray
                    continue
                    
                frame_delta = cv2.absdiff(previous_gray, gray)
                previous_gray = gray
                
                thresh_val = motion_settings.get("threshold", 25)
                thresh = cv2.threshold(frame_delta, thresh_val, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                motion_detected = False
                min_area = motion_settings.get("min_area", 500)
                
                for c in contours:
                    if cv2.contourArea(c) > min_area:
                        motion_detected = True
                        break
                
                if motion_detected:
                    print("Motion Detected!")
                    # Save current frame (Trigger)
                    filename = f"{datetime.now().strftime('%H%M%S_%f')}_TRIGGER.jpg"
                    filepath = os.path.join(run_dir, filename)
                    cv2.imwrite(filepath, packet.image)
                    print(f"Recorded trigger frame: {filename}")
                    
                    # Record next 2 frames
                    frames_to_record = 2
                    
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()
