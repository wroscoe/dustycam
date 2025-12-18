import time
import threading
import traceback
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Type, Tuple
from collections import deque


from ultralytics import YOLO



class Pipeline:
    """
    Base Pipeline class.
    Subclasses should implement the `run_loop` method.
    """
    def __init__(self, name: str = "Pipeline", settings_model: Type[BaseModel] = None):
        self.name = name
        self.settings_model = settings_model
        
        # Shared state for external access (e.g., WebApp)
        self.previews: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        
        # Settings persistence
        self.settings_dir = Path(os.path.expanduser("~/.dustycam/settings"))
        self.settings_file = self.settings_dir / f"{self.name}.json"
        
        # Init settings
        if self.settings_model:
            self.settings = self._load_settings()
        else:
            self.settings = {} # Fallback if no model provided

        
        # Control flags
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _load_settings(self) -> Any:
        """Load settings from disk if available, else return default model."""
        if not self.settings_model:
            return {}

        initial_settings = self.settings_model() # Defaults
        
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    saved_data = json.load(f)
                    # Validate and merge
                    initial_settings = self.settings_model(**saved_data)
                print(f"Loaded settings for '{self.name}': {initial_settings.model_dump()}")
            except Exception as e:
                print(f"Error loading settings for '{self.name}': {e}")
                # Fallback to defaults -> save them
                self._save_settings(initial_settings)
        else:
             # Ensure defaults are saved if new
             self._save_settings(initial_settings)
             
        return initial_settings

    def _save_settings(self, settings_to_save: Any = None):
        """Save current (or provided) settings to disk."""
        try:
            target_settings = settings_to_save or self.settings
            self.settings_dir.mkdir(parents=True, exist_ok=True)
            with open(self.settings_file, 'w') as f:
                if isinstance(target_settings, BaseModel):
                    # Pydantic model
                    f.write(target_settings.model_dump_json(indent=4))
                else:
                    # Dict fallback
                    json.dump(target_settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings for '{self.name}': {e}")

    def update_settings(self, new_settings: Dict[str, Any], restart: bool = False):
        """
        Update settings and persist to disk.
        If restart is True, the pipeline will be stopped and restarted.
        """
        if self.settings_model and isinstance(self.settings, BaseModel):
            # Update Pydantic model
            try:
                # Merge current values with new ones and validate
                current_data = self.settings.model_dump()
                current_data.update(new_settings)
                updated = self.settings_model.model_validate(current_data)
                self.settings = updated
            except Exception as e:
                print(f"Validation error updating settings: {e}")
                # We log and abort update on validation failure
                return
        else:
            # Dict fallback
            self.settings.update(new_settings)
            
        self._save_settings()
        
        if restart and self.running:
            print(f"Restarting pipeline '{self.name}' due to settings update...")
            self.stop()
            self.start()

    def get_setting(self, key: str, default: Any = None) -> Any:
        if isinstance(self.settings, BaseModel):
            return getattr(self.settings, key, default)
        return self.settings.get(key, default)


    def start(self):
        with self._lock:
            if self.running:
                return
            self.running = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            print(f"Pipeline '{self.name}' started.")

    def stop(self):
        with self._lock:
            self.running = False
        if self._thread:
            self._thread.join()
            self._thread = None
        print(f"Pipeline '{self.name}' stopped.")

    def _worker(self):
        try:
            self.run_loop()
        except Exception:
            traceback.print_exc()
            self.running = False

    def run_loop(self):
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement run_loop")

    def set_preview(self, name: str, frame: Any):
        """Update a preview image (numpy array)."""
        self.previews[name] = frame

    def get_preview(self, name: str) -> Any:
        return self.previews.get(name)


class PiMotionPipeline(Pipeline):
    class Settings(BaseModel):
        motion_min_area: int = Field(50, json_schema_extra={"group": "Detection", "min": 1, "max": 10000})
        motion_threshold: int = Field(25, json_schema_extra={"group": "Detection", "min": 0, "max": 255})
        motion_blur_size: int = Field(21, json_schema_extra={"group": "Preprocessing", "min": 1, "max": 51})
        motion_max_area: int = Field(500, json_schema_extra={"group": "Detection", "min": 1, "max": 10000})
        
        # Pipeline Control
        enable_motion: bool = Field(True, json_schema_extra={"group": "Pipeline Control"})
        enable_recognition: bool = Field(True, json_schema_extra={"group": "Pipeline Control"})
        enable_recording: bool = Field(True, json_schema_extra={"group": "Pipeline Control"})

        # Recording Filters
        record_all: bool = Field(True, json_schema_extra={"group": "Recording Filters"})
        record_person: bool = Field(True, json_schema_extra={"group": "Recording Filters"})
        record_car: bool = Field(True, json_schema_extra={"group": "Recording Filters"})
        record_motorbike: bool = Field(True, json_schema_extra={"group": "Recording Filters"})
        record_bus: bool = Field(True, json_schema_extra={"group": "Recording Filters"})
        record_train: bool = Field(True, json_schema_extra={"group": "Recording Filters"})
        record_truck: bool = Field(True, json_schema_extra={"group": "Recording Filters"})
        record_bicycle: bool = Field(True, json_schema_extra={"group": "Recording Filters"})


    def __init__(self):

        super().__init__(name="pi_motion", settings_model=self.Settings)

        self.previews = {'Source': None, 'Detections': None}

        self.low_res_frame_buffer_size = 3
        self.low_res_frame_buffer = deque(maxlen=self.low_res_frame_buffer_size)

        # Metrics (Sliding window of timestamps)
        self.metrics = {
            'motion': deque(),
            'recognition': deque(),
            'recording': deque()
        }

    def _record_metric(self, name: str):
        self.metrics[name].append(time.time())

    def _prune_metrics(self):
        now = time.time()
        for key in self.metrics:
            d = self.metrics[key]
            while len(d) > 0 and (now - d[0] > 60):
                d.popleft()


        
    def run_loop(self):
        from dustycam.nodes.camera_sources import Picamera2Source
        from dustycam.nodes.motion import detect_motion
        from dustycam.nodes.yolo import load_yolo_model, detect_objects
        from dustycam.nodes.drawing import draw_detections
        import cv2
        from dustycam.nodes.drawing import draw_detections
        import cv2
        from gpiozero import Button, LED

        # GPIO Control
        # Pull Up: True -> Default High (Open) -> is_pressed=False
        # Grounded -> Low -> is_pressed=True
        enable_switch = Button(20, pull_up=True)
        
        # Status LED (ACT is the green activity LED on Pi)
        status_led = None
        try:
            status_led = LED("ACT")
        except Exception as e:
            print(f"Warning: Could not access ACT LED: {e}")

        was_enabled = False

        # Create run directory
        run_timestamp = int(time.time())
        run_dir = Path("data") / f"run_{run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving images to {run_dir.absolute()}")

        source = Picamera2Source()
        previous_gray = None
        model = load_yolo_model("yolov8n_int8_320.tflite")
        previous_gray = None
        model = load_yolo_model("yolov8n_int8_320.tflite")
        
        # Base set of classes we care about (for 'Record All' or individual selection)
        supported_classes = ['person', 'car', 'motorbike', 'bus', 'train', 'truck', 'bicycle']

        # Init tracking vars
        last_time = time.time()
        frame_ctr = 0
        fps = 0.0
        hires_count = 0

        try:
            while self.running:
                image = source.read()
                self.set_preview("Source", image)
                self.low_res_frame_buffer.append(image)

                if image is None:
                    time.sleep(0.01)
                    continue

                # Pipeline Control Settings
                do_motion = self.settings.enable_motion
                do_recognition = self.settings.enable_recognition
                do_recording = self.settings.enable_recording

                # Update State
                is_gpio_active = enable_switch.is_pressed
                self.state['gpio_active'] = is_gpio_active
                self.state['recording_enabled'] = do_recording
                
                # Handle LED state (Based on recording setting)
                if status_led:
                    if do_recording and not was_enabled:
                        status_led.blink(on_time=1, off_time=1)
                    elif not do_recording and was_enabled:
                        status_led.off()
                was_enabled = do_recording


                # Motion Detection Stage
                motion_detected = False
                processed_image = image.copy()
                
                if do_motion:
                    threshold = self.settings.motion_threshold
                    blur_size = self.settings.motion_blur_size
                    min_area = self.settings.motion_min_area
                    max_area = self.settings.motion_max_area
                    processed_image, motion_detected, previous_gray = detect_motion(image, previous_gray, threshold=threshold, blur_size=blur_size, min_area=min_area, max_area=max_area)
                
                if motion_detected:
                    self._record_metric('motion')

                self.state['motion_detected'] = motion_detected
                
                # Update preview with motion boxes (or raw image if no motion check)
                self.set_preview("Source", processed_image)

                # FPS Calculation
                frame_ctr += 1
                curr_time = time.time()
                if curr_time - last_time > 1.0:
                    fps = frame_ctr / (curr_time - last_time)
                    frame_ctr = 0
                    last_time = curr_time
                
                # Metrics Maintenance
                self._prune_metrics()
                
                self.state['fps'] = round(fps, 1)
                self.state['hires_count'] = hires_count
                self.state['motion_events_1min'] = len(self.metrics['motion'])
                self.state['recognition_events_1min'] = len(self.metrics['recognition'])
                self.state['recording_events_1min'] = len(self.metrics['recording'])


                # Processing & Recording
                if motion_detected:
                    detections = []
                    
                    # Recognition Stage
                    if do_recognition:
                        self._record_metric('recognition')
                        detections = detect_objects(image, model, confidence=0.5, imgsz=320)
                        
                    # Filter for target classes if we have detections
                    has_relevant_detections = False
                    if len(detections) > 0:
                        detected = []
                        
                        # Determine active targets based on settings
                        current_targets = []
                        if self.settings.record_all:
                            current_targets = supported_classes
                        else:
                            if self.settings.record_person: current_targets.append('person')
                            if self.settings.record_car: current_targets.append('car')
                            if self.settings.record_motorbike: current_targets.append('motorbike')
                            if self.settings.record_bus: current_targets.append('bus')
                            if self.settings.record_train: current_targets.append('train')
                            if self.settings.record_truck: current_targets.append('truck')
                            if self.settings.record_bicycle: current_targets.append('bicycle')


                        print(detections)
                        for detection in detections:
                            if detection['label'] in current_targets:
                                detected.append(detection)
                        if len(detected) > 0:
                            has_relevant_detections = True

                    # Recording Logic
                    if do_recording:
                        if has_relevant_detections:
                            print('saving hi res image')
                            hi_res_image = source.take_photo()
                            if hi_res_image is not None:
                                ## Save hi res image
                                cv2.imwrite(str(run_dir / f"hires_{int(time.time())}.jpg"), hi_res_image)
                                hires_count += 1
                                self._record_metric('recording')
                        elif do_recognition and not has_relevant_detections:
                             # YOLO ran but didn't find targets -> Treat as false positive if specific recognition was requested
                             pass # Don't save false positives if we are depending on recognition? 
                             # Wait, logic before was: if detections > 0 (and filtered > 0) -> save hires. Else -> save false motion.
                             # Let's preserve that logic but gated by do_recording.
                             
                             print('saving false motion image')
                             for i,  frame in enumerate(self.low_res_frame_buffer):
                                 cv2.imwrite(str(run_dir / f"false_motion_{int(time.time())}_{i}.jpg"), frame)
                             self.low_res_frame_buffer.clear()
                        elif not do_recognition:
                             # No recognition enabled, but motion detected and recording enabled.
                             # Save generic motion event? Or treat as "nothing confirmed"?
                             # Previous logic relied on YOLO to decide HiRes vs FalseMotion.
                             # If YOLO is OFF, we probably just want to save... something?
                             # Let's assume if Recognition is OFF, we treat ANY motion as worthy of a "motion capture" (maybe low res buffer?)
                             # For now, let's just default to saving the Low Res buffer if only Motion is on.
                             print('saving motion sequence (no recognition)')
                             for i, frame in enumerate(self.low_res_frame_buffer):
                                 cv2.imwrite(str(run_dir / f"motion_{int(time.time())}_{i}.jpg"), frame)
                             self.low_res_frame_buffer.clear()



                
        finally:
            if status_led:
                status_led.close()
            source.close()


class BallYoloPipeline(Pipeline):
    class Settings(BaseModel):
        confidence: float = Field(0.5, json_schema_extra={"group": "Detection", "min": 0.0, "max": 1.0, "step": 0.05})
        draw_labels: bool = Field(True, json_schema_extra={"group": "Drawing"})

    def __init__(self):
        super().__init__(name="ball_yolo", settings_model=self.Settings)

    def run_loop(self):
        from dustycam.nodes.generated_sources import BouncingBall
        from dustycam.nodes.yolo import load_yolo_model, detect_objects
        import cv2
        from dustycam.nodes.drawing import draw_detections

        source = BouncingBall()
        model = load_yolo_model("yolov8n")
        detections = detect_objects(image, model, confidence=conf)


        while self.running:
            image = source.next_frame()
            if image is None:
                time.sleep(0.01)
                continue
            
            # Pure function detection
            conf = self.settings.confidence
            detections = detect_objects(image, model, confidence=conf)
            
            # Use refactored drawing function
            image = draw_detections(image, detections)

            self.set_preview("Main", image)
            time.sleep(0.03)


class SyntheticMotionPipeline(Pipeline):
    class Settings(BaseModel):
        threshold: int = Field(25, json_schema_extra={"group": "Detection", "min": 0, "max": 255})
        min_area: int = Field(500, json_schema_extra={"group": "Detection", "min": 100, "max": 5000})

    def __init__(self):
        super().__init__(name="synthetic_motion", settings_model=self.Settings)

    def run_loop(self):
        from dustycam.nodes.generated_sources import BouncingBall
        from dustycam.nodes.motion import detect_motion

        source = BouncingBall()
        previous_gray = None
        
        while self.running:
            image = source.next_frame()
            if image is None:
                time.sleep(0.01)
                continue
                
            thresh = self.settings.threshold
            min_a = self.settings.min_area
            processed_image, detected, previous_gray = detect_motion(image, previous_gray, threshold=thresh, min_area=min_a)
            
            self.set_preview("Main", processed_image)
            self.state['motion_detected'] = detected
            time.sleep(0.03)


# Factory / Registry
    
_PIPELINE_CLASSES: Dict[str, Type[Pipeline]] = {}

def register_pipeline(name: str, cls: Type[Pipeline]):
    _PIPELINE_CLASSES[name] = cls

def get_pipeline_by_name(name: str) -> Optional[Pipeline]:
    cls = _PIPELINE_CLASSES.get(name)
    if cls:
        return cls()

    return None

# Register them
register_pipeline("default", PiMotionPipeline)
register_pipeline("pi_motion", PiMotionPipeline)
register_pipeline("ball_yolo", BallYoloPipeline)
register_pipeline("synthetic_motion", SyntheticMotionPipeline)