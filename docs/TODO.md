This document serves as the architectural blueprint for the OpenLPR / TrailCam Pipeline Project. It outlines the core design decisions required to balance ease of development (Desktop) with high-efficiency deployment (Raspberry Pi).

1. Project Vision
To create a Python-based computer vision framework where a user can define a processing pipeline once, and execute it efficiently on both a high-end Desktop (for debugging/tinkering) and a Raspberry Pi (for deployment), with the framework automatically handling hardware acceleration and resource constraints.

2. Core Architectural Concepts
2.1 The Atomic Unit: The FramePacket
Constraint: Do not pass raw NumPy arrays between functions. Solution: We pass a state object called a FramePacket.

Why: We need to carry metadata (timestamps, detections, crop coordinates) alongside the image without modifying function signatures every time we add a feature.

Structure:

Python

@dataclass
class FramePacket:
    # The raw data
    frame_id: int
    timestamp: float
    image: np.ndarray  # The current image state

    # The Analysis State
    detections: List[Detection] # Standardized objects (bbox, class, conf)
    regions_of_interest: List[np.ndarray] # e.g., Cropped license plates
    ocr_results: List[str]

    # Control Flags
    drop_frame: bool = False # If True, Sinks ignore this packet
2.2 The Structure: Event-Driven DAG (Directed Acyclic Graph)
Constraint: A simple list [A, B, C] is insufficient for branching logic (e.g., "If Motion -> Detect. If No Motion -> Sleep"). Solution: A lightweight DAG runner.

Nodes: Independent units of logic (e.g., MotionDetector, YoloInference).

Edges: Data dependency (e.g., LPR_Reader requires output from Plate_Cropper).

Lazy Execution: Nodes only run if their output is requested by a Sink or a downstream node.

3. The Abstraction Layers (The "Must-Get-Rights")
To achieve the "Write Once, Run Anywhere" goal, we must abstract IO and Inference.

3.1 Source Abstraction (Input)
We need a Source Factory that detects the environment. The pipeline script simply asks for CameraSource().

Desktop: Falls back to OpenCV VideoCapture(0) or FileSource (folder of images).

Raspberry Pi: Automatically loads libcamera / Picamera2 for zero-copy access to the CSI bus. Crucial for performance.

3.2 Inference Abstraction (The "Smart" Layer)
This is the most critical component. We cannot run raw PyTorch (.pt) files efficiently on a Pi.

Strategy: The YoloNode must accept a model name (e.g., "my_v8_model") and intelligently select the backend.

Desktop Behavior: Loads my_v8_model.pt using Ultralytics/PyTorch. Uses CUDA if available.

Pi Behavior: Looks for my_v8_model.tflite (INT8 quantized). Uses tflite_runtime delegate.

Developer Requirement: The framework must raise a helpful error if the .tflite model is missing on the Pi, prompting the user to run the export script.

3.3 Sink Abstraction (Output)
ScreenSink:

Desktop: Uses cv2.imshow (High GUI).

Pi (Headless): Starts a lightweight Flask/MJPEG server on Port 5000 so the user can debug via a browser.

StorageSink: Saves JSON/CSV logs and images. Must run in a separate thread to prevent disk I/O from blocking the camera capture loop.

4. Development Workflow
We are building a tool that dictates this workflow for our users:

Tinker (Desktop): User writes pipeline.py. Uses TestFileSource to cycle through folder of driveway images. Adjusts thresholds using full PyTorch models.

Export (Desktop): User runs framework.export_model("my_model.pt"). We wrap the Ultralytics export script to generate the quantization-ready .tflite file.

Deploy (Pi): User scps the script and the .tflite file to the Pi.

Run (Pi): User runs python pipeline.py. The framework detects arm64, swaps cv2.VideoCapture for Picamera2, and swaps PyTorch for TFLite.

5. Implementation Roadmap
Phase 1: The Skeleton (v0.1)
Implement FramePacket.

Implement Node base class with connect() and forward() methods.

Implement a simple topological sort Runner.

Goal: A script that passes a dummy string from Node A to Node B.

Phase 2: The Hardware Abstraction (v0.2)
Implement Source Interface.

Create OpenCVSource (Webcam).

Create FileSource (Test images).

Goal: A pipeline that displays webcam feed or images on a desktop window.

Phase 3: The Intelligence (v0.3)
Implement YoloNode.

Add logic to auto-switch between YOLO() (Ultralytics) and Interpreter() (TFLite).

Strict Rule: Standardize the output format of this node. Bounding boxes must always be [x1, y1, x2, y2, confidence, class_id], normalized or pixel-based (pick one and stick to it).

Goal: Running object detection on Desktop (Torch) and Pi (TFLite) using the exact same user script.

Phase 4: Logic & LPR (v0.4)
Implement MotionDetector (Simple pixel difference or MOG2).

Implement CropNode (Takes detections, produces sub-images).

Implement OCRNode (Wrapper for custom LPR model).

Goal: End-to-end functionality: Motion -> Car Detect -> Crop -> Read Plate.

6. Technical Standards & Guidelines
Latency over Throughput: This is a real-time system. If processing takes longer than the frame interval (e.g., 33ms for 30fps), drop the frame. Do not queue frames indefinitely, or latency will spiral.

Zero-Copy: Minimize np.copy(). Pass references of the image in the FramePacket. Only copy if drawing bounding boxes for a debug display (keep the raw image clean for the OCR node).

Dependency Isolation:

The desktop install requirement should include torch, ultralytics, opencv-python.

The pi install requirement should only include tflite-runtime, opencv-python-headless, and numpy.

Do not force the Pi to install PyTorch