import argparse
import os
import time
import glob
from datetime import datetime
from dustycam.utils.image_gen import generate_image_prompts, generate_image, detect_license_plates, plot_bounding_boxes
from dustycam.nodes.yolo import load_yolo_model, detect_objects
from dustycam.utils.oneshot import run_oneshot_workflow
from dustycam.pipeline import get_pipeline_by_name
import cv2


def run_make(args):
    run_oneshot_workflow(args.prompt)


def run_yolo(args):
    model = load_yolo_model(args.model_path)
    if not model:
        print("Failed to load model.")
        return

    input_folder = args.input_folder
    output_dir = args.output_dir
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find images
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
    print(f"Found {len(image_files)} images in {input_folder}")

    from dustycam.nodes.drawing import draw_detections

    for img_path in image_files:
        print(f"Processing {img_path}...")
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        detections = detect_objects(image, model)
        print(f"  Found {len(detections)} objects.")
        
        if output_dir:
            # Draw and save
            annotated_image = draw_detections(image, detections)
            filename = os.path.basename(img_path)
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, annotated_image)
            print(f"  Saved to {out_path}")

def run_generate(args):
    # Determine output folder
    if args.folder:
        output_dir = args.folder
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("data", f"generated_plates_{timestamp}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using output directory: {output_dir}")

    prompts = generate_image_prompts(args.topic, args.count)
    clean_topic = "".join(x for x in args.topic if x.isalnum() or x in (' ','-','_')).replace(' ','_').lower()

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Processing...")
        generate_image(prompt, output_dir, clean_topic)
        time.sleep(1)

def run_detect(args):
    folder = args.folder
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    print(f"Running detection on images in: {folder}")
    # Find all png/jpg images that are NOT already annotated
    image_files = glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))
    target_files = [f for f in image_files if "_annotated" not in f]

    if not target_files:
        print("No images found to process.")
        return

    for img_path in target_files:
        bboxes = detect_license_plates(img_path)
        if bboxes:
            plot_bounding_boxes(img_path, bboxes)
        else:
            print(f"No objects detected in {os.path.basename(img_path)}")
        time.sleep(1)

def run_start(args):
    import threading
    import cv2
    from dustycam.pipeline import get_pipeline_by_name
    from dustycam.webapp import create_app

    pipeline = get_pipeline_by_name(args.pipeline)

    if not pipeline:
        print(f"Unknown pipeline: {args.pipeline}")
        return

    # Start Pipeline
    pipeline.start()
    
    # Pass pipeline to app
    app = create_app(pipeline=pipeline)
    
    print("Starting Webapp on port 5000...")
    import uvicorn
    print("Starting Webapp on port 5000...")
    try:
        # Note: uvicorn.run blocks.
        uvicorn.run(app, host='0.0.0.0', port=5000, log_level="info")
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description="DustyCam Image Generator & Detector")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # Generate Command
    gen_parser = subparsers.add_parser("generate", help="Generate images from a topic")
    gen_parser.add_argument("--topic", type=str, default="A license plate on a car", help="Description of images to generate")
    gen_parser.add_argument("--count", type=int, default=3, help="Number of images to generate")
    gen_parser.add_argument("--folder", type=str, help="Specific folder to save images (optional)")
    gen_parser.set_defaults(func=run_generate)

    # Detect Command
    detect_parser = subparsers.add_parser("detect", help="Detect license plates in existing images")
    detect_parser.add_argument("--folder", type=str, required=True, help="Folder containing images to process")
    detect_parser.set_defaults(func=run_detect)

    # YOLO Detect Command
    yolo_parser = subparsers.add_parser("yolodetect", help="Run YOLO object detection")
    yolo_parser.add_argument("input_folder", type=str, help="Folder containing images to process")
    yolo_parser.add_argument("--output_dir", type=str, help="Folder to save results (optional)")
    yolo_parser.add_argument("--output_format", type=str, choices=["yolo", "images", "all"], default="all", help="Output format")
    yolo_parser.add_argument("--model_path", type=str, default="yolov8n.pt", help="Path to YOLO model file")
    yolo_parser.set_defaults(func=run_yolo)

    # Make Command
    make_parser = subparsers.add_parser("make", help="Initialize a project from a description")
    make_parser.add_argument("prompt", type=str, help="Description of the camera model to build")
    make_parser.set_defaults(func=run_make)

    # Start Command
    start_parser = subparsers.add_parser("start", help="Start the camera application")
    start_parser.add_argument("--pipeline", type=str, default="default", help="Pipeline configuration to run")
    start_parser.set_defaults(func=run_start)

    args = parser.parse_args()
    args.func(args)
