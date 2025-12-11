import argparse
import os
import time
import glob
from datetime import datetime
from dustycam.utils.image_gen import generate_image_prompts, generate_image, detect_license_plates, plot_bounding_boxes
from dustycam.utils.yolo_detect import run_yolo_detection
from dustycam.utils.oneshot import run_oneshot_workflow


def run_make(args):
    run_oneshot_workflow(args.prompt)


def run_yolo(args):
    run_yolo_detection(
        input_folder=args.input_folder,
        output_dir=args.output_dir,
        output_format=args.output_format,
        model_path=args.model_path 
    )

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

    args = parser.parse_args()
    args.func(args)
