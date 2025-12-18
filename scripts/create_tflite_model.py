import argparse
import logging
import os
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TFLite format.")
    parser.add_argument("--model", type=str, default="yolo11n", help="Name of the model to export (default: yolo11n)")
    args = parser.parse_args()

    model_name = args.model
    model_file = f"{model_name}.pt" if not model_name.endswith(".pt") else model_name

    logger.info(f"Checking for model file: {model_file}")

    try:
        if not os.path.exists(model_file) and not model_name.startswith("yolo"):
             # If it's a local file that doesn't exist, warn unless it looks like a standard ultralytics model name 
             # which ultralytics will auto-download
             logger.warning(f"Local file {model_file} not found. Ultralytics might try to download it.")

        logger.info(f"Loading YOLO model {model_name}...")
        model = YOLO(model_name)
        
        logger.info(f"Exporting {model_name} to TFLite format...")
        model.export(format="tflite")
        
        expected_output = f"{model_name.replace('.pt', '')}_float32.tflite" # Standard output name format usually
        logger.info("Export complete.")

    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        exit(1)

if __name__ == "__main__":
    main()