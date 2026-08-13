import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def export_model(model_path: str, int8: bool = False, imgsz: int = 640):
    """
    Exports a YOLOv8 model to TFLite format.

    Args:
        model_path (str): Path to the .pt model file.
        int8 (bool): Whether to enable int8 quantization.
        imgsz (int): Image size for the model.
    """
    try:
        logger.info(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        
        logger.info(f"Starting export to TFLite (int8={int8}, imgsz={imgsz})...")
        # Export the model
        # Ultralytics export puts the output in the same directory as the model
        model.export(format="tflite", int8=True, imgsz=imgsz)
        
        logger.info("Export completed successfully.")
        
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to TFLite for Raspberry Pi.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the input .pt model file.")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization (recommended for RPi/EdgeTPU).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640).")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        # If default, might auto-download, but let's warn if specific path missing
        if args.model != "yolov8n.pt":
            logger.error(f"Model file '{args.model}' not found.")
            return

    export_model(args.model, args.int8, args.imgsz)

if __name__ == "__main__":
    main()
