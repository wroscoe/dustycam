import os
import glob
from ultralytics import YOLO
from PIL import Image, ImageDraw

def run_yolo_detection(input_folder, output_dir=None, output_format="all", model_path="yolov8n.pt"):
    """
    Run YOLO detection on images in a folder.
    
    Args:
        input_folder: Folder containing input images.
        output_dir: Folder to save outputs. If None, uses input_folder/results.
        output_format: 'yolo', 'images', or 'all'.
        model_path: Path to the YOLO model file.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    if output_dir is None:
        output_dir = os.path.join(input_folder, "yolo_results")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load model
    print(f"Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    image_files = glob.glob(os.path.join(input_folder, "*.png")) + glob.glob(os.path.join(input_folder, "*.jpg"))
    # Filter out annotated images if they exist in the same folder
    target_files = [f for f in image_files if "_annotated" not in f and "yolo_result" not in f]

    if not target_files:
        print("No images found to process.")
        return

    print(f"Processing {len(target_files)} images...")

    for img_path in target_files:
        try:
            results = model(img_path, verbose=False)
            result = results[0] # Single image inference
            
            basename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(basename)[0]

            # Save YOLO txt format
            if output_format in ["yolo", "all"]:
                txt_path = os.path.join(output_dir, f"{name_no_ext}.txt")
                with open(txt_path, "w") as f:
                    for box in result.boxes:
                        # YOLO format: class_id x_center y_center width height (normalized)
                        cls = int(box.cls.item())
                        xywhn = box.xywhn[0].cpu().numpy()
                        line = f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
                        f.write(line)
                # print(f"Saved labels to {txt_path}")

            # Save annotated image
            if output_format in ["images", "all"]:
                # Ultralytics has a plot() method that returns a BGR numpy array
                # But to be consistent with our PIL usage, let's use the result array
                # Or just use the model's plotter for convenience
                res_plotted = result.plot()
                
                # Convert BGR to RGB for PIL
                import cv2
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(res_rgb)
                
                img_out_path = os.path.join(output_dir, f"{name_no_ext}_yolo.jpg")
                im_pil.save(img_out_path)
                # print(f"Saved annotated image to {img_out_path}")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Done. Results saved to {output_dir}")
