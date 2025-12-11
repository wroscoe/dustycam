import os
import time
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageColor
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Helper class to represent a bounding box
class BoundingBox(BaseModel):
    box_2d: list[int]
    label: str

def get_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return None

def detect_license_plates(image_path: str) -> list[BoundingBox]:
    client = get_client()
    if not client: return []
    print(f"Detecting license plates in {image_path}...")
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        prompt = "Return bounding boxes for all license plates in the image. Label them as 'license plate'."
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[BoundingBox],
                safety_settings=[types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH")],
            )
        )
        return response.parsed if response.parsed else []
    except Exception as e:
        print(f"Failed to detect objects: {e}")
        return []

def plot_bounding_boxes(image_path: str, bounding_boxes: list[BoundingBox]) -> str:
    if not bounding_boxes: return None
    try:
        with Image.open(image_path) as im:
            width, height = im.size
            draw = ImageDraw.Draw(im)
            for bbox in bounding_boxes:
                # Gemini returns [y_min, x_min, y_max, x_max] scaled 0-1000
                abs_y_min = int(bbox.box_2d[0] / 1000 * height)
                abs_x_min = int(bbox.box_2d[1] / 1000 * width)
                abs_y_max = int(bbox.box_2d[2] / 1000 * height)
                abs_x_max = int(bbox.box_2d[3] / 1000 * width)
                
                # Check if box is valid (has area)
                if abs_x_max > abs_x_min and abs_y_max > abs_y_min:
                    draw.rectangle(((abs_x_min, abs_y_min), (abs_x_max, abs_y_max)), outline="red", width=4)
                    if bbox.label:
                        draw.text((abs_x_min + 8, abs_y_min + 6), bbox.label, fill="red")
            
            base, ext = os.path.splitext(image_path)
            annotated_path = f"{base}_annotated{ext}"
            im.save(annotated_path)
            print(f"Saved annotated image to {annotated_path}")
            return annotated_path
    except Exception as e:
        print(f"Failed to plot bounding boxes: {e}")
        return None

def generate_image_prompts(topic: str, count: int = 5) -> list[str]:
    client = get_client()
    if not client: return []
    print(f"\n--- Generating {count} prompts for topic: '{topic}' ---")
    
    prompt_request = f"""
    Act as an expert prompt engineer for Google Vertex AI Imagen 3.
    Goal: Generate {count} distinct image generation prompts for the topic: "{topic}".
    Output: Provide ONLY the output as a simple list of {count} prompts, one per line.
    """
    try:
        response = client.models.generate_content(model='gemini-2.0-flash-exp', contents=prompt_request)
        lines = response.text.split('\n')
        prompts = [line.strip() for line in lines if line.strip() and not line.strip()[0].isdigit()]
        if not prompts: prompts = [line.strip() for line in lines if line.strip()]
        return prompts[:count]
    except Exception as e:
        print(f"Failed to generate prompts: {e}")
        return []

def generate_image(prompt: str, output_dir: str, filename_prefix: str) -> str:
    client = get_client()
    if not client: return None
    model_name = 'imagen-4.0-fast-generate-001'
    print(f"\nGenerating image with model: {model_name}\nPrompt: {prompt}")
    try:
        response = client.models.generate_images(model=model_name, prompt=prompt)
        if response.generated_images:
            image = response.generated_images[0].image
            filename = f"{filename_prefix}_{abs(hash(prompt)) % 10000}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"Success! Image saved to {filepath}")
            return filepath
        print("No images returned.")
        return None
    except Exception as e:
        print(f"Failed to generate image: {e}")
        return None
