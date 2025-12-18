import time
import sys
# Ensure current dir is in path
sys.path.append(".")

from dustycam.pipeline import BallYoloPipeline
import numpy as np

def test_class_pipeline():
    print("Initializing BallYoloPipeline...")
    pipeline = BallYoloPipeline()
    
    print("Starting pipeline...")
    pipeline.start()
    
    # Let it run for a bit
    time.sleep(1.0)
    
    # Check if we have a preview
    frame = pipeline.get_preview("Main")
    
    if frame is None:
        print("Error: No preview frame captured!")
        pipeline.stop()
        sys.exit(1)
        
    print(f"Captured frame shape: {frame.shape}")
    assert isinstance(frame, np.ndarray)
    
    print("Stopping pipeline...")
    pipeline.stop()
    print("Test passed!")

if __name__ == "__main__":
    test_class_pipeline()
