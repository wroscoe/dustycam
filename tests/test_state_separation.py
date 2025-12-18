import sys
import os
import time
import threading
from pathlib import Path

# Ensure current dir is in path
sys.path.append(".")

from dustycam.pipeline import Pipeline, register_pipeline, get_pipeline_by_name

# Mock pipeline to test restarts
class TestRestartPipeline(Pipeline):
    def __init__(self):
        super().__init__(name="test_state_pipeline")
        self.start_count = 0
        
    def run_loop(self):
        print(f"TestRestartPipeline run_loop started (Count: {self.start_count})")
        while self.running:
            time.sleep(0.01)
            
    def start(self):
        self.start_count += 1
        super().start()

register_pipeline("test_state", TestRestartPipeline)

def test_state_separation():
    print("Running state separation test...")
    settings_dir = Path(os.path.expanduser("~/.dustycam/settings"))
    settings_file = settings_dir / "test_state_pipeline.json"
    
    if settings_file.exists():
        settings_file.unlink()
        
    p = get_pipeline_by_name("test_state")
    
    # 1. Test State Separation
    p.state['motion_detected'] = True
    p.settings['threshold'] = 50
    p.update_settings({'fps': 30})
    
    # Check that settings persisted but state didn't
    import json
    with open(settings_file, 'r') as f:
        saved = json.load(f)
        
    assert 'threshold' in saved
    assert 'fps' in saved
    assert 'motion_detected' not in saved, "State should not be saved to disk"
    print("State separation verified.")
    
    # 2. Test Restart
    p.start()
    current_count = p.start_count
    print(f"Initial start count: {current_count}")
    
    print("Updating settings with restart=True...")
    p.update_settings({'fps': 60}, restart=True)
    
    time.sleep(0.5) # Give it time to restart
    
    assert p.start_count > current_count, f"Pipeline passed start count check: {p.start_count} > {current_count}"
    print("Restart verified.")
    
    p.stop()
    
    # Cleanup
    if settings_file.exists():
        settings_file.unlink()
        
    print("Test passed!")

if __name__ == "__main__":
    test_state_separation()
