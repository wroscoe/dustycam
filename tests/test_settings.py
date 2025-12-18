import sys
import os
import shutil
from pathlib import Path

# Ensure current dir is in path
sys.path.append(".")

from dustycam.pipeline import Pipeline, register_pipeline, get_pipeline_by_name

class TestPipeline(Pipeline):
    def __init__(self):
        super().__init__(name="test_settings_pipeline")
    def run_loop(self):
        pass

register_pipeline("test_settings", TestPipeline)

def test_settings_persistence():
    print("Running settings persistence test...")
    settings_dir = Path(os.path.expanduser("~/.dustycam/settings"))
    settings_file = settings_dir / "test_settings_pipeline.json"
    
    # 1. Clean up previous valid runs
    if settings_file.exists():
        settings_file.unlink()
    
    # 2. Instantiate and verify defaults (should be empty initially, or whatever we set)
    p1 = get_pipeline_by_name("test_settings")
    assert p1.settings == {}
    print("Initial settings verified.")
    
    # 3. Update settings and save
    new_settings = {"sensitivity": 80, "mode": "night"}
    p1.update_settings(new_settings)
    assert p1.settings == new_settings
    print("Settings updated in memory.")
    
    assert settings_file.exists()
    print("Settings file created.")
    
    # 4. Instantiate NEW instance and verify it loads settings
    p2 = get_pipeline_by_name("test_settings")
    assert p2.settings == new_settings
    print("Settings loaded correctly in new instance.")
    
    # Cleanup
    if settings_file.exists():
        settings_file.unlink()
        
    print("Test passed!")

if __name__ == "__main__":
    test_settings_persistence()
