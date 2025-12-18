import sys
import os
import shutil
from pathlib import Path
from pydantic import BaseModel, Field

# Ensure current dir is in path
sys.path.append(".")

from dustycam.pipeline import Pipeline, register_pipeline, get_pipeline_by_name

# Mock pipeline setup
class TestPydanticPipeline(Pipeline):
    class Settings(BaseModel):
        test_int: int = Field(10, json_schema_extra={"group": "TestGroup", "min": 0, "max": 100})
        test_bool: bool = Field(False, json_schema_extra={"group": "TestGroup", "restart_required": True})

    def __init__(self):
        super().__init__(name="test_pydantic_pipeline", settings_model=self.Settings)
    
    def run_loop(self):
        pass

register_pipeline("test_pydantic", TestPydanticPipeline)

def test_pydantic_settings():
    print("Running Pydantic settings test...")
    settings_dir = Path(os.path.expanduser("~/.dustycam/settings"))
    settings_file = settings_dir / "test_pydantic_pipeline.json"
    
    if settings_file.exists():
        settings_file.unlink()
        
    p = get_pipeline_by_name("test_pydantic")
    
    # 1. Defaults
    assert p.settings.test_int == 10
    assert p.settings.test_bool is False
    print("Defaults verified.")
    
    # 2. Schema Check
    schema = p.settings.model_json_schema()
    props = schema['properties']
    assert 'test_int' in props
    assert props['test_int']['group'] == 'TestGroup'
    assert props['test_int']['max'] == 100
    assert props['test_bool']['restart_required'] is True
    print("Schema metadata verified.")
    
    # 3. Update Valid
    p.update_settings({"test_int": 50})
    assert p.settings.test_int == 50
    # Persistence check
    p._load_settings() # Reload from disk
    assert p.settings.test_int == 50
    print("Valid update and persistence verified.")
    
    # 4. Update Invalid (Should warn/fail gracefully but not crash, in our impl we catch exceptions)
    # Our impl catches Exception and logs, so values shouldn't change
    p.update_settings({"test_int": "not_an_int"}) 
    # Just verify it didn't change (or if pydantic coerced it, but string text wont coerce to int easily)
    # Actually pydantic might try to coerce, but "not_an_int" definitely fails.
    assert p.settings.test_int == 50 
    print("Invalid update handled (ignored).")

    # Cleanup
    if settings_file.exists():
        settings_file.unlink()
        
    print("Test passed!")

if __name__ == "__main__":
    test_pydantic_settings()
