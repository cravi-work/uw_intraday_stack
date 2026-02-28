# patch_tests_task12.py
import glob
import re
import os

# 1. Update live config.yaml
cfg_path = "src/config/config.yaml"
if os.path.exists(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("realized_snapshot_tolerance_minutes", "tolerance_minutes")
    if "max_horizon_drift_minutes" not in content:
        content = content.replace("tolerance_minutes: 10", "tolerance_minutes: 10\n  max_horizon_drift_minutes: 10")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)

# 2. Update unit test mock configs
for filepath in glob.glob("tests/*.py"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    original_content = content
    
    if '"invalid_after_minutes"' in content and '"tolerance_minutes"' not in content:
        content = re.sub(
            r'("invalid_after_minutes":\s*60,)', 
            r'\1\n            "tolerance_minutes": 10,\n            "max_horizon_drift_minutes": 10,\n            "flat_threshold_pct": 0.001,', 
            content
        )
        
    if "'invalid_after_minutes'" in content and "'tolerance_minutes'" not in content:
        content = re.sub(
            r"('invalid_after_minutes':\s*60,)", 
            r"\1\n            'tolerance_minutes': 10,\n            'max_horizon_drift_minutes': 10,\n            'flat_threshold_pct': 0.001,", 
            content
        )

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Patched test file: {filepath}")

print("All test configurations patched successfully!")