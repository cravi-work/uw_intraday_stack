# patch_tests.py
import glob
import re

print("Patching test configurations to match strict Phase C contracts...")

for filepath in glob.glob("tests/*.py"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    original_content = content
    
    # 1. Patch missing validation keys (Task 6 enforcement)
    if '"invalid_after_minutes"' not in content and '"validation":' in content:
        content = re.sub(
            r'("validation":\s*\{)', 
            r'\1\n            "invalid_after_minutes": 60,\n            "fallback_max_age_minutes": 15,', 
            content
        )
        
    if "'invalid_after_minutes'" not in content and "'validation':" in content:
        content = re.sub(
            r"('validation':\s*\{)", 
            r"\1\n            'invalid_after_minutes': 60,\n            'fallback_max_age_minutes': 15,", 
            content
        )

    # 2. Patch missing root dependencies for Replay Parity & Config Tests (Task 5 enforcement)
    if "test_replay_parity" in filepath or "test_config_contract" in filepath:
        if '"storage":' not in content and '"ingestion":' in content:
            content = re.sub(
                r'("ingestion":\s*\{)', 
                r'"storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},\n        "system": {},\n        "network": {},\n        \1', 
                content
            )
        if "'storage':" not in content and "'ingestion':" in content:
            content = re.sub(
                r"('ingestion':\s*\{)", 
                r"'storage': {'duckdb_path': ':memory:', 'cycle_lock_path': 'mock.lock', 'writer_lock_path': 'mock.lock'},\n        'system': {},\n        'network': {},\n        \1", 
                content
            )

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Patched: {filepath}")

print("All test configurations patched successfully!")