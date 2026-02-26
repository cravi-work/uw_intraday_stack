# tests/conftest.py
import sys
import os
from pathlib import Path

# Ensure project root is importable (so `import src...` works in all runners)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Inject a mock API key into the test environment to satisfy UwClient validation globally
os.environ["UW_API_KEY"] = "mock_api_key_for_testing"