import sys
from pathlib import Path

# Add project root to python path for pytest test runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
