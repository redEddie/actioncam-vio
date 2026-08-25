"""Make the deploy package importable for all tests (config/, control/, inference/, ...)."""
from pathlib import Path
import sys

DEPLOY = Path(__file__).resolve().parents[1]
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))
