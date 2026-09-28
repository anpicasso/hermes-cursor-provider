from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "provider"
if str(PROVIDER) not in sys.path:
    sys.path.insert(0, str(PROVIDER))
