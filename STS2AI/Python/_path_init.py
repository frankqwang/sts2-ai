"""sys.path initializer — ensures STS2AI/Python is on sys.path.

Only adds the Python root directory. All imports should use package-qualified
paths (e.g., `from core.vocab import Vocab`, not `from vocab import Vocab`).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parent
_root_str = str(_PYTHON_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
