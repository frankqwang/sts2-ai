"""sys.path initializer for STS2AI/Python and its library subdirs.

Every runnable/importable script under `STS2AI/Python/` should import this
module near the top, after any `from __future__` imports and before any
project-relative imports.

The import is idempotent. It keeps the mainline flat-import namespace working
even when a script lives in a subdirectory such as `search/` or `ipc/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PYTHON_ROOT = _THIS_FILE.parent

_LIBRARY_SUBDIRS: list[str] = [
    "core",
    "ipc",
    "search",
]

_ALL_PATHS = [_PYTHON_ROOT] + [_PYTHON_ROOT / subdir for subdir in _LIBRARY_SUBDIRS]

for _path in _ALL_PATHS:
    _path_str = str(_path)
    if _path.exists() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)
