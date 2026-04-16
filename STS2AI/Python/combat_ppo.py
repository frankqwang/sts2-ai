"""Backward-compat shim — moved to training/ folder."""
import importlib as _il
_m = _il.import_module(f"training.{__name__.split('.')[-1]}")
from types import ModuleType as _MT
import sys as _sys
_mod = _sys.modules[__name__]
for _k in dir(_m):
    if not _k.startswith('__'):
        setattr(_mod, _k, getattr(_m, _k))
