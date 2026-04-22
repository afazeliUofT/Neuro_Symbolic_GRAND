from __future__ import annotations

import importlib
import sys


def dependency_report() -> str:
    names = ["numpy", "torch", "tensorflow", "sionna", "yaml"]
    lines = [f"python: {sys.version.replace(chr(10), ' ')}"]
    for name in names:
        try:
            mod = importlib.import_module(name)
            lines.append(f"{name}: {getattr(mod, '__version__', 'ok')}")
        except Exception as e:
            lines.append(f"{name}: NOT IMPORTABLE ({e})")
    return "\n".join(lines)
