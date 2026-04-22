from __future__ import annotations

import importlib
import sys


def dependency_report() -> str:
    names = ["numpy", "yaml", "tensorflow", "sionna", "torch"]
    lines = [f"python: {sys.version.replace(chr(10), ' ')}"]
    for name in names:
        try:
            mod = importlib.import_module(name)
            suffix = "" if name != "torch" else " (optional)"
            lines.append(f"{name}{suffix}: {getattr(mod, '__version__', 'ok')}")
        except Exception as e:
            label = "optional not importable" if name == "torch" else "NOT IMPORTABLE"
            lines.append(f"{name}: {label} ({e})")
    return "\n".join(lines)
