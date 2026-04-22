from __future__ import annotations

import csv
import gzip
import json
import numpy as np
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_default)


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_csv(path: str | Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        return
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    exists = path.exists()
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "at" if path.suffix == ".gz" else "a"
    with opener(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def write_csv(path: str | Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
    append_csv(path, rows, fieldnames)
