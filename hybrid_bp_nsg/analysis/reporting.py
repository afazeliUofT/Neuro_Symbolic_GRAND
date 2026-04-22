from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from ..utils.io import ensure_dir, write_csv
from ..utils.logging import get_logger


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_report(cfg: Dict[str, object]) -> None:
    logger = get_logger("report")
    out_dir = Path(cfg["project"]["output_dir"])
    eval_root = out_dir / "evaluation"
    rows: List[dict] = []
    for p in sorted(eval_root.glob("profile_*/snr_*dB/summary.csv")):
        rows.extend(_read_csv(p))
    if not rows:
        logger.info("No evaluation summary rows found under %s", eval_root)
        return
    write_csv(eval_root / "evaluation_summary.csv", rows)
    # Lightweight markdown report.
    report_dir = ensure_dir(out_dir / "reports")
    md = ["# Hybrid BP + Channel-Aligned GRAND evaluation", "", "| profile | snr_db | decoder | samples | frame_errors | BLER | avg_queries | rescue_rate |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r.get('profile','')} | {float(r.get('snr_db',0)):g} | {r.get('decoder','')} | {r.get('samples','')} | {r.get('frame_errors','')} | {float(r.get('bler',0)):.6g} | {float(r.get('avg_queries',0)):.3g} | {float(r.get('rescue_rate',0)):.3g} |")
    (report_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("Wrote %s and %s", eval_root / "evaluation_summary.csv", report_dir / "README.md")
