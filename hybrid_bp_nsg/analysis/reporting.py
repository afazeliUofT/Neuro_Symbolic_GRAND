from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from ..utils.io import ensure_dir, write_csv
from ..utils.logging import get_logger


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(r: dict, k: str, default: float = 0.0) -> float:
    try:
        return float(r.get(k, default))
    except Exception:
        return float(default)


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

    report_dir = ensure_dir(out_dir / "reports")
    md = [
        "# Hybrid BP + PUSCH-aligned CRC-aware AI/Tanner-GRAND evaluation",
        "",
        "| profile | snr_db | decoder | BLER | avg_latency_ms | avg_queries | rescue_rate | crc_fail_rate | avg_crc_valid_candidates | avg_parity_valid_candidates |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r.get('profile','')} | {_f(r,'snr_db'):g} | {r.get('decoder','')} | {_f(r,'bler'):.6g} | {_f(r,'avg_latency_ms'):.4g} | {_f(r,'avg_queries'):.4g} | {_f(r,'rescue_rate'):.4g} | {_f(r,'crc_fail_rate'):.4g} | {_f(r,'avg_crc_valid_candidates'):.4g} | {_f(r,'avg_parity_valid_candidates'):.4g} |"
        )
    (report_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("Wrote %s and %s", eval_root / "evaluation_summary.csv", report_dir / "README.md")
