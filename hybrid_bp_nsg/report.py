from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict


def report(cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg["project"]["output_dir"])
    src = out_dir / "evaluation" / "evaluation_summary.csv"
    if not src.exists():
        raise RuntimeError(f"Missing evaluation summary: {src}")
    rows = list(csv.DictReader(src.open()))
    rep = out_dir / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    cols = ["profile", "snr_db", "decoder", "fer", "ber_internal", "payload_ber", "avg_latency_ms", "avg_queries", "rescue_rate", "crc_fail_rate", "avg_crc_valid_candidates", "avg_parity_valid_candidates"]
    lines = ["# Hybrid BP-NSG GRAND v14 Evaluation", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            try:
                if c not in {"profile", "decoder"}:
                    v = f"{float(v):.6g}"
            except Exception:
                pass
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    (rep / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {src} and {rep/'README.md'}")
