from __future__ import annotations

from pathlib import Path
from typing import Dict


def make_plots(cfg: Dict[str, object]) -> None:
    # Optional lightweight plotting hook. The core package avoids requiring matplotlib.
    out_dir = Path(cfg["project"]["output_dir"])
    plot_dir = out_dir / "TWC_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "README.md").write_text(
        "Plot generation is intentionally lightweight in v11. Use evaluation/evaluation_summary.csv for plotting.\n",
        encoding="utf-8",
    )
