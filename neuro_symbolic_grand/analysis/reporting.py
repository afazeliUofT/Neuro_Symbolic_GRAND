from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from ..config import ExperimentConfig
from ..utils.io import write_dataframe_csv, write_json


def _line_plot(df: pd.DataFrame, x: str, y: str, hue: str, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 4.2))
    for label, sub_df in df.groupby(hue):
        ordered = sub_df.sort_values(x)
        plt.plot(ordered[x], ordered[y], marker="o", label=str(label))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _single_series_plot(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 4.2))
    ordered = df.sort_values(x)
    plt.plot(ordered[x], ordered[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def build_reports(cfg: ExperimentConfig, output_dir: Path, logger) -> Dict[str, object]:
    eval_root = output_dir / "evaluation"
    report_root = output_dir / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(eval_root / "evaluation_summary.csv")
    raw_paths = sorted(eval_root.glob("profile_*/snr_*dB/raw_records.csv"))
    raw_df = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True) if raw_paths else pd.DataFrame()

    overview = (
        summary_df.pivot_table(
            index=["profile", "snr_db"],
            columns="decoder",
            values=["bler", "avg_queries", "avg_elapsed_ms", "fallback_rate"],
        )
        .reset_index()
    )
    overview.columns = [
        "_".join(str(c) for c in col if c != "") if isinstance(col, tuple) else str(col)
        for col in overview.columns
    ]
    if "avg_queries_baseline" in overview.columns and "avg_queries_nsgrand" in overview.columns:
        overview["query_reduction_pct"] = 100.0 * (
            overview["avg_queries_baseline"] - overview["avg_queries_nsgrand"]
        ) / overview["avg_queries_baseline"].clip(lower=1e-9)
    if "bler_baseline" in overview.columns and "bler_nsgrand" in overview.columns:
        overview["bler_delta"] = overview["bler_nsgrand"] - overview["bler_baseline"]
    write_dataframe_csv(overview, report_root / "overview_by_profile_snr.csv")

    profiles: List[str] = sorted(summary_df["profile"].unique().tolist())
    for profile in profiles:
        profile_df = summary_df[summary_df["profile"] == profile].copy()
        _line_plot(profile_df, "snr_db", "bler", "decoder", f"BLER vs SNR ({profile})", report_root / f"bler_vs_snr_{profile}.png")
        _line_plot(profile_df, "snr_db", "avg_queries", "decoder", f"Average queries vs SNR ({profile})", report_root / f"queries_vs_snr_{profile}.png")
        _line_plot(profile_df, "snr_db", "p95_queries", "decoder", f"P95 queries vs SNR ({profile})", report_root / f"p95_queries_vs_snr_{profile}.png")
        _line_plot(profile_df, "snr_db", "avg_elapsed_ms", "decoder", f"Average latency vs SNR ({profile})", report_root / f"latency_vs_snr_{profile}.png")
        ns_df = profile_df[profile_df["decoder"] == "nsgrand"]
        if not ns_df.empty:
            _single_series_plot(ns_df, "snr_db", "fallback_rate", f"NS-GRAND fallback rate vs SNR ({profile})", report_root / f"fallback_vs_snr_{profile}.png")

    if not raw_df.empty:
        weight_df = (
            raw_df.groupby(["decoder", "true_error_weight"], as_index=False)
            .agg(
                block_error_rate=("block_error", "mean"),
                avg_queries=("queries", "mean"),
                avg_elapsed_ms=("elapsed_ms", "mean"),
            )
            .sort_values(["decoder", "true_error_weight"])
        )
        write_dataframe_csv(weight_df, report_root / "metrics_by_true_error_weight.csv")
        _line_plot(
            weight_df,
            "true_error_weight",
            "block_error_rate",
            "decoder",
            "Block error rate vs true hard-decision error weight",
            report_root / "bler_vs_true_error_weight.png",
        )
        _line_plot(
            weight_df,
            "true_error_weight",
            "avg_queries",
            "decoder",
            "Average queries vs true hard-decision error weight",
            report_root / "queries_vs_true_error_weight.png",
        )

    history_path = output_dir / "training" / "training_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        _single_series_plot(history_df, "epoch", "train_loss", "Training loss by epoch", report_root / "train_loss_by_epoch.png")
        _single_series_plot(history_df, "epoch", "val_loss", "Validation loss by epoch", report_root / "val_loss_by_epoch.png")

    key_findings: List[str] = []
    if not overview.empty and "query_reduction_pct" in overview.columns:
        best_row = overview.sort_values("query_reduction_pct", ascending=False).iloc[0]
        key_findings.append(
            f"Best query reduction occurs at profile={best_row['profile']} and snr_db={best_row['snr_db']:.1f} with reduction={best_row['query_reduction_pct']:.2f}%"
        )
    if not overview.empty and "bler_delta" in overview.columns:
        worst_row = overview.sort_values("bler_delta", ascending=False).iloc[0]
        key_findings.append(
            f"Largest BLER increase for nsgrand relative to baseline is at profile={worst_row['profile']} and snr_db={worst_row['snr_db']:.1f}, delta={worst_row['bler_delta']:.6f}"
        )

    report_md = [
        f"# Neuro-Symbolic GRAND report: {cfg.experiment_name}",
        "",
        "## Generated artifacts",
        "",
        "- `evaluation/evaluation_summary.csv` contains point-level metrics.",
        "- `reports/overview_by_profile_snr.csv` compiles BLER, query count, latency, and fallback rate deltas.",
        "- `evaluation/profile_*/snr_*dB/raw_records.csv` contains per-sample decoder behavior for later forensic analysis.",
        "- `evaluation/profile_*/snr_*dB/traces.jsonl` stores detailed candidate-query traces for sampled blocks.",
        "",
        "## Key findings",
        "",
    ]
    if key_findings:
        report_md.extend([f"- {item}" for item in key_findings])
    else:
        report_md.append("- No evaluation findings available.")
    (report_root / "report.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")

    summary = {
        "report_dir": str(report_root),
        "num_eval_points": int(len(summary_df)),
        "raw_record_files": len(raw_paths),
        "key_findings": key_findings,
    }
    write_json(summary, report_root / "report_summary.json")
    logger.info("Report artifacts written to %s", report_root)
    return summary
