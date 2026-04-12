from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from ..config import ExperimentConfig
from ..utils.io import copy_file, read_jsonl, write_dataframe_csv, write_json, write_jsonl


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


def _bar_plot(df: pd.DataFrame, x: str, y: str, title: str, path: Path, rotate_xticks: bool = False) -> None:
    plt.figure(figsize=(8, 4.5))
    ordered = df.sort_values(y, ascending=False)
    plt.bar(ordered[x].astype(str), ordered[y])
    if rotate_xticks:
        plt.xticks(rotation=45, ha="right")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _build_overview(summary_df: pd.DataFrame) -> pd.DataFrame:
    overview = (
        summary_df.pivot_table(
            index=["profile", "snr_db"],
            columns="decoder",
            values=[
                "bler",
                "avg_queries",
                "avg_primary_queries",
                "avg_fallback_queries",
                "avg_elapsed_ms",
                "fallback_rate",
                "avg_predicted_overflow_prob",
                "avg_confidence_prob",
            ],
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
    if "avg_elapsed_ms_baseline" in overview.columns and "avg_elapsed_ms_nsgrand" in overview.columns:
        overview["latency_reduction_pct"] = 100.0 * (
            overview["avg_elapsed_ms_baseline"] - overview["avg_elapsed_ms_nsgrand"]
        ) / overview["avg_elapsed_ms_baseline"].clip(lower=1e-9)
    return overview


def _build_paired_summary(paired_df: pd.DataFrame, *, ref_decoder: str, target_decoder: str) -> pd.DataFrame:
    if paired_df.empty:
        return pd.DataFrame()
    target_better_col = f"{target_decoder}_better_block_error"
    ref_better_col = f"{ref_decoder}_better_block_error"
    paired_summary = (
        paired_df.groupby(["profile", "snr_db"], as_index=False)
        .agg(
            num_samples=("sample_id", "count"),
            avg_query_delta=("query_delta", "mean"),
            median_query_delta=("query_delta", "median"),
            avg_latency_delta_ms=("latency_delta_ms", "mean"),
            target_better_rate=(target_better_col, "mean"),
            reference_better_rate=(ref_better_col, "mean"),
            tied_block_error_rate=("block_error_delta", lambda s: float((s == 0).mean())),
        )
        .sort_values(["profile", "snr_db"])
    )
    paired_summary["reference_decoder"] = ref_decoder
    paired_summary["target_decoder"] = target_decoder
    return paired_summary


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        copy_file(src, dst)


def build_reports(cfg: ExperimentConfig, output_dir: Path, logger) -> Dict[str, object]:
    eval_root = output_dir / "evaluation"
    report_root = output_dir / "reports"
    export_root = output_dir / "repo_export"
    plots_root = report_root / "plots"
    export_plots_root = export_root / "plots"
    report_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(eval_root / "evaluation_summary.csv")
    raw_global_path = eval_root / "all_raw_records.csv.gz"
    if raw_global_path.exists():
        raw_df = pd.read_csv(raw_global_path)
    else:
        raw_paths = sorted(eval_root.glob("profile_*/snr_*dB/raw_records.csv"))
        raw_df = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True) if raw_paths else pd.DataFrame()

    paired_global_path = eval_root / "all_paired_records.csv.gz"
    if paired_global_path.exists():
        paired_df = pd.read_csv(paired_global_path)
    else:
        paired_paths = sorted(eval_root.glob("profile_*/snr_*dB/paired_records.csv"))
        paired_df = pd.concat([pd.read_csv(path) for path in paired_paths], ignore_index=True) if paired_paths else pd.DataFrame()

    strong_pairwise_path = eval_root / "strong_vs_nsgrand_paired_records.csv.gz"
    if strong_pairwise_path.exists():
        strong_pairwise_df = pd.read_csv(strong_pairwise_path)
    else:
        strong_pairwise_paths = sorted(eval_root.glob("profile_*/snr_*dB/strong_vs_nsgrand_paired_records.csv"))
        strong_pairwise_df = pd.concat([pd.read_csv(path) for path in strong_pairwise_paths], ignore_index=True) if strong_pairwise_paths else pd.DataFrame()

    overview = _build_overview(summary_df)
    write_dataframe_csv(overview, report_root / "overview_by_profile_snr.csv")

    paired_summary = _build_paired_summary(paired_df, ref_decoder="baseline", target_decoder="nsgrand")
    if not paired_summary.empty:
        write_dataframe_csv(paired_summary, report_root / "paired_summary_by_profile_snr.csv")

    strong_paired_summary = _build_paired_summary(strong_pairwise_df, ref_decoder="strong_symbolic", target_decoder="nsgrand")
    if not strong_paired_summary.empty:
        write_dataframe_csv(strong_paired_summary, report_root / "strong_vs_nsgrand_paired_summary.csv")

    profiles: List[str] = sorted(summary_df["profile"].unique().tolist())
    for profile in profiles:
        profile_df = summary_df[summary_df["profile"] == profile].copy()
        _line_plot(profile_df, "snr_db", "bler", "decoder", f"BLER vs SNR ({profile})", plots_root / f"bler_vs_snr_{profile}.png")
        _line_plot(profile_df, "snr_db", "avg_queries", "decoder", f"Average queries vs SNR ({profile})", plots_root / f"queries_vs_snr_{profile}.png")
        _line_plot(profile_df, "snr_db", "avg_elapsed_ms", "decoder", f"Average latency vs SNR ({profile})", plots_root / f"latency_vs_snr_{profile}.png")
        ns_df = profile_df[profile_df["decoder"] == "nsgrand"]
        if not ns_df.empty:
            _single_series_plot(ns_df, "snr_db", "fallback_rate", f"Fallback rate vs SNR ({profile})", plots_root / f"fallback_vs_snr_{profile}.png")
            _single_series_plot(ns_df, "snr_db", "avg_fallback_queries", f"Fallback queries vs SNR ({profile})", plots_root / f"fallback_queries_vs_snr_{profile}.png")
            if "avg_predicted_overflow_prob" in ns_df.columns:
                _single_series_plot(ns_df, "snr_db", "avg_predicted_overflow_prob", f"Predicted overflow vs SNR ({profile})", plots_root / f"overflow_prob_vs_snr_{profile}.png")

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
            plots_root / "bler_vs_true_error_weight.png",
        )
        _line_plot(
            weight_df,
            "true_error_weight",
            "avg_queries",
            "decoder",
            "Average queries vs true hard-decision error weight",
            plots_root / "queries_vs_true_error_weight.png",
        )

        ns_df = raw_df[raw_df["decoder"] == "nsgrand"].copy()
        if not ns_df.empty:
            stage_df = (
                ns_df.groupby(["stage"], as_index=False)
                .agg(count=("sample_id", "count"), bler=("block_error", "mean"), avg_queries=("queries", "mean"))
                .sort_values("count", ascending=False)
            )
            write_dataframe_csv(stage_df, report_root / "nsgrand_stage_summary.csv")
            _bar_plot(stage_df, "stage", "count", "NS-GRAND stage counts", plots_root / "nsgrand_stage_counts.png", rotate_xticks=True)

    gate_summary_path = eval_root / "nsgrand_gate_summary.csv"
    if gate_summary_path.exists():
        gate_df = pd.read_csv(gate_summary_path)
        write_dataframe_csv(gate_df, report_root / "nsgrand_gate_summary.csv")

    action_summary_path = eval_root / "nsgrand_action_summary.csv"
    if action_summary_path.exists():
        action_df = pd.read_csv(action_summary_path)
        write_dataframe_csv(action_df, report_root / "nsgrand_action_summary.csv")

    conf_bins_path = eval_root / "nsgrand_confidence_bins.csv"
    if conf_bins_path.exists():
        conf_df = pd.read_csv(conf_bins_path)
        write_dataframe_csv(conf_df, report_root / "nsgrand_confidence_bins.csv")

    overflow_bins_path = eval_root / "nsgrand_overflow_bins.csv"
    if overflow_bins_path.exists():
        ov_df = pd.read_csv(overflow_bins_path)
        write_dataframe_csv(ov_df, report_root / "nsgrand_overflow_bins.csv")

    tail_path = eval_root / "nsgrand_tail_cases_top.csv"
    if tail_path.exists():
        tail_df = pd.read_csv(tail_path)
        write_dataframe_csv(tail_df, report_root / "nsgrand_tail_cases_top.csv")

    interesting_trace_records = []
    for path in sorted(eval_root.glob("profile_*/snr_*dB/interesting_traces.jsonl")):
        interesting_trace_records.extend(read_jsonl(path))
    if interesting_trace_records:
        write_jsonl(interesting_trace_records, report_root / "interesting_traces_combined.jsonl")

    history_path = output_dir / "training" / "training_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        _single_series_plot(history_df, "epoch", "train_loss", "Training loss by epoch", plots_root / "train_loss_by_epoch.png")
        _single_series_plot(history_df, "epoch", "val_loss", "Validation loss by epoch", plots_root / "val_loss_by_epoch.png")
        if "val_weight_acc" in history_df.columns:
            _single_series_plot(history_df, "epoch", "val_weight_acc", "Validation weight accuracy by epoch", plots_root / "val_weight_acc_by_epoch.png")
        if "val_confidence_brier" in history_df.columns:
            _single_series_plot(history_df, "epoch", "val_confidence_brier", "Validation confidence Brier by epoch", plots_root / "val_confidence_brier_by_epoch.png")

    key_findings: List[str] = []
    if not overview.empty and "query_reduction_pct" in overview.columns:
        best_query_row = overview.sort_values("query_reduction_pct", ascending=False).iloc[0]
        key_findings.append(
            f"Best query reduction occurs at profile={best_query_row['profile']} and snr_db={best_query_row['snr_db']:.1f} with reduction={best_query_row['query_reduction_pct']:.2f}%"
        )
    if not overview.empty and "bler_delta" in overview.columns:
        worst_row = overview.sort_values("bler_delta", ascending=False).iloc[0]
        best_row = overview.sort_values("bler_delta", ascending=True).iloc[0]
        key_findings.append(
            f"Largest BLER increase for nsgrand relative to baseline is at profile={worst_row['profile']} and snr_db={worst_row['snr_db']:.1f}, delta={worst_row['bler_delta']:.6f}"
        )
        key_findings.append(
            f"Largest BLER decrease for nsgrand relative to baseline is at profile={best_row['profile']} and snr_db={best_row['snr_db']:.1f}, delta={best_row['bler_delta']:.6f}"
        )
    if not paired_summary.empty:
        best_paired = paired_summary.sort_values("target_better_rate", ascending=False).iloc[0]
        key_findings.append(
            f"Highest paired-sample nsgrand win rate against baseline is at profile={best_paired['profile']} and snr_db={best_paired['snr_db']:.1f} with win_rate={best_paired['target_better_rate']:.4f}"
        )
    if not strong_paired_summary.empty:
        best_strong = strong_paired_summary.sort_values("target_better_rate", ascending=False).iloc[0]
        key_findings.append(
            f"Highest paired-sample nsgrand win rate against strong_symbolic is at profile={best_strong['profile']} and snr_db={best_strong['snr_db']:.1f} with win_rate={best_strong['target_better_rate']:.4f}"
        )
    action_summary_file = report_root / "nsgrand_action_summary.csv"
    if action_summary_file.exists():
        action_df = pd.read_csv(action_summary_file)
        if not action_df.empty:
            action_totals = (
                action_df.groupby("policy_action", as_index=False)["count"].sum().sort_values("count", ascending=False)
            )
            top_action = action_totals.iloc[0]
            key_findings.append(
                f"Most common nsgrand policy action is {top_action['policy_action']} with count={int(top_action['count'])}"
            )

    report_md = [
        f"# Neuro-Symbolic GRAND report: {cfg.experiment_name}",
        "",
        "## Generated artifacts",
        "",
        "- `evaluation/evaluation_summary.csv` contains point-level metrics.",
        "- `evaluation/all_raw_records.csv.gz` contains per-sample decoder behavior across all operating points.",
        "- `evaluation/all_paired_records.csv.gz` aligns baseline and nsgrand on the same sample ids.",
        "- `evaluation/strong_vs_nsgrand_paired_records.csv.gz` aligns strong_symbolic and nsgrand on the same sample ids when the strong reference is enabled.",
        "- `reports/overview_by_profile_snr.csv` compiles BLER, query count, latency, fallback, and confidence summaries.",
        "- `reports/paired_summary_by_profile_snr.csv` contains paired-sample baseline-vs-nsgrand win/loss and delta statistics.",
        "- `reports/strong_vs_nsgrand_paired_summary.csv` compares nsgrand against the stronger symbolic reference when available.",
        "- `reports/nsgrand_gate_summary.csv` diagnoses how often each gate and fallback path is used.",
        "- `reports/nsgrand_action_summary.csv` summarizes high-level nsgrand policy actions.",
        "- `reports/interesting_traces_combined.jsonl` stores selected diagnostic traces for hard and representative cases.",
        "",
        "## Key findings",
        "",
    ]
    if key_findings:
        report_md.extend([f"- {item}" for item in key_findings])
    else:
        report_md.append("- No evaluation findings available.")
    (report_root / "report.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")

    # Build repo export bundle for GitHub push.
    export_manifest = {
        "experiment_name": cfg.experiment_name,
        "source_output_dir": str(output_dir),
        "exported_files": [],
    }
    export_items = [
        (eval_root / "evaluation_summary.csv", export_root / "evaluation_summary.csv"),
        (eval_root / "evaluation_summary.json", export_root / "evaluation_summary.json"),
        (eval_root / "all_raw_records.csv.gz", export_root / "all_raw_records.csv.gz"),
        (eval_root / "all_paired_records.csv.gz", export_root / "all_paired_records.csv.gz"),
        (eval_root / "strong_vs_nsgrand_paired_records.csv.gz", export_root / "strong_vs_nsgrand_paired_records.csv.gz"),
        (report_root / "overview_by_profile_snr.csv", export_root / "overview_by_profile_snr.csv"),
        (report_root / "paired_summary_by_profile_snr.csv", export_root / "paired_summary_by_profile_snr.csv"),
        (report_root / "strong_vs_nsgrand_paired_summary.csv", export_root / "strong_vs_nsgrand_paired_summary.csv"),
        (report_root / "metrics_by_true_error_weight.csv", export_root / "metrics_by_true_error_weight.csv"),
        (report_root / "nsgrand_gate_summary.csv", export_root / "nsgrand_gate_summary.csv"),
        (report_root / "nsgrand_action_summary.csv", export_root / "nsgrand_action_summary.csv"),
        (report_root / "nsgrand_confidence_bins.csv", export_root / "nsgrand_confidence_bins.csv"),
        (report_root / "nsgrand_overflow_bins.csv", export_root / "nsgrand_overflow_bins.csv"),
        (report_root / "nsgrand_tail_cases_top.csv", export_root / "nsgrand_tail_cases_top.csv"),
        (report_root / "interesting_traces_combined.jsonl", export_root / "interesting_traces_combined.jsonl"),
        (report_root / "report.md", export_root / "report.md"),
        (output_dir / "training" / "training_history.csv", export_root / "training_history.csv"),
        (output_dir / "training" / "training_summary.json", export_root / "training_summary.json"),
        (output_dir / "artifacts" / "resolved_config.json", export_root / "resolved_config.json"),
        (output_dir / "artifacts" / "runtime_snapshot.json", export_root / "runtime_snapshot.json"),
    ]
    for src, dst in export_items:
        if src.exists():
            copy_file(src, dst)
            export_manifest["exported_files"].append(str(dst.relative_to(export_root)))

    for plot_path in sorted(plots_root.glob("*.png")):
        dst = export_plots_root / plot_path.name
        copy_file(plot_path, dst)
        export_manifest["exported_files"].append(str(dst.relative_to(export_root)))

    (export_root / "README.md").write_text(
        "\n".join(
            [
                f"# GitHub-ready export bundle: {cfg.experiment_name}",
                "",
                "This directory is the compact bundle intended to be pushed to GitHub for later remote analysis.",
                "It excludes heavyweight dataset shards, checkpoints, and runtime logs, but retains the key aggregated tables,",
                "compressed per-sample records, selected traces, training summaries, and plots.",
                "",
                "Most useful files:",
                "- `evaluation_summary.csv`",
                "- `overview_by_profile_snr.csv`",
                "- `paired_summary_by_profile_snr.csv`",
                "- `all_raw_records.csv.gz`",
                "- `all_paired_records.csv.gz`",
                "- `strong_vs_nsgrand_paired_records.csv.gz`",
                "- `nsgrand_gate_summary.csv`",
                "- `nsgrand_action_summary.csv`",
                "- `interesting_traces_combined.jsonl`",
                "- `report.md`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(export_manifest, export_root / "manifest.json")

    summary = {
        "report_dir": str(report_root),
        "export_dir": str(export_root),
        "num_eval_points": int(len(summary_df)),
        "num_raw_records": int(len(raw_df)),
        "num_paired_records": int(len(paired_df)),
        "num_strong_pairwise_records": int(len(strong_pairwise_df)),
        "key_findings": key_findings,
    }
    write_json(summary, report_root / "report_summary.json")
    logger.info("Report artifacts written to %s", report_root)
    return summary
