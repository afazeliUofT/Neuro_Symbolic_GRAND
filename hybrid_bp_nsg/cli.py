from __future__ import annotations

import argparse
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
import json

from .config import load_config
from .training.dataset import generate_supervised_dataset, split_complete
from .training.train import train_rescue_model
from .training.evaluation import evaluate_grid
from .analysis.reporting import build_reports
from .utils.io import ensure_dir, save_json
from .utils.logging import get_logger


def _prepare_output(cfg):
    output_dir = Path(cfg["project"]["output_dir"])
    ensure_dir(output_dir / "artifacts")
    save_json(cfg, output_dir / "artifacts" / "resolved_config.json")
    runtime = {
        "hostname": socket.gethostname(),
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }
    save_json(runtime, output_dir / "artifacts" / "runtime_snapshot.json")
    return output_dir


def _stage_complete(action: str, output_dir: Path, cfg) -> bool:
    if action == "generate":
        return (
            (output_dir / "artifacts" / "code_summary.json").exists()
            and split_complete(output_dir, "train", int(cfg["data"]["train_shards"]))
            and split_complete(output_dir, "val", int(cfg["data"]["val_shards"]))
        )
    if action == "train":
        return (output_dir / "checkpoints" / "rescue_net.pt").exists() and (output_dir / "training" / "training_summary.json").exists()
    if action in {"evaluate", "tail_evaluate"}:
        return (output_dir / "evaluation" / "evaluation_summary.csv").exists() and (output_dir / "evaluation" / "all_raw_records.csv.gz").exists()
    if action == "report":
        return (output_dir / "reports" / "report.md").exists() and (output_dir / "TWC_plots" / "manifest.csv").exists() and (output_dir / "TWC_plots" / "README.md").exists()
    return False


def _assert_stage_outputs(action: str, output_dir: Path, cfg) -> None:
    if not _stage_complete(action, output_dir, cfg):
        if action == "generate":
            raise FileNotFoundError(
                f"Stage generate incomplete: expected {cfg['data']['train_shards']} train shards, "
                f"{cfg['data']['val_shards']} val shards, and artifacts/code_summary.json in {output_dir}"
            )
        raise FileNotFoundError(f"Stage {action} did not produce expected outputs in {output_dir}")


def _mark_success(output_dir: Path, action: str) -> None:
    save_json({"action": action, "ok": True, "utc_time": datetime.now(timezone.utc).isoformat()}, output_dir / "artifacts" / f"{action}_success.json")


def _run_generate(cfg, output_dir, logger):
    if _stage_complete("generate", output_dir, cfg):
        logger.info("Generate stage already complete; skipping")
    else:
        generate_supervised_dataset(cfg, output_dir, logger)
    _assert_stage_outputs("generate", output_dir, cfg)
    _mark_success(output_dir, "generate")


def _run_train(cfg, output_dir, logger):
    if _stage_complete("train", output_dir, cfg):
        logger.info("Train stage already complete; skipping")
    else:
        train_rescue_model(cfg, output_dir, logger)
    _assert_stage_outputs("train", output_dir, cfg)
    _mark_success(output_dir, "train")


def _run_evaluate(cfg, output_dir, logger, tail=False):
    action = "tail_evaluate" if tail else "evaluate"
    if _stage_complete(action, output_dir, cfg):
        logger.info("%s stage already complete; skipping", action)
    else:
        evaluate_grid(cfg, output_dir, logger, tail=tail)
    _assert_stage_outputs(action, output_dir, cfg)
    _mark_success(output_dir, action)


def _run_report(cfg, output_dir, logger, tail_summary_path=None):
    # Report is cheap; always regenerate when requested so plots reflect latest eval data.
    build_reports(cfg, output_dir, logger, tail_summary_path=tail_summary_path)
    _assert_stage_outputs("report", output_dir, cfg)
    _mark_success(output_dir, "report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid BP + Neuro-Symbolic GRAND rescue pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("action", choices=["generate", "train", "evaluate", "tail_evaluate", "report", "pipeline"])
    parser.add_argument("--tail-summary-path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = _prepare_output(cfg)
    logger = get_logger("hybrid_bp_nsg", output_dir / "logs" / f"{args.action}.log")

    if args.action == "generate":
        _run_generate(cfg, output_dir, logger)
    elif args.action == "train":
        _run_train(cfg, output_dir, logger)
    elif args.action == "evaluate":
        _run_evaluate(cfg, output_dir, logger, tail=False)
    elif args.action == "tail_evaluate":
        _run_evaluate(cfg, output_dir, logger, tail=True)
    elif args.action == "report":
        _run_report(cfg, output_dir, logger, tail_summary_path=args.tail_summary_path)
    elif args.action == "pipeline":
        logger.info("Starting resumable pipeline. Re-running this command is safe after wall-time cancellation.")
        _run_generate(cfg, output_dir, logger)
        _run_train(cfg, output_dir, logger)
        _run_evaluate(cfg, output_dir, logger, tail=False)
        _run_report(cfg, output_dir, logger, tail_summary_path=args.tail_summary_path)
        _mark_success(output_dir, "pipeline")


if __name__ == "__main__":
    main()
