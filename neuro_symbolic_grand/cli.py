from __future__ import annotations

import argparse
from pathlib import Path

from .analysis.reporting import build_reports
from .config import ExperimentConfig, load_config
from .training.data_generation import generate_train_val_datasets
from .training.evaluation import evaluate_grid
from .training.trainer import train_model
from .utils.io import write_json
from .utils.logging_utils import setup_logging


def _resolve_output_dir(cfg: ExperimentConfig) -> Path:
    return Path(cfg.output_dir)


def _run_generate(cfg: ExperimentConfig) -> None:
    output_dir = _resolve_output_dir(cfg)
    logger = setup_logging(output_dir / "logs" / "generate.log")
    write_json(cfg.to_dict(), output_dir / "artifacts" / "resolved_config.json")
    generate_train_val_datasets(cfg, output_dir, logger)


def _run_train(cfg: ExperimentConfig) -> None:
    output_dir = _resolve_output_dir(cfg)
    logger = setup_logging(output_dir / "logs" / "train.log")
    train_model(cfg, output_dir, logger)


def _run_evaluate(cfg: ExperimentConfig) -> None:
    output_dir = _resolve_output_dir(cfg)
    logger = setup_logging(output_dir / "logs" / "evaluate.log")
    evaluate_grid(cfg, output_dir, logger)


def _run_report(cfg: ExperimentConfig) -> None:
    output_dir = _resolve_output_dir(cfg)
    logger = setup_logging(output_dir / "logs" / "report.log")
    build_reports(cfg, output_dir, logger)


def _run_pipeline(cfg: ExperimentConfig) -> None:
    _run_generate(cfg)
    _run_train(cfg)
    _run_evaluate(cfg)
    _run_report(cfg)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neuro-Symbolic GRAND research pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "train", "evaluate", "report", "pipeline"):
        subparsers.add_parser(name)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "generate":
        _run_generate(cfg)
    elif args.command == "train":
        _run_train(cfg)
    elif args.command == "evaluate":
        _run_evaluate(cfg)
    elif args.command == "report":
        _run_report(cfg)
    elif args.command == "pipeline":
        _run_pipeline(cfg)
    else:
        raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
