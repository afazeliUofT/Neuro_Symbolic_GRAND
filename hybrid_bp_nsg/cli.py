from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Hybrid BP + AI/Tanner-GRAND rescue v14")
    p.add_argument("command", choices=["selftest", "generate", "train", "evaluate", "report", "pipeline", "plots"])
    p.add_argument("--config", "-c", default="configs/fir_hybrid_bp_nsg_full_v14.yaml")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "selftest":
        from .selftest import selftest
        selftest(cfg)
    elif args.command == "generate":
        from .generate import generate
        generate(cfg)
    elif args.command == "train":
        from .train import train
        train(cfg)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(cfg)
    elif args.command == "report":
        from .report import report
        report(cfg)
    elif args.command == "pipeline":
        # Run stages in fresh Python processes so CPU-only generation does not hide
        # the GPU from training/evaluation in the same interpreter.
        import subprocess
        for cmd in ["generate", "train", "evaluate", "report"]:
            subprocess.check_call([sys.executable, "-m", "hybrid_bp_nsg.cli", cmd, "--config", str(args.config)])
    elif args.command == "plots":
        from .report import report
        report(cfg)


if __name__ == "__main__":
    main()
