#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v14-eval-resume
#SBATCH --time=24:00:00
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:USR1@300

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG="configs/fir_hybrid_bp_nsg_full_v14.yaml"

bash scripts/bootstrap_fir_env.sh "$PWD"
source .venv/bin/activate
python -m pip install -e . --no-deps

python scripts/check_package_version.py

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TF_NUM_INTEROP_THREADS=8

PLAN_DIR="outputs/hybrid_bp_nsg_v14_pusch_cdl_full/evaluation_resume_plan"
PLAN_FILE="${PLAN_DIR}/plan.tsv"
mkdir -p "${PLAN_DIR}"

write_plan() {
python - "$CONFIG" "$PLAN_DIR" "$PLAN_FILE" <<'PY'
from __future__ import annotations

import copy
import csv
import re
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
plan_dir = Path(sys.argv[2])
plan_file = Path(sys.argv[3])

cfg = yaml.safe_load(config_path.read_text())
out_dir = Path(cfg["project"]["output_dir"])
eval_dir = out_dir / "evaluation"
tmp_cfg_dir = plan_dir / "configs"
tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

profiles = list(cfg.get("eval", {}).get("profiles", ["AWGN"]))
snrs = list(cfg.get("eval", {}).get("snr_db_grid", [0, 2, 4]))
decoders = list(cfg.get("eval", {}).get("stop_decoders", ["hybrid_bp_nsg", "bp_nms_20", "bp_nms_50"]))
samples_per_point = int(cfg.get("eval", {}).get("samples_per_point", 0))
base_seed = int(cfg.get("project", {}).get("seed", 31415))

def safe_name(x: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(x))

def snr_dir_name(snr: object) -> str:
    return f"snr_{float(snr):+0.1f}dB"

def summary_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        rows = list(csv.DictReader(path.open(newline="")))
    except Exception:
        return False
    if not rows:
        return False

    by_decoder = {str(r.get("decoder", "")): r for r in rows}
    for dec in decoders:
        if dec not in by_decoder:
            return False
        if samples_per_point > 0:
            try:
                samples = int(float(by_decoder[dec].get("samples", "0")))
            except Exception:
                return False
            if samples < samples_per_point:
                return False
    return True

rows = []
idx = 0
for profile in profiles:
    for snr in snrs:
        idx += 1
        point_cfg = copy.deepcopy(cfg)
        point_cfg["eval"]["profiles"] = [profile]
        point_cfg["eval"]["snr_db_grid"] = [float(snr)]

        # Keep the LDPC/code seed unchanged, but make the evaluation RNG seed
        # deterministic and different per point. This avoids every one-point
        # config using exactly the same RNG stream.
        point_cfg.setdefault("project", {})
        point_cfg["project"]["seed"] = base_seed + 10007 * idx

        cfg_name = f"eval_point_{idx:02d}_{safe_name(profile)}_{float(snr):+0.1f}dB.yaml"
        cfg_path = tmp_cfg_dir / cfg_name
        cfg_path.write_text(yaml.safe_dump(point_cfg, sort_keys=False))

        summary_path = eval_dir / f"profile_{profile}" / snr_dir_name(snr) / "summary.csv"
        status = "DONE" if summary_ok(summary_path) else "MISSING"
        rows.append((status, str(profile), f"{float(snr):.10g}", str(summary_path), str(cfg_path)))

with plan_file.open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["status", "profile", "snr", "summary", "config"])
    w.writerows(rows)

print(f"[PLAN] wrote {plan_file}")
for r in rows:
    print("[PLAN]", "\t".join(r))
PY
}

rebuild_summary() {
python - "$CONFIG" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text())

out_dir = Path(cfg["project"]["output_dir"])
eval_dir = out_dir / "evaluation"
eval_dir.mkdir(parents=True, exist_ok=True)

profiles = list(cfg.get("eval", {}).get("profiles", ["AWGN"]))
snrs = list(cfg.get("eval", {}).get("snr_db_grid", [0, 2, 4]))
decoders = list(cfg.get("eval", {}).get("stop_decoders", ["hybrid_bp_nsg", "bp_nms_20", "bp_nms_50"]))
samples_per_point = int(cfg.get("eval", {}).get("samples_per_point", 0))

def snr_dir_name(snr: object) -> str:
    return f"snr_{float(snr):+0.1f}dB"

all_rows = []
missing = []
fieldnames = None

for profile in profiles:
    for snr in snrs:
        path = eval_dir / f"profile_{profile}" / snr_dir_name(snr) / "summary.csv"
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
            continue

        try:
            rows = list(csv.DictReader(path.open(newline="")))
        except Exception:
            missing.append(str(path))
            continue

        by_decoder = {str(r.get("decoder", "")): r for r in rows}
        ok = True
        for dec in decoders:
            if dec not in by_decoder:
                ok = False
                break
            if samples_per_point > 0:
                try:
                    samples = int(float(by_decoder[dec].get("samples", "0")))
                except Exception:
                    ok = False
                    break
                if samples < samples_per_point:
                    ok = False
                    break

        if not ok:
            missing.append(str(path))
            continue

        if fieldnames is None and rows:
            fieldnames = list(rows[0].keys())

        # Preserve configured decoder order.
        for dec in decoders:
            all_rows.append(by_decoder[dec])

out = eval_dir / "evaluation_summary.csv"

if fieldnames and all_rows:
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[SUMMARY] wrote {out} rows={len(all_rows)}")
else:
    print("[SUMMARY] no complete rows available yet")

if missing:
    print("[SUMMARY] missing/incomplete point summaries:")
    for m in missing:
        print("  ", m)
    Path(eval_dir / "evaluation_summary.MISSING").write_text("\n".join(missing) + "\n")
else:
    miss = eval_dir / "evaluation_summary.MISSING"
    if miss.exists():
        miss.unlink()
    print("[SUMMARY] all configured profile/SNR points are complete")
PY
}

on_signal() {
    echo "[SIGNAL] received wall-time/preemption signal; rebuilding summary from completed points."
    rebuild_summary || true
    exit 0
}

trap on_signal USR1 TERM

python scripts/check_tf_gpu.py --require-gpu
python scripts/check_config.py "$CONFIG"
python scripts/check_channel_models.py --config "$CONFIG" --profiles AWGN CDL_C

write_plan

echo
echo "========== Evaluation resume plan =========="
cat "$PLAN_FILE"
echo "============================================"
echo

# Run only missing profile/SNR points. Completed points are skipped.
tail -n +2 "$PLAN_FILE" | while IFS=$'\t' read -r status profile snr summary cfg_path; do
    if [[ "$status" == "DONE" ]]; then
        echo "[SKIP] profile=${profile} snr=${snr} summary=${summary}"
        continue
    fi

    echo
    echo "################################################################################"
    echo "[RUN] profile=${profile} snr=${snr}"
    echo "[RUN] point config=${cfg_path}"
    echo "[RUN] expected summary=${summary}"
    echo "################################################################################"
    echo

    python -m hybrid_bp_nsg.cli evaluate --config "$cfg_path"

    echo
    echo "[POINT DONE] profile=${profile} snr=${snr}; rebuilding combined summary."
    rebuild_summary
done

# Recompute plan after the run. If anything is still missing, stop before report.
write_plan

echo
echo "========== Post-run evaluation resume plan =========="
cat "$PLAN_FILE"
echo "====================================================="
echo

if grep -q '^MISSING' "$PLAN_FILE"; then
    echo "[STOP] Some evaluation points are still missing or incomplete."
    echo "[STOP] Re-submit this same slurm later; completed points will be skipped."
    rebuild_summary || true
    exit 10
fi

rebuild_summary

echo
echo "################################################################################"
echo "[REPORT] all profile/SNR points complete; generating final report"
echo "################################################################################"
echo

python -m hybrid_bp_nsg.cli report --config "$CONFIG"

echo
echo "[DONE] resumable evaluation/report complete."
