# Hybrid BP + Channel-Aligned TensorFlow AI/Tanner-GRAND Rescue — v11.2

This is the FIR-specific TensorFlow/Keras GPU package derived from v11.1.

The research-side fix from v11.1 is preserved:

* The old package trained and searched for the mask `BP_final_hard XOR true_codeword`. On the failed BP states this residual had mean weight about 88–89 bits, so a low/medium-weight GRAND search could reach only a tiny fraction of failures.
* v11 trains and searches for the **channel noise / channel-basis correction** `GRAND_base_hard XOR true_codeword`, where `GRAND_base_hard` is the channel hard decision on transmitted positions and the BP posterior decision on punctured positions.
* The rescue decoder applies candidate masks to this GRAND base, not to the failed BP hard output.
* The AI rank-prior indexing bug is fixed.
* A Tanner-syndrome OSD repair stage solves `H[:, support] e = syndrome(base)` over GF(2) on increasingly large low-cost supports.

v11.2 changes the implementation side:

* The neural rescue model, training loop, checkpointing, and AI inference path are TensorFlow/Keras based.
* PyTorch is no longer required for training/evaluation.
* The Slurm scripts request FIR GPUs using explicit H100 GPU types and partitions seen in the FIR probe.
* A GPU preflight script, `scripts/check_tf_gpu.py`, verifies that TensorFlow sees the allocated GPU before training or evaluation starts.

## Why TensorFlow in v11.2

The FIR probe showed that the active `.venv` contains TensorFlow 2.19.1, Keras 3.14.0, Torch 2.11.0, and Sionna 1.2.2/no-RT. TensorFlow is CUDA-built, and the installed Sionna LDPC encoder accepts NumPy/TensorFlow tensors but not Torch tensors. Therefore, the correct FIR-native implementation for this environment is TensorFlow/Keras.

## Install

Use the `.venv` that already contains TensorFlow and Sionna on FIR.

```bash
cd /home/rsadve1/scratch
rm -rf Neuro_Symbolic_GRAND
unzip Hybrid_GRAND_v11_2_TF_H100_GPU.zip -d Neuro_Symbolic_GRAND
cd Neuro_Symbolic_GRAND

source .venv/bin/activate
python -m pip install -e . --no-deps
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/check_runtime_deps.py || true
python scripts/check_tf_gpu.py || true      # login node normally has no GPU
```

## Recommended run order

Selftest is CPU/PEG and should run before GPU jobs:

```bash
sbatch slurm/fir_hybrid_bp_nsg_selftest.sbatch
```

Full 5G generation is CPU-heavy and does not request GPU:

```bash
sbatch slurm/fir_hybrid_bp_nsg_generate.sbatch
```

Training and evaluation request one full H100:

```bash
sbatch slurm/fir_hybrid_bp_nsg_train.sbatch
sbatch slurm/fir_hybrid_bp_nsg_evaluate_report.sbatch
```

Smoke GPU run:

```bash
sbatch slurm/fir_hybrid_bp_nsg_smoke.sbatch
```

A single all-in-one GPU run is also provided:

```bash
sbatch slurm/fir_hybrid_bp_nsg_pipeline.sbatch
```

If the full-H100 queue is slow or unavailable, MIG 40GB alternatives are included:

```bash
sbatch slurm/fir_hybrid_bp_nsg_train_gpu_mig40gb.sbatch
sbatch slurm/fir_hybrid_bp_nsg_evaluate_report_gpu_mig40gb.sbatch
```

## FIR Slurm GPU choices

The probe showed GPU partitions such as `gpubase_bygpu_b1`, `gpubase_bygpu_b3`, and `gpubase_bygpu_b4`, and GPU types including:

```text
h100
nvidia_h100_80gb_hbm3_3g.40gb
nvidia_h100_80gb_hbm3_2g.20gb
nvidia_h100_80gb_hbm3_1g.10gb
```

v11.2 uses:

* `h100:1` for the main train/evaluate/pipeline Slurms.
* `nvidia_h100_80gb_hbm3_3g.40gb:1` for the MIG alternatives.
* `gpubase_bygpu_b3` for 24-hour train/evaluate jobs.
* `gpubase_bygpu_b4` for the 48-hour all-in-one pipeline job.
* `gpubase_bygpu_b1` for the short smoke job.

## Output directories

The default full config writes to:

```text
outputs/hybrid_bp_nsg_v11_channel_aligned_full/
```

The smoke and selftest configs write to separate output directories.

## Notes

This package still compares against `bp_nms_20` and `bp_nms_50`. The TensorFlow rewrite is not a new decoding theory by itself; it makes the neural component and Sionna stack align with FIR's actual runtime and makes GPU allocation explicit and verifiable.
