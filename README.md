# Hybrid GRAND v14.2 AI-Residual Standalone

Standalone source package for the residual AI/Tanner-GRAND rescue decoder.

v14.2 preserves the v14 residual-label fixes and adds a stronger CPU-only generation guard for FIR/ComputeCanada Slurm jobs:

- package version marker `14.2.0`;
- generation uses `CUDA_VISIBLE_DEVICES=-1` and `NVIDIA_VISIBLE_DEVICES=none`;
- TensorFlow GPUs are explicitly hidden before Sionna/TensorFlow imports during generation;
- generation workers use Python `spawn` plus single-thread CPU settings;
- smoke/full generation slurms clean their own v14 output directory to avoid stale `.npz` files;
- every slurm prints the installed package version at the top.

The full command flow is in `COMMAND_FLOW.md`.
