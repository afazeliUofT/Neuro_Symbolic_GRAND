# Hybrid BP + PUSCH-aligned CRC-aware AI/Tanner-GRAND Rescue — v12.0

This package is a standalone FIR-oriented replacement for the current GitHub version.

It keeps the TensorFlow/Keras FIR-native stack and the channel-aligned GRAND basis from v11, and adds the missing pieces that the GitHub results showed were necessary:

- **5G NR PUSCH-derived coding path** using Sionna NR transport-block sizing and transport-block encode/decode utilities.
- **Standard 5G NR LDPC code block** underneath the rescue decoder.
- **Outer CRC-aware acceptance** so parity-valid but wrong codewords are rejected instead of being counted as “rescues”.
- **Codeword-level candidate-bank reranking** rather than only residual-mask reachability training.
- **Fair latency logging** for `bp_nms_20`, `bp_nms_50`, and the hybrid decoder.
- **Best-checkpoint evaluation** instead of always using the final epoch.
- **GPU graph/XLA mode** for the neural component and **32-CPU-aware** dataset generation/training scripts for FIR.

## What changed versus the GitHub repo

The current GitHub repo still has the following research-side limitations:

1. The rescue stage can accept **wrong zero-syndrome codewords** because parity validity is treated as enough.
2. The reranker is trained mostly on **target-mask reachability**, not on selecting the correct codeword from a valid candidate list.
3. Baseline BP latency is not measured fairly, which makes legacy LDPC appear artificially “near zero latency”.
4. The 5G side is based on a standard LDPC code, but not on a **PUSCH-derived TB/CRC/interleaver path**.

v12 addresses each of those.

## 5G NR / PUSCH scope

This package uses Sionna NR utilities to derive a **short-block PUSCH-style transport-block configuration** and then builds the underlying standard 5G LDPC code block used by the hybrid BP + rescue decoder.

In particular, the `sionna_nr_pusch_ldpc` code family uses Sionna NR:

- `calculate_tb_size()` for 5G-consistent short-block sizing,
- `TBEncoder` / `TBDecoder` for the transport-block CRC/interleaver path,
- the standard 5G NR LDPC encoder/PCM for the code block that the hybrid decoder operates on.

The default full configuration uses:

- `num_coded_bits = 512`
- `target_coderate = 0.5`
- `num_bits_per_symbol = 2` (QPSK)
- `num_layers = 1`

This keeps the coded block short while preserving the necessary 5G NR CRC/interleaver overhead.

## Important modeling note

The coding chain is PUSCH-derived and 5G-NR-consistent at the transport-block / CRC / LDPC / output-interleaver level.

The channel/equalization loop is still a **bit-domain research simulator** with profiles A/C/E used in the prior repo:

- A: AWGN BPSK/QPSK-equivalent bit-domain channel
- C: coherent memoryless Rayleigh-style fading with perfect CSI at the LLR stage
- E: mild impulsive-noise mixture

That means this package is **not a full OFDM resource-grid PUSCH receiver simulation** with DMRS channel estimation and equalization. It is a PUSCH-derived coding + CRC + interleaver stack designed to make the hybrid rescue research question well-posed and 5G-consistent without making the experiment prohibitively slow.

## FIR install

Use the existing FIR `.venv` that already contains TensorFlow and Sionna.

```bash
cd /home/rsadve1/scratch
rm -rf Neuro_Symbolic_GRAND
unzip Hybrid_GRAND_v12_PUSCH_CRC_LISTWISE.zip -d Neuro_Symbolic_GRAND
cd Neuro_Symbolic_GRAND

source .venv/bin/activate
python -m pip install -e . --no-deps
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/check_runtime_deps.py
python scripts/check_tf_gpu.py || true
```

## Recommended run order

```bash
sbatch slurm/fir_hybrid_bp_nsg_selftest.sbatch
sbatch slurm/fir_hybrid_bp_nsg_smoke.sbatch
sbatch slurm/fir_hybrid_bp_nsg_generate.sbatch
sbatch slurm/fir_hybrid_bp_nsg_train.sbatch
sbatch slurm/fir_hybrid_bp_nsg_evaluate_report.sbatch
```

Or all at once:

```bash
sbatch slurm/fir_hybrid_bp_nsg_pipeline.sbatch
```

## Outputs

The full run writes to:

```text
outputs/hybrid_bp_nsg_v12_pusch_crc_full/
```

## Notes on speed

- Train/eval Slurms request **one H100** and **32 CPUs**.
- Training uses TensorFlow graph mode with optional XLA and mixed precision.
- Dataset generation uses **spawned CPU workers** and is configured for a 32-core FIR allocation.
- Generation workers force their internal BLAS/OMP thread counts to 1 to avoid oversubscription.
