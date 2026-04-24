# v12 changelog versus the GitHub repo

## Decoder and search

- Rescue now prefers **CRC-valid codewords** rather than merely parity-valid ones.
- Added `require_crc_for_accept` for outer-CRC rejection of wrong valid codewords.
- Added codeword-level candidate features and selection metrics.
- Kept channel-aligned GRAND base and Tanner-syndrome OSD/greedy repair.
- Fixed the AI inverse-rank prior indexing issue in the TensorFlow rescue path.

## Training target

- Candidate reranking is now trained on **candidate codewords** instead of only residual masks.
- Candidate bank construction injects an oracle-positive candidate when needed for supervision.
- Saved both final and validation-best Keras weight files.

## Evaluation/reporting

- Measures latency for all decoders fairly.
- Records CRC failure rates and counts of CRC-valid / parity-valid rescue candidates.
- Evaluates with the best checkpoint when present.

## 5G NR / coding

- Added `sionna_nr_pusch_ldpc` code family.
- Uses Sionna NR transport-block sizing, TB CRC/interleaver metadata, and the standard 5G LDPC code block underneath.
- Default full/smoke/tail configs now use a short `num_coded_bits = 512` setup.

## Runtime

- TensorFlow runtime helper now supports XLA, mixed precision, and explicit CPU-thread control.
- Slurm scripts updated for FIR H100 and 32-CPU usage.
- Generation path parallelized across CPU workers with oversubscription protection.


## v12.0.1 hotfix

- Fixed CRC validation for `sionna_nr_pusch_ldpc`. The previous `crc_check_internal()` routed a noiseless codeword through `TBDecoder`, which can be tripped by decoder-input sign-convention ambiguities on some Sionna/FIR builds. v12.0.1 now validates the outer TB CRC directly in the internal systematic domain by regenerating the expected code-block input (`payload + TB CRC + zero padding`) and comparing it with the first `cb_size` internal bits.
- Removed the silent fallback that previously skipped TB CRC attachment if `tb_crc_encoder` raised. This now fails loudly during code construction instead of creating CRC-invalid training data.
- `scripts/check_pusch_nr_chain.py` now also checks that `internal_to_tx` matches the Sionna `TBEncoder` output exactly.
