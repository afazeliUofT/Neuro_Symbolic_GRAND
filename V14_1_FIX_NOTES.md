# v14.1 fix notes

Observed failure in `slurm-hyb-grand-v14-smoke-37356880.out`:

```text
CUDA_ERROR_NOT_INITIALIZED ... BrokenProcessPool
```

Root cause: `generate.py` imported TensorFlow/Sionna-dependent channel code before generation workers hid CUDA, and the smoke slurm ran generation inside a GPU allocation. Multiprocessing workers then inherited/saw CUDA state and crashed while setting the CUDA context.

Fixes:

1. `generate.py` hides CUDA before TensorFlow/Sionna imports.
2. Generation workers force CPU-only, single-thread execution.
3. `ProcessPoolExecutor` uses `spawn`, avoiding inherited CUDA context.
4. Smoke slurm runs the generate stage with `CUDA_VISIBLE_DEVICES=""`; train/evaluate still use GPU.
5. CLI pipeline runs stages in separate Python processes.
