# Hybrid GRAND v14.1 AI-Residual Standalone

This is a full standalone repo package. Unzip it into an empty `~/scratch/Neuro_Symbolic_GRAND` directory and run the command flow provided with the package.

v14 fixes the v13 label/search mismatch by training GRAND on residual failed-BP masks (`target_basis: bp`), filtering the curriculum to reachable target weights, injecting training-only oracle-positive reranker candidates, and using Tanner-aware neural features from channel statistics, BP traces, syndrome structure, and LDPC graph degrees.

The package intentionally does not include generated datasets/checkpoints. Generate, sanity-check, train, and evaluate on FIR.


## v14.1 smoke-generation fix

The v14 smoke job failed because dataset generation used multiprocessing on a GPU allocation and workers touched CUDA/TensorFlow/Sionna context. v14.1 makes generation CPU-only with spawned workers and runs only the generate stage with `CUDA_VISIBLE_DEVICES=""`; training and evaluation still use the H100.
