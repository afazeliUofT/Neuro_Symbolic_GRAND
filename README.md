# Hybrid GRAND v14 AI-Residual Standalone

This is a full standalone repo package. Unzip it into an empty `~/scratch/Neuro_Symbolic_GRAND` directory and run the command flow provided with the package.

v14 fixes the v13 label/search mismatch by training GRAND on residual failed-BP masks (`target_basis: bp`), filtering the curriculum to reachable target weights, injecting training-only oracle-positive reranker candidates, and using Tanner-aware neural features from channel statistics, BP traces, syndrome structure, and LDPC graph degrees.

The package intentionally does not include generated datasets/checkpoints. Generate, sanity-check, train, and evaluate on FIR.
