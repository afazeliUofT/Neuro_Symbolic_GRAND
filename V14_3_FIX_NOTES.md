# v14.3 fix notes

The v14.2 smoke job reached training and failed inside `tf.einsum` because Keras mixed precision cast Tanner graph states to bfloat16 while the LDPC parity matrix constants were float32. TensorFlow requires Einsum inputs to have matching dtypes.

v14.3 explicitly keeps the Tanner message-passing path in float32 while still allowing the configured mixed-precision policy. It also adds `scripts/check_model_forward.py` and calls it from smoke/train slurms before the expensive training loop.

Expected smoke behavior: CPU-only generation may still print `CUDA_ERROR_NO_DEVICE` messages from TensorFlow when CUDA is intentionally hidden, but it should complete generation, pass the dataset sanity probe, pass model forward, train, evaluate, and report.
