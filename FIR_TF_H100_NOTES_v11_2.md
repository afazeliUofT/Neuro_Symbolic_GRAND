# v11.2 FIR TensorFlow/H100 notes

This package was created after reviewing the FIR TensorFlow/GPU probe pushed to GitHub.

Observed FIR facts used for this release:

* `.venv` Python: 3.12.4.
* TensorFlow: 2.19.1+computecanada, CUDA build with CUDA 12.5.1 and cuDNN 9.
* Keras: 3.14.0.
* Sionna import path: `sionna.phy.fec.ldpc.encoding.LDPC5GEncoder` and `sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`.
* Sionna version at import: 1.2.2.
* Sionna encoder accepts NumPy and TensorFlow tensors; it rejects Torch tensors with `Tensor` has no `get_shape`.
* Generic `#SBATCH --gpus-per-node=1` is rejected on FIR. Slurm requires explicit GPU type: `h100`, `nvidia_h100_80gb_hbm3_3g.40gb`, `nvidia_h100_80gb_hbm3_2g.20gb`, or `nvidia_h100_80gb_hbm3_1g.10gb`.
* Available GPU partitions include `gpubase_bygpu_b1`, `gpubase_bygpu_b3`, `gpubase_bygpu_b4`, plus others.

Implementation changes relative to v11.1:

* `hybrid_bp_nsg/models/rescue_net.py` is TensorFlow/Keras.
* `hybrid_bp_nsg/training/train.py` is TensorFlow/Keras and saves `rescue_net_tf.weights.h5`.
* `hybrid_bp_nsg/decoders/grand_rescue.py` uses TensorFlow inference and reranking.
* `hybrid_bp_nsg/training/evaluation.py` loads TensorFlow weights and evaluates with the same channel-aligned Tanner-GRAND rescue.
* Slurm scripts request explicit FIR H100 GPU types and partitions.
* `scripts/check_tf_gpu.py` fails early if a GPU job does not expose a TensorFlow GPU.

The decoder-side conceptual changes from v11.1 remain unchanged: channel-aligned GRAND basis, fixed rank prior, no aggressive skip gate, and Tanner-syndrome OSD repair.
