# v14.2 fix notes

The pushed `.out` logs visible on GitHub for the previous run installed version 14.0.0 and the smoke generation died in TensorFlow/CUDA multiprocessing with `CUDA_ERROR_NOT_INITIALIZED` and `BrokenProcessPool`.

v14.2 makes generation more robust by hiding CUDA with `CUDA_VISIBLE_DEVICES=-1` and `NVIDIA_VISIBLE_DEVICES=none`, explicitly hiding TensorFlow GPUs before Sionna/TensorFlow imports, using spawn workers, and cleaning stale v14 output directories before smoke/full generation.
