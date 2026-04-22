from __future__ import annotations
import os, sys, platform, importlib, traceback, subprocess, json, time
from importlib import metadata

def banner(name):
    print("\n" + "="*88)
    print(f"[BEGIN {name}]")
    print("="*88, flush=True)

def end(name):
    print("="*88)
    print(f"[END {name}]")
    print("="*88, flush=True)

def safe(name, fn):
    banner(name)
    try:
        fn()
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        end(name)

def meta_version(pkg):
    try:
        return metadata.version(pkg)
    except Exception as e:
        return f"metadata unavailable: {e}"

def try_import(name):
    try:
        mod = importlib.import_module(name)
        print(f"{name}: IMPORT OK")
        print(f"  __version__={getattr(mod, '__version__', 'unknown')}")
        print(f"  __file__={getattr(mod, '__file__', 'unknown')}")
        return mod
    except Exception as e:
        print(f"{name}: IMPORT FAILED: {type(e).__name__}: {e}")
        return None

safe("basic_python_env", lambda: (
    print("sys.executable:", sys.executable),
    print("sys.version:", sys.version.replace("\n", " ")),
    print("platform:", platform.platform()),
    print("cwd:", os.getcwd()),
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")),
    print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID")),
    print("SLURM_GPUS:", os.environ.get("SLURM_GPUS")),
    print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS")),
    print("SLURM_STEP_GPUS:", os.environ.get("SLURM_STEP_GPUS")),
    print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK")),
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH")),
))

safe("package_metadata", lambda: [
    print(f"{pkg}: {meta_version(pkg)}")
    for pkg in [
        "numpy", "scipy", "tensorflow", "keras", "torch", "sionna",
        "sionna-phy", "sionna-sys", "sionna-rt", "pyyaml", "h5py"
    ]
])

def probe_numpy():
    np = try_import("numpy")
    if np is not None:
        print("numpy version:", np.__version__)
        print("numpy file:", np.__file__)
safe("numpy", probe_numpy)

def probe_tensorflow():
    tf = try_import("tensorflow")
    if tf is None:
        return
    print("tf version:", tf.__version__)
    print("tf file:", tf.__file__)
    try:
        print("tf build info:", json.dumps(tf.sysconfig.get_build_info(), indent=2, default=str))
    except Exception as e:
        print("tf build info failed:", e)
    try:
        print("tf built with cuda:", tf.test.is_built_with_cuda())
    except Exception as e:
        print("tf built-with-cuda check failed:", e)
    try:
        devices = tf.config.list_physical_devices()
        print("all physical devices:", devices)
        gpus = tf.config.list_physical_devices("GPU")
        print("gpu physical devices:", gpus)
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("set memory growth OK:", gpu)
            except Exception as e:
                print("set memory growth failed:", gpu, e)
            try:
                print("gpu details:", tf.config.experimental.get_device_details(gpu))
            except Exception as e:
                print("gpu details failed:", e)
        if gpus:
            with tf.device("/GPU:0"):
                a = tf.random.normal((2048, 2048))
                b = tf.linalg.matmul(a, a)
                print("gpu matmul result shape:", b.shape, "mean:", float(tf.reduce_mean(b).numpy()))
        else:
            a = tf.random.normal((512, 512))
            b = tf.linalg.matmul(a, a)
            print("cpu tf matmul result shape:", b.shape, "mean:", float(tf.reduce_mean(b).numpy()))
    except Exception as e:
        print("tf device/matmul probe failed:", type(e).__name__, e)
        traceback.print_exc()
safe("tensorflow", probe_tensorflow)

def probe_torch():
    torch = try_import("torch")
    if torch is None:
        return
    print("torch version:", torch.__version__)
    print("torch file:", torch.__file__)
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    try:
        print("torch cuda available:", torch.cuda.is_available())
        print("torch cuda device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"torch gpu {i}:", torch.cuda.get_device_name(i))
        if torch.cuda.is_available():
            x = torch.randn(2048, 2048, device="cuda")
            y = x @ x
            print("torch gpu matmul:", tuple(y.shape), float(y.mean().detach().cpu()))
    except Exception as e:
        print("torch cuda probe failed:", type(e).__name__, e)
        traceback.print_exc()
safe("torch", probe_torch)

def probe_sionna():
    sionna = try_import("sionna")
    if sionna is None:
        return
    paths = [
        ("new_phy_encoding", "sionna.phy.fec.ldpc.encoding", "LDPC5GEncoder"),
        ("new_phy_decoding", "sionna.phy.fec.ldpc.decoding", "LDPC5GDecoder"),
        ("old_encoding", "sionna.fec.ldpc.encoding", "LDPC5GEncoder"),
        ("old_decoding", "sionna.fec.ldpc.decoding", "LDPC5GDecoder"),
        ("flat_ldpc", "sionna.fec.ldpc", "LDPC5GEncoder"),
    ]
    enc_cls = None
    for label, mod_name, cls_name in paths:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            print(f"{label}: OK {mod_name}.{cls_name} -> {cls}")
            if cls_name == "LDPC5GEncoder" and enc_cls is None:
                enc_cls = cls
        except Exception as e:
            print(f"{label}: FAILED {mod_name}.{cls_name}: {type(e).__name__}: {e}")
    if enc_cls is None:
        print("No LDPC5GEncoder found.")
        return

    try:
        import numpy as np
        enc = enc_cls(k=256, n=512)
        print("LDPC5GEncoder instantiated with keyword args.")
    except Exception as e1:
        try:
            import numpy as np
            enc = enc_cls(256, 512)
            print("LDPC5GEncoder instantiated with positional args.")
        except Exception as e2:
            print("LDPC5GEncoder instantiation failed:", type(e1).__name__, e1, "|", type(e2).__name__, e2)
            return

    for attr in ["k", "n", "_k", "_n", "coderate", "_coderate", "pcm", "_pcm", "pcm_a", "_pcm_a", "num_bits_per_symbol", "_bg", "bg", "_z", "z"]:
        try:
            val = getattr(enc, attr)
            if hasattr(val, "shape"):
                print(f"encoder attr {attr}: shape={val.shape} type={type(val)}")
            else:
                print(f"encoder attr {attr}: {val}")
        except Exception:
            pass

    # Try one encode call using whichever tensor backend works.
    try:
        import numpy as np
        u_np = np.zeros((1, 256), dtype=np.float32)
        tried = []
        try:
            out = enc(u_np)
            tried.append(("numpy", out))
        except Exception as e:
            print("encode numpy failed:", type(e).__name__, e)
        try:
            import tensorflow as tf
            out = enc(tf.zeros((1, 256), dtype=tf.float32))
            tried.append(("tensorflow", out))
        except Exception as e:
            print("encode tensorflow failed:", type(e).__name__, e)
        try:
            import torch
            out = enc(torch.zeros((1, 256), dtype=torch.float32))
            tried.append(("torch", out))
        except Exception as e:
            print("encode torch failed:", type(e).__name__, e)
        for kind, out in tried:
            print(f"encode {kind} OK: type={type(out)} shape={getattr(out, 'shape', None)}")
    except Exception as e:
        print("encode probe wrapper failed:", type(e).__name__, e)
safe("sionna", probe_sionna)

def probe_nvidia_smi():
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True, timeout=20)
        print(out)
    except Exception as e:
        print("nvidia-smi failed:", type(e).__name__, e)
safe("nvidia_smi_from_python", probe_nvidia_smi)
