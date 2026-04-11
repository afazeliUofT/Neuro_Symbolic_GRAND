import importlib
import json
import os
import platform

mods = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "yaml": "yaml",
    "tensorflow": "tensorflow",
    "sionna": "sionna",
    "sionna_tdl": "sionna.phy.channel.tr38901.tdl",
    "sionna_cdl": "sionna.phy.channel.tr38901.cdl",
    "sionna_tb_encoder": "sionna.phy.nr.tb_encoder",
    "sionna_tb_decoder": "sionna.phy.nr.tb_decoder",
    "sionna_pusch_tx": "sionna.phy.nr.pusch_transmitter",
    "sionna_pusch_rx": "sionna.phy.nr.pusch_receiver",
}

out = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "env": {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "TF_NUM_INTRAOP_THREADS": os.environ.get("TF_NUM_INTRAOP_THREADS"),
        "TF_NUM_INTEROP_THREADS": os.environ.get("TF_NUM_INTEROP_THREADS"),
    },
    "imports": {},
}

for name, modname in mods.items():
    try:
        m = importlib.import_module(modname)
        out["imports"][name] = {
            "ok": True,
            "version": getattr(m, "__version__", "n/a"),
        }
    except Exception as e:
        out["imports"][name] = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }

try:
    import torch
    out["torch_parallel_info"] = torch.__config__.parallel_info()
    out["torch_num_threads"] = torch.get_num_threads()
    out["torch_num_interop_threads"] = torch.get_num_interop_threads()
    out["torch_cuda_available"] = torch.cuda.is_available()
except Exception as e:
    out["torch_parallel_info_error"] = repr(e)

try:
    import importlib.metadata as md
    out["package_versions"] = {
        "sionna-no-rt": md.version("sionna-no-rt"),
        "tensorflow": md.version("tensorflow"),
        "torch": md.version("torch"),
        "numpy": md.version("numpy"),
    }
except Exception as e:
    out["package_versions_error"] = repr(e)

print(json.dumps(out, indent=2))
