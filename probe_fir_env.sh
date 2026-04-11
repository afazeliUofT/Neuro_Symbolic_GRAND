#!/bin/bash -l
set -uo pipefail

ROOT="/home/rsadve1/scratch/Neuro_Symbolic_GRAND"
LOGDIR="$ROOT/probe_logs"
TMPDIR="$ROOT/.tmp_probe"
PIP_CACHE_DIR="$ROOT/.pip_cache"

mkdir -p "$ROOT" "$LOGDIR" "$TMPDIR" "$PIP_CACHE_DIR"
cd "$ROOT"

export TMPDIR
export PIP_CACHE_DIR

exec > >(tee -a "$LOGDIR/probe_main_console.log") 2>&1

echo "=== BASIC INFO ==="
date -Is
hostname
whoami
pwd

echo
echo "=== MODULE DISCOVERY ==="
module reset || true
module -t avail python 2>&1 | tee "$LOGDIR/module_avail_python.txt"
module -t avail virtualenv 2>&1 | tee "$LOGDIR/module_avail_virtualenv.txt"
module spider python 2>&1 | tee "$LOGDIR/module_spider_python.txt"

if command -v avail_wheels >/dev/null 2>&1; then
    avail_wheels > "$LOGDIR/avail_wheels_all.txt"
    grep -E '^(torch|numpy|scipy|pandas|matplotlib|pyyaml|packaging|tqdm|pytest|pip|setuptools|wheel)[[:space:]]' \
        "$LOGDIR/avail_wheels_all.txt" > "$LOGDIR/avail_wheels_focus.txt" || true
else
    echo "avail_wheels command not found" | tee "$LOGDIR/avail_wheels_focus.txt"
fi

echo
echo "=== SYSTEM INFO ==="
uname -a | tee "$LOGDIR/uname.txt"
lscpu | tee "$LOGDIR/lscpu.txt"
free -h | tee "$LOGDIR/free.txt" || true

PY_MODULES=$(module -t avail python 2>&1 | grep -Eo 'python/3\.(11|12|13)(\.[0-9]+)?' | sort -Vu || true)

echo
echo "=== FILTERED PYTHON CANDIDATES ==="
echo "$PY_MODULES" | tee "$LOGDIR/python_modules_filtered.txt"

if [ -z "$PY_MODULES" ]; then
    echo "No python/3.11-3.13 modules found. Stopping."
    exit 1
fi

for PYMOD in $PY_MODULES; do
    TAG=$(echo "$PYMOD" | tr '/.' '_')
    ENV_DIR="$ROOT/.probe_env_${TAG}"

    echo
    echo "============================================================"
    echo "TESTING $PYMOD"
    echo "============================================================"

    module reset || true
    module load "$PYMOD" || { echo "FAILED: module load $PYMOD"; continue; }
    module load virtualenv || true

    {
        echo "PYMOD=$PYMOD"
        which python || true
        python -V || true
        python -m pip --version || true
    } | tee "$LOGDIR/${TAG}_python_info.txt"

    python - <<'PY' | tee "$LOGDIR/${TAG}_connectivity.txt"
import urllib.request
for url in [
    "https://pypi.org/simple/sionna-no-rt/",
    "https://files.pythonhosted.org/"
]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            print(url, "OK", getattr(r, "status", "NA"))
    except Exception as e:
        print(url, "FAIL", repr(e))
PY

    rm -rf "$ENV_DIR"
    if command -v virtualenv >/dev/null 2>&1; then
        virtualenv --no-download "$ENV_DIR"
    else
        python -m venv "$ENV_DIR"
    fi

    # shellcheck disable=SC1090
    source "$ENV_DIR/bin/activate"

    python -V | tee "$LOGDIR/${TAG}_venv_python_version.txt"
    which python | tee "$LOGDIR/${TAG}_venv_which_python.txt"

    python -m pip install --no-index --upgrade pip setuptools wheel \
        > "$LOGDIR/${TAG}_pip_upgrade.txt" 2>&1
    echo "exit_code=$?" >> "$LOGDIR/${TAG}_pip_upgrade.txt"

    python -m pip install --no-index torch numpy scipy pandas matplotlib pyyaml packaging tqdm pytest \
        > "$LOGDIR/${TAG}_wheelhouse_install.txt" 2>&1
    echo "exit_code=$?" >> "$LOGDIR/${TAG}_wheelhouse_install.txt"

    python - <<'PY' > "$LOGDIR/${TAG}_stack_imports.txt" 2>&1
import importlib, json
mods = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "pyyaml": "yaml",
    "packaging": "packaging",
    "tqdm": "tqdm",
    "pytest": "pytest",
}
out = {}
for name, modname in mods.items():
    try:
        m = importlib.import_module(modname)
        out[name] = {"ok": True, "version": getattr(m, "__version__", "n/a")}
    except Exception as e:
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
print(json.dumps(out, indent=2))

try:
    import torch
    print("\n=== torch.parallel_info() ===")
    print(torch.__config__.parallel_info())
    print("torch.get_num_threads() =", torch.get_num_threads())
    print("torch.get_num_interop_threads() =", torch.get_num_interop_threads())
    print("torch.cuda.is_available() =", torch.cuda.is_available())
except Exception as e:
    print("\nTorch info unavailable:", repr(e))
PY

    python -m pip install --dry-run --report "$LOGDIR/${TAG}_sionna_dryrun_report.json" -v sionna-no-rt \
        > "$LOGDIR/${TAG}_sionna_dryrun.txt" 2>&1
    echo "exit_code=$?" >> "$LOGDIR/${TAG}_sionna_dryrun.txt"

    python -m pip install -v sionna-no-rt \
        > "$LOGDIR/${TAG}_sionna_install.txt" 2>&1
    echo "exit_code=$?" >> "$LOGDIR/${TAG}_sionna_install.txt"

    python - <<'PY' > "$LOGDIR/${TAG}_sionna_import_check.txt" 2>&1
import importlib, json
status = {}
mods = [
    "sionna",
    "sionna.phy",
    "sionna.phy.channel.tr38901.tdl",
    "sionna.phy.channel.tr38901.cdl",
    "sionna.phy.nr.tb_encoder",
    "sionna.phy.nr.tb_decoder",
    "sionna.phy.nr.pusch_transmitter",
    "sionna.phy.nr.pusch_receiver",
]
for m in mods:
    try:
        importlib.import_module(m)
        status[m] = "OK"
    except Exception as e:
        status[m] = f"FAIL: {type(e).__name__}: {e}"
print(json.dumps(status, indent=2))

try:
    import importlib.metadata as md
    print("\n=== installed package metadata ===")
    print("sionna-no-rt version:", md.version("sionna-no-rt"))
    reqs = md.metadata("sionna-no-rt").get_all("Requires-Dist")
    print("Requires-Dist:")
    if reqs:
        for r in reqs:
            print("  ", r)
    else:
        print("   <none listed>")
except Exception as e:
    print("Could not read metadata:", repr(e))

try:
    import sionna, torch
    print("\nRuntime versions:")
    print("sionna.__version__ =", getattr(sionna, "__version__", "unknown"))
    print("torch.__version__  =", torch.__version__)
except Exception as e:
    print("Could not print runtime versions:", repr(e))
PY

    python -m pip freeze > "$LOGDIR/${TAG}_pip_freeze.txt" 2>&1

    deactivate
done

tar -czf "$ROOT/probe_logs_$(date +%Y%m%d_%H%M%S).tar.gz" -C "$ROOT" probe_logs
echo
echo "Probe finished. Logs are in $LOGDIR and the tarball is in $ROOT."
