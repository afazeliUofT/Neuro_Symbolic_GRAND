#!/bin/bash -l
set -euo pipefail

ROOT="/home/rsadve1/scratch/Neuro_Symbolic_GRAND"
VENV_DIR="$ROOT/.venv"
REQ_WHEEL="$ROOT/requirements_fir_py312_wheelhouse.txt"
REQ_PYPI="$ROOT/requirements_fir_py312_pypi.txt"
VERIFY_SCRIPT="$ROOT/verify_fir_env.py"

mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -f .gitignore ]; then
    touch .gitignore
fi
for p in ".venv/" "venv/" ".probe_env_*/" ".pip_cache/" "probe_logs/" "probe_logs_*.tar.gz"; do
    grep -qxF "$p" .gitignore || echo "$p" >> .gitignore
done

module reset
module load python/3.12.4

python -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export PIP_CACHE_DIR="$ROOT/.pip_cache"
mkdir -p "$PIP_CACHE_DIR"

python -m pip install --no-index --upgrade pip setuptools wheel
python -m pip install --no-index -r "$REQ_WHEEL"
python -m pip install --no-deps -r "$REQ_PYPI"

python "$VERIFY_SCRIPT"

echo
echo "Venv installation completed successfully at: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
