# v11.1 installation fix

v11.0 declared mandatory dependencies in `pyproject.toml`, including `sionna>=0.19`.
On Compute Canada, that made `pip install -e .` try to resolve/install Sionna and its full dependency tree.
Because current PyPI has both legacy TensorFlow-based Sionna 0.19.x and newer Sionna 2.x releases,
pip backtracked across TensorFlow, NumPy, Sionna, Sionna-RT, matplotlib, contourpy, and PyYAML.
The observed terminal failure was a PyYAML source-build error, but that was only the final symptom.

v11.1 fixes this by:

* removing mandatory runtime dependencies from `pyproject.toml`;
* changing all Slurm scripts to use `python -m pip install -e . --no-deps --quiet`;
* changing `scripts/activate_env.sh` to use `--no-deps`;
* adding `scripts/fir_install_editable.sh` and `scripts/check_runtime_deps.py`;
* improving `_to_numpy()` in the Sionna wrapper to handle PyTorch tensors as well as TensorFlow tensors.

Use:

```bash
source .venv/bin/activate 2>/dev/null || true
python -m pip install -e . --no-deps
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/check_runtime_deps.py
```

If `sionna` is missing, install it using the same FIR/Compute Canada environment that supported the old v10 run.
Avoid unconstrained `pip install sionna` inside the project environment unless you intentionally want pip to update the numerical stack.
