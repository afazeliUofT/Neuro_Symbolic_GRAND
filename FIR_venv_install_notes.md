# FIR venv decision for Neuro_Symbolic_GRAND

Chosen stack from the user's probe archive:

- Use `python/3.12.4`
- Do **not** use Python 3.13.x for this project
- Install `torch==2.11.0+computecanada`
- Install `tensorflow==2.19.1+computecanada`
- Install `sionna-no-rt==1.2.2`
- Keep `numpy==2.1.1+computecanada`

Why:

- Python 3.11.5 and 3.12.4 both successfully installed and imported the required Sionna NR/TR38901 modules.
- Python 3.13.x failed because compatible `tensorflow`/`h5py` distributions were not available in that environment.
- Letting pip resolve `sionna-no-rt` unconstrained caused it to examine 2.x first and then settle on 1.2.2.
- A controlled two-stage install is safer than a loose `pip install sionna-no-rt`.
