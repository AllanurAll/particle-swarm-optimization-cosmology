# Reproducibility

The production results are deterministic because every PSO calculation uses an explicit NumPy random seed.

## Production settings

| Calculation | Particles | Iterations | Seed |
|---|---:|---:|---:|
| Flat Lambda-CDM | 100 | 300 | 42 |
| Flat Lambda-CDM in curvature comparison | 200 | 500 | 42 |
| Non-flat Lambda-CDM | 200 | 500 | 42 |
| Flat Lambda-CDM baseline in XCDM comparison | 200 | 500 | 42 |
| Flat XCDM | 250 | 700 | 43 |
| Non-flat XCDM | 300 | 900 | 44 |

All calculations use inertia 0.72 and cognitive/social coefficients 1.49.


## Reusing an existing environment

When testing a newly extracted checkout with an existing `pso-cosmology` Conda environment, refresh the editable install and development requirements from the current directory:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

Use `python -m pytest -q` rather than a bare `pytest` command. This guarantees that the tests run with the same Python interpreter as the active environment.

## Checks

`python -m pytest -q` runs the automated model and PSO-mechanics tests.

`python scripts/check_reference_results.py` reruns the production optimizations and compares the recovered parameter vectors and minimum chi-square values with the reference results recorded in `RESULTS.md`.

`python scripts/run_notebooks.py --quick` uses reduced swarm sizes and iteration counts to check notebook execution and plotting without attempting to reproduce the production optima exactly.

`python scripts/run_notebooks.py` executes all notebooks with the production settings. Generated files are written under `results/`, which is ignored by Git.

## Optimization-diagnostic notebook

`notebooks/00_pso_mechanics.ipynb` uses the same flat-$\Lambda$CDM production settings as the main fit and regenerates the `pbest`/`gbest`, iteration-snapshot, global-best-path, convergence, and parameter-history diagnostics. It also checks the five-particle hand example used in the project presentation against the same 31-point objective function.
