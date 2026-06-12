# Hit and Run
[![DOI](https://zenodo.org/badge/152282588.svg)](https://zenodo.org/badge/latestdoi/152282588)  

Small Python package for uniformly sampling convex polytopes with the
Hit-and-Run algorithm. Polytopes are given in H-representation (`A x <= b`).

Includes `Polytope` for constraints and `HitAndRun` for generating samples.

## Installation

```bash
python -m pip install .
```

Runtime dependencies are declared in `pyproject.toml`. The core package depends
on NumPy; install the optional `progress` extra to show tqdm progress bars:

```bash
python -m pip install ".[progress]"
```

## Usage

```python
import numpy as np

from hitandrun import HitAndRun, Polytope

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
b = np.array([1, 1, 1, 1], dtype=np.float64)
x0 = np.array([0, 0], dtype=np.float64)

polytope = Polytope(A=A, b=b)
sampler = HitAndRun(polytope=polytope, starting_point=x0)
samples = sampler.get_samples(n_samples=100)
```

## Notebook

See `notebooks/hitandrun_usage.ipynb` for a plotted walkthrough of defining a
polytope, finding a feasible starting point with `MinOver`, sampling it with
`HitAndRun`, and inspecting empirical marginals. The notebook uses Matplotlib
for figures:

```bash
python -m pip install ".[progress]" matplotlib
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Contact
Francesc Font-Clos  
https://github.com/fontclos  
francesc.font@gmail.com
