# Hit and Run
[![DOI](https://zenodo.org/badge/152282588.svg)](https://zenodo.org/badge/latestdoi/152282588)  

Python implementation of the Hit-and-Run algorithm to uniformly sample convex
polytopes in H-representation (`A x <= b`).

## Installation

```bash
python -m pip install .
```

Runtime dependencies are declared in `pyproject.toml` and include NumPy, SciPy,
and tqdm.

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

## Tests

```bash
python -m unittest discover -s tests -v
```

## Contact
Francesc Font-Clos  
https://github.com/fontclos  
francesc.font@gmail.com
