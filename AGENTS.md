# AGENTS.md

## Cursor Cloud specific instructions

`hitandrun` is a pure-Python library (no servers, databases, or GUI) for uniformly
sampling convex polytopes with the Hit-and-Run algorithm. Public API lives in
`hitandrun/__init__.py` (`Polytope`, `HitAndRun`, `MinOver`).

Dependencies are installed into a virtualenv at `.venv` by the startup update
script. Activate it before running anything:

```bash
. .venv/bin/activate
```

- Test: `python -m unittest discover -s tests -v` (the documented runner; `pytest` also works).
- Lint: not configured in-repo; `flake8` is installed for convenience.
- "Run" the library: import it and sample a polytope, e.g.
  ```python
  import numpy as np
  from hitandrun import HitAndRun, Polytope
  A = np.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=np.float64)
  b = np.array([1,1,1,1], dtype=np.float64)
  s = HitAndRun(polytope=Polytope(A=A, b=b), starting_point=np.zeros(2))
  print(s.get_samples(n_samples=100))
  ```

Note: the optional `tqdm` progress bar is auto-enabled when the `progress` extra
is installed (it is, via the update script). Pass an explicit `rng=` to
`HitAndRun` for reproducible samples.
