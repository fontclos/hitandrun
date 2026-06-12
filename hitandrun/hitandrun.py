"""
Hit-and-run sampler.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
from numpy.linalg import norm


class HitAndRun:
    """Hit-and-run sampler."""

    def __init__(self, polytope=None, starting_point=None,
                 n_samples=100, thin=1, rng=None):
        """
        Create a hit-and-run sampler.

        Parameters
        ----------
        polytope: hitandrun.polytope
            The convex polytope to be sampled.
        starting_point: np.array
            Initial condition. Must be inside the polytope.
        n_samples: int
            Number of desired samples.
        thin : int
            Thinning factor, increase to get independent samples.
        rng : numpy.random.Generator, optional
            Random number generator to use. Defaults to numpy's global
            random module to preserve existing seeding behavior.

        """
        # make sure we got a point inside the polytope
        if polytope is None:
            raise ValueError("polytope must be provided.")
        if starting_point is None:
            raise ValueError("starting_point must be provided.")

        starting_point = np.asarray(starting_point)

        if len(starting_point) != polytope.dim:
            raise ValueError("starting_point must match the polytope dimension.")
        if not polytope.check_inside(starting_point):
            raise ValueError("starting_point must be inside the polytope.")

        self.polytope = polytope
        self.starting_point = starting_point
        self.n_samples = n_samples
        self.thin = thin
        self.rng = rng
        # place starting point as current point
        self.current = starting_point.copy()
        # set a starting random direction
        self._set_random_direction()
        # create empty list of samples
        self.samples = []

    def get_samples(self, n_samples=None, thin=None):
        """Get the requested samples."""
        self.samples = []
        if n_samples is not None:
            self.n_samples = n_samples
        if thin is not None:
            self.thin = thin

        # keep only one every thin
        for _ in _with_progress(
            range(self.n_samples),
            desc="hit-and-run steps:"
        ):
            for _ in range(self.thin):
                self._step()
            self._add_current_to_samples()
        return np.array(self.samples)

    # private functions
    def _step(self):
        """Make one step."""
        # set random direction
        self._set_random_direction()
        # find lambdas
        self._find_lambdas()
        # find smallest positive and negative lambdas
        positive_lambdas = self.lambdas[self.lambdas > 0]
        negative_lambdas = self.lambdas[self.lambdas < 0]
        if positive_lambdas.size == 0 or negative_lambdas.size == 0:
            raise RuntimeError("The current direction does not intersect "
                               "any of the hyperplanes.")
        lam_plus = np.min(positive_lambdas)
        lam_minus = np.max(negative_lambdas)
        # throw random point between lambdas
        lam = self._uniform(low=lam_minus, high=lam_plus)
        # compute new point and add it
        new_point = self.current + lam * self.direction
        self.current = new_point

    def _find_lambdas(self):
        """
        Find the lambda value for each hyperplane.

        The lambda value is the distance we have to travel
        in the current direction, from the current point, to
        reach a given hyperplane.
        """
        A = self.polytope.A
        numerators = self.polytope.b - A @ self.current
        denominators = A @ self.direction
        parallel = np.isclose(denominators, 0)
        lambdas = np.full_like(numerators, np.nan, dtype=np.float64)
        np.divide(numerators, denominators, out=lambdas, where=~parallel)
        self.lambdas = lambdas

    def _set_random_direction(self):
        """Set a unitary random direction in which to travel."""
        direction = self._normal(size=self.polytope.dim)
        self.direction = direction / norm(direction)

    def _uniform(self, low, high):
        """Draw from a uniform distribution using the configured RNG."""
        if self.rng is None:
            return np.random.uniform(low=low, high=high)
        return self.rng.uniform(low=low, high=high)

    def _normal(self, size):
        """Draw standard-normal values using the configured RNG."""
        if self.rng is None:
            return np.random.randn(size)
        return self.rng.normal(size=size)

    def _add_current_to_samples(self):
        self.samples.append(self.current.copy())


def _with_progress(iterable, **kwargs):
    """Wrap an iterable with tqdm when the optional dependency is installed."""
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, **kwargs)
