"""
Hit-and-run sampler.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
from scipy.spatial.distance import norm

import tqdm


class HitAndRun(object):
    """Hit-and-run sampler."""

    def __init__(self, polytope=None, starting_point=None,
                 n_samples=100, thin=1):
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

        """
        # make sure we got a point inside the polytope
        assert starting_point is not None
        assert len(starting_point) == polytope.dim
        assert polytope.check_inside(starting_point)

        self.polytope = polytope
        self.starting_point = starting_point
        self.n_samples = n_samples
        self.thin = thin
        # place starting point as current point
        self.current = starting_point
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
        for i in tqdm.tqdm(
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
        try:
            lam_plus = np.min(self.lambdas[self.lambdas > 0])
            lam_minus = np.max(self.lambdas[self.lambdas < 0])
        except(Exception):
            raise RuntimeError("The current direction does not intersect"
                               "any of the hyperplanes.")
        # throw random point between lambdas
        lam = np.random.uniform(low=lam_minus, high=lam_plus)
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
        p = self.polytope.auxiliar_points

        lambdas = []
        for i in range(self.polytope.nplanes):
            if np.isclose(self.direction @ A[i], 0):
                lambdas.append(np.nan)
            else:
                lam = ((p[i] - self.current) @ A[i]) / (self.direction @ A[i])
                lambdas.append(lam)
        self.lambdas = np.array(lambdas)

    def _set_random_direction(self):
        """Set a unitary random direction in which to travel."""
        direction = np.random.randn(self.polytope.dim)
        self.direction = direction / norm(direction)

    def _add_current_to_samples(self):
        self.samples.append(list(self.current))
