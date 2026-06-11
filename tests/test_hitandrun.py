"""Basic tests for polytopes."""

import unittest
import numpy as np
from hitandrun import HitAndRun, Polytope


class TestHitAndRun(unittest.TestCase):
    """Basic tests for hit and run."""

    def setUp(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        self.starting_point = np.array([-.5, -.5], dtype=np.float64)
        self.polytope = Polytope(A=A, b=b)

    def test_hitandrun_instantiate(self):
        """Test if HitAndRun object can be created."""
        hitandrun = HitAndRun(polytope=self.polytope,
                              starting_point=self.starting_point,
                              thin=1,
                              n_samples=100
                              )
        self.assertIsInstance(hitandrun, HitAndRun)

    def test_hitandrun_sampling(self):
        """Test if we can get samples."""
        hitandrun = HitAndRun(polytope=self.polytope,
                              starting_point=self.starting_point)
        samples = hitandrun.get_samples(n_samples=100)
        checks = samples @ self.polytope.A.T - self.polytope.b
        self.assertTrue(np.all(checks < 0))
