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

    def test_hitandrun_finds_random_starting_point(self):
        """Test if HitAndRun can find a feasible starting point."""
        rng = np.random.default_rng(3)
        hitandrun = HitAndRun(polytope=self.polytope, rng=rng)

        self.assertTrue(self.polytope.check_inside(hitandrun.starting_point))

        samples = hitandrun.get_samples(n_samples=100)
        checks = samples @ self.polytope.A.T - self.polytope.b
        self.assertTrue(np.all(checks < 0))

    def test_hitandrun_random_starting_point_uses_rng(self):
        """Test if automatic starting points are reproducible."""
        sampler0 = HitAndRun(polytope=self.polytope,
                             rng=np.random.default_rng(3))
        sampler1 = HitAndRun(polytope=self.polytope,
                             rng=np.random.default_rng(3))

        self.assertTrue(np.allclose(sampler0.starting_point,
                                    sampler1.starting_point))

    def test_hitandrun_random_starting_point_requires_feasible_polytope(self):
        """Test if infeasible polytopes fail automatic initialization."""
        A = np.array([[1],
                      [-1]], dtype=np.float64)
        b = np.array([0, -1], dtype=np.float64)
        polytope = Polytope(A=A, b=b)

        with self.assertRaisesRegex(ValueError, "feasible starting_point"):
            HitAndRun(polytope=polytope, rng=np.random.default_rng(3))

    def test_find_lambdas_handles_parallel_planes(self):
        """Test lambda computation for intersecting and parallel planes."""
        hitandrun = HitAndRun(polytope=self.polytope,
                              starting_point=np.zeros(2, dtype=np.float64))
        hitandrun.current = np.zeros(2, dtype=np.float64)
        hitandrun.direction = np.array([1, 0], dtype=np.float64)

        hitandrun._find_lambdas()

        expected = np.array([1, -1, np.nan, np.nan], dtype=np.float64)
        self.assertTrue(np.allclose(hitandrun.lambdas[:2], expected[:2]))
        self.assertTrue(np.all(np.isnan(hitandrun.lambdas[2:])))
