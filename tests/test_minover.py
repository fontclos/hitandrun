"""Basic tests for minover."""

import unittest
import numpy as np
from hitandrun.polytope import Polytope
from hitandrun.minover import MinOver
import time


class TestMinOver(unittest.TestCase):
    """Basic tests for hit and run."""

    def test_minover_instantiate(self):
        """Test if MinOver object can be created."""
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        polytope = Polytope(A=A, b=b)
        minover = MinOver(polytope=polytope)
        self.assertTrue(isinstance(minover, MinOver))

    def test_minover_convergence(self):
        """Test if MinOver converges on a simple body."""
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        polytope = Polytope(A=A, b=b)
        minover = MinOver(polytope=polytope)
        x0 = np.array([-2, -2], dtype=np.float64)
        point, convergence = minover.run(starting_point=x0)
        self.assertTrue(convergence)

    def test_minover_inside_high_speed(self):
        """Test if MinOver converges on a simple body."""
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        polytope = Polytope(A=A, b=b)
        minover = MinOver(polytope=polytope)
        x0 = np.array([-2, -2], dtype=np.float64)
        point, convergence = minover.run(starting_point=x0, max_iters=1000,
                                         speed=0.1)
        self.assertTrue(polytope.check_inside(point))

    def test_minover_timing(self):
        """Test if we can get samples."""
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        x0 = np.array([1000, 1000], dtype=np.float64)
        polytope = Polytope(A=A, b=b)
        minover = MinOver(polytope=polytope)
        start = time.time()
        point, convergence = minover.run(starting_point=x0, speed=.1, max_iters=10000)
        end = time.time()
        print("Running time = %s" % (end - start))
        assert True
