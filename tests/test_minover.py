"""Basic tests for minover."""

import unittest
import numpy as np
from hitandrun.polytope import Polytope
from hitandrun.minover import MinOver


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
                                         speed=1000)
        self.assertTrue(polytope.check_inside(point))
