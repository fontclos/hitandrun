"""Basic tests for minover."""

import unittest
import numpy as np
from hitandrun.polytope import Polytope
from hitandrun.minover import MinOver


class TestMinOver(unittest.TestCase):
    """Basic tests for hit and run."""

    def setUp(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        self.polytope = Polytope(A=A, b=b)
        x0 = np.array([-2, -2], dtype=np.float64)
        self.starting_point = x0

    def test_minover_instantiate(self):
        """Test if MinOver object can be created."""
        minover = MinOver(polytope=self.polytope)
        self.assertIsInstance(minover, MinOver)

    def test_minover_convergence(self):
        """Test if MinOver converges on a simple body."""
        minover = MinOver(polytope=self.polytope)
        point, convergence = minover.run(starting_point=self.starting_point)
        self.assertTrue(convergence)
        self.assertTrue(self.polytope.check_inside(point))

    def test_minover_inside_high_speed(self):
        """Test if MinOver converges on a simple body."""
        minover = MinOver(polytope=self.polytope)
        point, convergence = minover.run(starting_point=self.starting_point,
                                         max_iters=1000, speed=1000)
        self.assertTrue(convergence)
        self.assertTrue(self.polytope.check_inside(point))
