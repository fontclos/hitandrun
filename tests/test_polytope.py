"""Basic tests for polytopes."""

import unittest
import numpy as np
from hitandrun.polytope import Polytope


class TestPolytope(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1, 0],
                           [-1, 0],
                           [0, 1],
                           [0, -1]], dtype=np.float64)
        self.b = np.array([1, -1, 1, -1], dtype=np.float64)

    def test_polytope_instantiation(self):
        polytope = Polytope(A=self.A, b=self.b)

        self.assertIsInstance(polytope, Polytope)

    def test_polytope_dimension(self):
        polytope = Polytope(A=self.A, b=self.b)

        self.assertEqual(polytope.dim, 2)

    def test_polytope_nplanes(self):
        polytope = Polytope(A=self.A, b=self.b)

        self.assertEqual(polytope.nplanes, 4)

    def test_polytope_auxiliar_points(self):
        polytope = Polytope(A=self.A, b=self.b)
        checks = (polytope.auxiliar_points @ polytope.A.T) - self.b
        self.assertTrue(np.allclose(np.diag(checks), 0))
