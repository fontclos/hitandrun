"""Basic tests for polytopes."""

import unittest
import numpy as np
from hitandrun.polytope import Polytope


class TestPolytope(unittest.TestCase):
    def test_polytope_instantiation(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, -1, 1, -1], dtype=np.float64)
        polytope = Polytope(A=A, b=b)

        self.assertTrue(isinstance(polytope, Polytope))

    def test_polytope_dimension(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, -1, 1, -1], dtype=np.float64)
        polytope = Polytope(A=A, b=b)

        self.assertEqual(polytope.dim, 2)

    def test_polytope_nplanes(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, -1, 1, -1], dtype=np.float64)
        polytope = Polytope(A=A, b=b)

        self.assertEqual(polytope.nplanes, 4)

    def test_polytope_auxiliar_points(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]], dtype=np.float64)
        b = np.array([1, -1, 1, -1], dtype=np.float64)
        polytope = Polytope(A=A, b=b)
        checks = (polytope.auxiliar_points @ polytope.A.T) - b
        self.assertTrue(np.allclose(np.diag(checks), 0))
