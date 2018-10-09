"""Basic tests for polytopes."""

import unittest
import numpy as np
from hitandrun.polytope import Polytope


class TestPolytope(unittest.TestCase):
    def test_instantiation(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]])
        b = np.array([1, -1, 1, -1])
        polytope = Polytope(A=A, b=b)

        self.assertTrue(isinstance(polytope, Polytope))

    def test_dimension(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]])
        b = np.array([1, -1, 1, -1])
        polytope = Polytope(A=A, b=b)

        self.assertEqual(polytope.dim, 2)

    def test_nplanes(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]])
        b = np.array([1, -1, 1, -1])
        polytope = Polytope(A=A, b=b)

        self.assertEqual(polytope.nplanes, 4)

    def test_auxiliar_points(self):
        A = np.array([[1, 0],
                      [-1, 0],
                      [0, 1],
                      [0, -1]])
        b = np.array([1, -1, 1, -1])
        polytope = Polytope(A=A, b=b)
        checks = (polytope.auxiliar_points @ polytope.A.T) - b
        self.assertTrue(np.allclose(np.diag(checks), 0))
