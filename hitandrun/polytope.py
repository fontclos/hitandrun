"""
A class to hold polytopes in H-representation.

Francesc Font-Clos
Oct 2018
"""
import numpy as np


class Polytope:
    """A polytope in H-representation."""

    def __init__(self, A=None, b=None):
        """
        Create a polytope in H-representation.

        The polytope is defined as the set of
        points x in Rn such that

        A x <= b

        """
        # dimensionality verifications
        if A is None or b is None:
            raise ValueError("A and b must be provided.")

        A = np.asarray(A)
        b = np.asarray(b)

        if b.ndim != 1:
            raise ValueError("b must be a one-dimensional array.")
        if A.ndim != 2:
            raise ValueError("A must be a two-dimensional array.")
        if A.shape[0] != b.shape[0]:
            raise ValueError("A must have one row for each entry in b.")

        # store data
        self.A = A
        self.b = b
        self.dim = A.shape[1]
        self.nplanes = A.shape[0]
        self._find_auxiliary_points_in_planes()

    def check_inside(self, point):
        """Check if a point is inside the polytope."""
        checks = self.A @ point <= self.b
        return bool(np.all(checks))

    def _find_auxiliary_points_in_planes(self):
        """Find an auxiliary point for each plane."""
        aux_points = [self._find_auxiliary_point(self.A[i],
                                                 self.b[i])
                      for i in range(self.nplanes)]
        self.auxiliary_points = aux_points
        self.auxiliar_points = self.auxiliary_points

    def _find_auxiliary_point(self, Ai, bi):
        """Find an auxiliary point for one plane."""
        p = np.zeros(self.dim)
        j = np.argmax(Ai != 0)
        p[j] = bi / Ai[j]
        return p
