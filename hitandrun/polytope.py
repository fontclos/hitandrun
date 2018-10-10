"""
A class to hold polytopes in H-representation.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
from numba import jitclass
from numba import int64, float32
spec = [
    ("A", float32[:, :]),
    ("b", float32[:]),
    ("dim", int64),
    ("nplanes", int64),
    ("auxiliar_points", float32[:, :])
]


@jitclass(spec)
class Polytope(object):
    """A polytope in H-representation."""

    def __init__(self, A=None, b=None):
        """
        Create a polytope in H-representation.

        The polytope is defined as the set of
        points x in Rn such that

        A x <= b

        """
        # dimensionality verifications
        assert A is not None and b is not None
        assert len(b.shape) == 1
        assert len(A.shape) == 2
        assert A.shape[0] == len(b)
        # store data
        self.A = A
        self.b = b
        self.dim = A.shape[1]
        self.nplanes = A.shape[0]
        self.auxiliar_points = np.zeros(shape=A.shape, dtype=np.float32)
        self._find_auxiliar_points_in_planes()

    def check_inside(self, point):
        """Check if a point is inside the polytope."""
        checks = self.A@point <= self.b
        check = np.all(checks)
        return check

    def _find_auxiliar_points_in_planes(self):
        """Find an auxiliar point for each plane."""
        auxiliar_points = np.zeros((self.nplanes, self.dim), dtype=np.float32)
        for i in range(self.nplanes):
            auxiliar_points[i] = self._find_auxiliar_point(self.A[i], self.b[i])
        self.auxiliar_points = auxiliar_points

    def _find_auxiliar_point(self, Ai, bi):
        """Find an auxiliar point for one plane."""
        p = np.zeros(self.dim, dtype=np.float32)
        j = np.argmax(Ai != 0)
        p[j] = bi / Ai[j]
        return p
