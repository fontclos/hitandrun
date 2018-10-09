"""
A class to hold polytopes in H-representation.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
import numba


@numba.jitclass([
    ("A", numba.types.float64[:, :]),
    ("b", numba.types.float64[:]),
    ("auxiliar_points", numba.types.float64[:, :]),
    ("dim", numba.types.int8),
    ("nplanes", numba.types.int8)
    ])
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
        self._find_auxiliar_points_in_planes()

    def check_inside(self, point):
        """Check if a point is inside the polytope."""
        checks = self.A@point <= self.b
        check = np.all(checks)
        return check

    def _find_auxiliar_points_in_planes(self):
        """Find an auxiliar point for each plane."""
        # aux_points = [self._find_auxiliar_point(self.A[i],
        #                                         self.b[i])
        #               for i in range(self.nplanes)]
        aux_points = np.array(shape=(self.nplanes, self.dim))
        for i in range(self.nplanes):
            aux_points_l.append(self._find_auxiliar_point(self.A[i], self.b[i]))
        aux_points = np.array(aux_points_l)
        self.auxiliar_points = aux_points

    def _find_auxiliar_point(self, Ai, bi):
        """Find an auxiliar point for one plane."""
        p = np.zeros(self.dim)
        j = np.argmax(Ai != 0)
        p[j] = bi / Ai[j]
        return p
