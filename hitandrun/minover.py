"""
MinOver algorithm to find a point inside a polytope.

Francesc Font-Clos
Oct 2018
"""
import numpy as np
import numba


class MinOver(object):
    """MinOver solver."""

    def __init__(self, polytope, ):
        """
        Create a MinOver solver.

        Parameters
        ----------
        polytope: hitandrun.polytope
            Polytope in H-representation

        """
        self.polytope = polytope

    def run(self, speed=1, starting_point=None, max_iters=100, verbose=False):
        """
        Run the MinOver algorithm.

        Parameters
        ----------
        speed: float
            Distance moved at each learning step
        max_iters: int
            Maximum number of iterations (per hyperplan).
        starting_poin: np.array
            Initial condition.

        Returns
        -------
        current: np.array
            The final point.
        convergence: bool
            True if the algorithm converged, False, otherwise.

        """
        self.max_iters = max_iters * self.polytope.nplanes
        self.speed = speed
        if starting_point is None:
            self.current = np.zeros(self.polytope.dim)
        else:
            self.current = starting_point
        # compute step 0 worst planes
        # this is a trick to handle first steps
        self.worst_indexes = [-1, -2]
        self.worst_distances = [-1, -2]
        self._set_worst_constraint()
        for i in range(self.max_iters):
            convergence = self._step()
            # self._check_speed()
            if verbose:
                self.iter = i
                self._print_worst()
            if convergence:
                break
        return self.current, convergence

    def _step(self):
        new = _move_towards_worst_plane(current=self.current,
                                        speed=self.speed,
                                        plane=self.polytope.A[self.worst])
        self.current = new
        self._set_worst_constraint()
        return np.all(self.distances < 0)

    def _check_speed(self):
        i0, i1, i2 = self.worst_indexes[::-1][:3]
        d0, d1, d2 = self.worst_distances[::-1][:3]
        if i0 != i1 and i0 == i2 and d0 >= d2:
            self.speed *= 0.9

    def _set_worst_constraint(self):
        self.distances = self.polytope.A @ self.current - self.polytope.b
        self.worst = np.argmax(self.distances)
        self.worst_indexes.append(self.worst)
        self.worst_distances.append(self.distances[self.worst])

    def _print_worst(self):
        worst_distance = self.distances[self.worst]
        print("iter", self.iter,
              "index:", self.worst,
              "distance:", worst_distance,
              "speed:", self.speed)

@numba.njit
def _move_towards_worst_plane(current, speed, plane):
    # self.current = self.current - self.speed * self.polytope.A[self.worst]
    return current - speed * plane
