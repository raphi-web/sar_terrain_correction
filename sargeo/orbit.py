"""L1 assignment: the orbit.

From the 17 cached ECEF state vectors (sargeo.io.orbit_arrays), implement:

    position(t) -> (..., 3) m      velocity(t) -> (..., 3) m/s

t = scalar or ndarray of Unix epoch seconds; OSVs are 10 s apart, cover scene ±80 s.
Method is yours. Cubic Hermite on (pos, vel) is the classic (C1 continuous, exact at
nodes); any other method is fine as long as tests/test_l1_orbit.py stays green —
it demands the product's own geolocation grid be reproduced to < 0.1 m (reference
Hermite achieves 0.0002 m, so sloppy interpolation WILL be caught, hehe~).

Return contract (both functions):
    t scalar  -> np.ndarray of shape (3,)
    t (N,)    -> np.ndarray of shape (N, 3)
    dtype float64, C-contiguous; last axis = (x, y, z) ECEF.
"""

from __future__ import annotations

import numpy as np

from . import io


def position(t: float | np.ndarray) -> np.ndarray:
    """Interpolate satellite ECEF position from orbit state vectors.

    Computes S(t), the satellite position in Earth-Centered,
    Earth-Fixed (ECEF) coordinates using cubic Hermite interpolation
    of orbit state vectors (position and velocity).

    Args:
        t:
            Query time in seconds. The time reference/epoch must match
            the orbit state vector timestamps provided by the orbit data.

            Accepts:
                - float: single time value
                - ndarray: array of query times with shape (N,)

    Returns:
        Satellite ECEF position in meters.

        Shape:
            - (3,) for scalar input
            - (N, 3) for array input

        The last dimension contains:
            [x, y, z] ECEF coordinates.

    Notes:
        The interpolation uses the position and velocity at the
        surrounding orbit state vectors to provide a smooth trajectory.

        Units:
            - time: seconds
            - position: meters
            - velocity: meters/second
    """

    is_scalar = np.ndim(t) == 0
    t = np.atleast_1d(t)


    t_orb, p_orb, v_orb = io.orbit_arrays() # load orbit data
    i = np.searchsorted(t_orb, t) - 1  # 	which 10-s segment each query time falls in
    i = np.clip(i, 0, len(t_orb) - 2) # protect the boundaries of the t_orb array
    h = t_orb[i + 1] - t_orb[i]  # calculate segment length

    # get the endpoint states
    v_i = v_orb[i]  # velocity, endpoints of state vectors
    v_next = v_orb[i + 1]
    p_i = p_orb[i] # position, endpoints of state vectors
    p_next = p_orb[i + 1]

    m_i = h[:, None] * v_i # convert velocities into tangent vectory (m / s) to (m)
    m_next = h[:, None] * v_next

    s = (t - t_orb[i]) / h #  normalized positions inside orbit segment
    s_ = s[:, None] # compute the hermite basis function (N,) into (N, 1)

    # now P(t) becomes P(s) where s=0 is first orbit point and s=1 is second orbit point
    # this guarantees that the interpolated orbit passes exactly
    # through the known satellite positions
    h00 = 2 * s_**3 - 3 * s_**2 + 1
    h10 = s_**3 - 2 * s_**2 + s_
    h01 = -2 * s_**3 + 3 * s_**2
    h11 = s_**3 - s_**2

    s_t = h00 * p_i + h10 * m_i + h01 * p_next + h11 * m_next

    if is_scalar:
        return s_t[0]

    return s_t

def velocity(t: float | np.ndarray) -> np.ndarray:
    """V(t): ECEF satellite velocity [m/s], same shape contract as position().
       Args:
        t:
            Query time in seconds. The time reference/epoch must match
            the orbit state vector timestamps provided by the orbit data.

            Accepts:
                - float: single time value
                - ndarray: array of query times with shape (N,)
    """

    is_scalar = np.ndim(t) == 0
    t = np.atleast_1d(t)
    t_orb, p_orb, v_orb = io.orbit_arrays()


    i = np.searchsorted(t_orb, t) - 1  # find correct orbit segment
    i = np.clip(i, 0, len(t_orb) - 2) # clamp index
    h = t_orb[i + 1] - t_orb[i]  # 	segment length

    v_i = v_orb[i]  # velocity,
    v_next = v_orb[i + 1] # velocity end point

    p_i = p_orb[i] # position
    p_next = p_orb[i + 1] # position end point

    # convert velocity to hermite tangent
    m_i = h[:, None] * v_i
    m_next = h[:, None] * v_next

    s = (t - t_orb[i]) / h
    s_ = s[:, None]

    # we need the derivative of the hermite basis functions for the velocity
    dh00 = 6 * s_**2 - 6 * s_
    dh10 = 3 * s_**2 - 4 * s_ + 1
    dh01 = -6 * s_**2 + 6 * s_
    dh11 = 3 * s_**2 - 2 * s_

    # compute deriviative in normalized coordinated
    v_t = dh00 * p_i + dh10 * m_i + dh01 * p_next + dh11 * m_next # meters
    v_t /= h[:, None] # convert back to  m / s
    return v_t[0] if is_scalar else v_t
