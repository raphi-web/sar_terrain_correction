from __future__ import annotations

import numpy as np

from . import io, orbit, srgr
from .ellipsoid import lla2ecef


def zero_doppler_time(P: np.ndarray) -> float:
    """Acquisition time t* of ground point P (ECEF m): root of Dopp(t)
    solves: given point on earth (lon, lat, h) where does the point appear in the sar image
    """

    t_orb, _, _ = io.orbit_arrays() # loads time position and velocity
    lo, hi = float(t_orb[0]), float(t_orb[-1]) # search intervall

    # doppler equation D(t) = (S(t) - P) * V(t)
    # Doppler = (satellite position - ground point) * satellite velocity
    def f(tt):
        return float(np.dot(orbit.position(tt) - P, orbit.velocity(tt)))

    # D(t) = 0.5 * d/dt |S(t) - P|^2, so its sign is the sign of the range rate:
    # if D(t) < 0 -> range shrinking, satellite still approaching P
    # if D(t) > 0 -> range growing, satellite has passed P

    # we are looking for D(t) = (S(t)-P) * V(t) = 0
    # only t is the unknown
    # use bisection to solve, root must be between high and low
    f_lo = f(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (f_lo > 0):   # mid has the same sign as lo
            lo, f_lo = mid, fm        # -> root is in [mid, hi]: move lo up
        else:
            hi = mid                  # -> root is in [lo, mid]: pull hi down
    return 0.5 * (lo + hi)


def geolocate(lon: float, lat: float, h: float) -> dict:
    """
    given a point (lon, lat, height) find corresponding sar image coordinates
    """

    P = lla2ecef(lat, lon, h) # convert geographic coordinats to earth centered earth fixed
    t = zero_doppler_time(P) # find when satellite observed the point

    # compute the slant range
    r = float(
        np.linalg.norm(
            orbit.position(t) - P
        )
    )
    # convert slant range to ground range
    gr = float(
        srgr.slant_to_ground(r, t)
    )

    tim = io.timing() # load timing metadata
    # convert the acquisition time to image line (row number)
    line = (t - tim["t0"]) / tim["dt_az"]

    # convert the ground range to pixels (the column)
    # 10m spacing for sentinel-1
    return {
        'line': line, 'pixel': gr / 10.0, 'slant_range': r, 't_az': t
    }