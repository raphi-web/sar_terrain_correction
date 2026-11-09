

from __future__ import annotations

import numpy as np

A_E = 6_378_137.0 # earths equatorial radius the semi-major axis [m]
B_E = 6_356_752.314245  # polar radius the semi minor axis [m]
F = (A_E - B_E) / A_E # Flattening
E2 = 1.0 - (B_E / A_E) ** 2 # eccentricity squared


def lla2ecef(
    lat: float | np.ndarray, lon: float | np.ndarray, h: float | np.ndarray
) -> np.ndarray:
    """Geodetic -> Earth Centered Earth Fixed(WGS84).

    Args:
        lat: latitude, **degrees**, float or ndarray (N,)
        lon: longitude, **degrees**, same shape as lat
        h:   ellipsoid height, **meters**, same shape as lat

    Returns:
        ndarray float64, shape (N, 3) [or (3,) for scalar inputs].
        Columns are ECEF (x, y, z) in meters.

    """
    # convert to radians
    phi = np.radians(lat)
    lam = np.radians(lon)

    #   compute the radius of the curvature
    N = A_E / np.sqrt(1 - E2 * np.sin(phi) ** 2)

    # Earth-Centered Earth-Fixed coordinates
    x = (N + h) * np.cos(phi) * np.cos(lam)
    y = (N + h) * np.cos(phi) * np.sin(lam)
    z = (N * (1 - E2) + h) * np.sin(phi)

    return np.stack([x, y, z], axis=-1)
