"""L1 acceptance: orbit + ellipsoid vs the product's geolocation grid.

Run:
    python tests/test_l1_orbit.py
(or: same interpreter, -m pytest tests -x -q)

Truth statement: for all 210 grid points, |S(azimuthTime) - P_ecef(lat,lon,h)| == c*tau/2.
Reference (Hermite) residuals: max 0.0002 m, mean 0.000114 m.
Tolerances below: 0.1 m — 500x slack for method choice, zero slack for bugs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sargeo import io
from sargeo.ellipsoid import lla2ecef
from sargeo import orbit

C0 = 299_792_458.0


def test_position_reproduces_osv_epochs():
    t, pos, vel = io.orbit_arrays()
    got = orbit.position(t)
    assert got.shape == pos.shape, f"shape {got.shape} != {pos.shape}"
    err = np.linalg.norm(got - pos, axis=1)
    assert err.max() < 1e-3, f"epoch reproduction failed: max err {err.max():.3f} m"


def test_velocity_at_osv_epochs():
    t, pos, vel = io.orbit_arrays()
    got = orbit.velocity(t)
    assert got.shape == vel.shape
    err = np.linalg.norm(got - vel, axis=1)
    assert err.max() < 1.0, f"velocity at epochs off by {err.max():.3f} m/s"


def test_velocity_physical_sanity():
    t, pos, vel = io.orbit_arrays()
    tt = np.linspace(t[2], t[-3], 100)
    v = np.linalg.norm(orbit.velocity(tt), axis=1)
    assert np.all(v > 7.0e3) and np.all(v < 8.1e3), f"|v| {v.min():.1f}-{v.max():.1f} m/s, orbit is off"


def test_slant_range_against_geolocation_grid():
    G, tg = io.grid()
    S = orbit.position(tg)
    P = lla2ecef(G[:, io.LAT], G[:, io.LON], G[:, io.H])
    resid = np.linalg.norm(S - P, axis=1) - C0 * G[:, io.TAU] / 2.0
    mx = np.abs(resid).max()
    assert mx < 0.1, (f"grid residual max {mx:.4f} m (reference impl: 0.0002 m). "
                     f"mean {np.abs(resid).mean():.4f} m")


def test_scalar_contract():
    """position()/velocity() must accept ONE float and return (3,) — not (1,3), not a crash.

    This is not a formality: the L3 range-Doppler solver calls them per-pixel
    with scalar times, so an array-only implementation paints itself into a corner.
    """
    t, _, _ = io.orbit_arrays()
    t1 = float(t[3])
    p1 = orbit.position(t1)
    assert p1.shape == (3,), f"position(scalar).shape = {p1.shape}, contract says (3,)"
    assert np.allclose(p1, orbit.position(np.asarray([t1]))[0], rtol=0, atol=1e-9), \
        "scalar and length-1 array calls disagree"
    v1 = orbit.velocity(t1)
    assert v1.shape == (3,), f"velocity(scalar).shape = {v1.shape}, contract says (3,)"


if __name__ == "__main__":
    tests = [test_position_reproduces_osv_epochs, test_velocity_at_osv_epochs,
             test_velocity_physical_sanity, test_slant_range_against_geolocation_grid,
             test_scalar_contract]
    fails = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except NotImplementedError as e:
            print(f"RED   {fn.__name__}: {e}")
            fails += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            fails += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            fails += 1
    print(f"\n{len(tests)-fails}/{len(tests)} green" if fails else "\nAll green~ did you do that on purpose?")
    sys.exit(1 if fails else 0)
