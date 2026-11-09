"""L3 acceptance: range-Doppler geolocation vs the product's geolocation grid.

Run:
    python tests/test_l3_geo.py

For every grid point, feed its (lon, lat, h) into your solver and compare the
recovered radar coordinates with the ones the product reports.

Reference implementation (bisection, 60 iterations) measured on the 210 points:
    line   max 0.191 px    (that is the GRID's own time quantization: its
                            azimuthTime vs t0 + line*dt deviates up to 0.28 ms)
    pixel  max 0.0153 px   (the L2 SRGR residual, not your solver)
    range  max 0.0002 m
    |Dopp| max 12 m^2/s    (orbit interpolation noise, ~2e-7 s)
Tolerances: line < 0.3 px, pixel < 0.05 px, range < 0.01 m, |Dopp| < 1000 m^2/s.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sargeo import geo, io, orbit
from sargeo.ellipsoid import lla2ecef

C0 = 299_792_458.0
TAU, LINE, PIX, LAT, LON, H = 0, 1, 2, 3, 4, 5

_CACHE = None


def _errors():
    """One solve per grid point; cached so the tests don't redo 210 bisections each."""
    global _CACHE
    if _CACHE is None:
        G, _ = io.grid()
        e_line, e_pix, e_r, dop = [], [], [], []
        for n in range(len(G)):
            g = geo.geolocate(G[n, LON], G[n, LAT], G[n, H])
            e_line.append(g["line"] - G[n, LINE])
            e_pix.append(g["pixel"] - G[n, PIX])
            e_r.append(g["slant_range"] - C0 * G[n, TAU] / 2.0)
            P = lla2ecef(G[n, LAT], G[n, LON], G[n, H])
            dop.append(float(np.dot(orbit.position(g["t_az"]) - P, orbit.velocity(g["t_az"]))))
        _CACHE = (np.array(e_line), np.array(e_pix), np.array(e_r), np.array(dop))
    return _CACHE


def test_grid_lines():
    e_line, _, _, _ = _errors()
    mx = np.abs(e_line).max()
    assert mx < 0.3, f"line error max {mx:.4f} px (reference impl: 0.191 px)"


def test_grid_pixels():
    _, e_pix, _, _ = _errors()
    mx = np.abs(e_pix).max()
    assert mx < 0.05, f"pixel error max {mx:.5f} px (reference impl: 0.0153 px)"


def test_grid_slant_range():
    _, _, e_r, _ = _errors()
    mx = np.abs(e_r).max()
    assert mx < 0.01, f"slant range error max {mx:.5f} m (reference impl: 0.0002 m)"


def test_doppler_is_zero_at_solution():
    _, _, _, dop = _errors()
    mx = np.abs(dop).max()
    assert mx < 1000.0, f"(S-P).V at solution = {mx:.1f} m^2/s (should be ~12)"


def test_api_contract():
    G, _ = io.grid()
    g = geo.geolocate(float(G[0, LON]), float(G[0, LAT]), float(G[0, H]))
    for k in ("line", "pixel", "slant_range", "t_az"):
        assert k in g, f"missing key {k!r}"
        assert isinstance(g[k], float), f"{k} should be float, got {type(g[k]).__name__}"
    assert abs(g["line"]) < 0.3, f"first grid point is at line {G[0, LINE]}, got {g['line']:.3f}"


if __name__ == "__main__":
    tests = [test_api_contract, test_grid_lines, test_grid_pixels,
             test_grid_slant_range, test_doppler_is_zero_at_solution]
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