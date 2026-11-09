"""L5 acceptance: local incidence + layover/shadow masks.

Run:
    python tests/test_l5_terrain.py

Reference measurements:
    flat-normals psi vs the product's incidence grid : max 0.037 deg (mean 0.033)
    synthetic wall (1000 m, 26 m pixels, elev 30)    : shadow behind it, none in front
    real AOI (window 3500,1800, 200x200)             : layover 1.82%, shadow 0.0%
        (shadow needs horizon angles > 56 deg at 34 deg incidence — this scene's
         26 m DEM peaks at ~52 deg, so genuine zero.  Logic is covered synthetically.)

Tolerances: psi vs grid < 0.1 deg; normals flat/tilted < 1e-3; synthetic masks exact;
real layover fraction in (0.001, 0.05).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sargeo import io, orbit, terrain
from sargeo.ellipsoid import lla2ecef

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRD = os.path.join(
    ROOT,
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
    "measurement",
    "s1d-iw-grd-vv-20261031t165004-20261031t165029-005262-00a00b-001-cog.tiff",
)
DEM = os.path.join(ROOT, "dem.tif")


def test_normals_flat_dem():
    dem = np.full((20, 20), 500.0)
    lon = np.full((20, 20), 15.4)
    lat = np.full((20, 20), 47.05)
    n = terrain.surface_normals(dem, 26.0, 26.0, lon, lat)
    up = terrain.geodetic_up(lat, lon)
    err = np.abs(n - up).max()
    assert err < 1e-3, f"flat DEM normal deviates from up by {err:.2e}"


def test_normals_tilted_dem():
    # plane rising to the east at 30 %: dh/dE = 0.3  ->  normal tilts west
    dem = 0.3 * 26.0 * np.arange(40)[None, :].repeat(10, axis=0)
    lon = np.full(dem.shape, 15.4)
    lat = np.full(dem.shape, 47.05)
    n = terrain.surface_normals(dem, 26.0, 26.0, lon, lat)
    up = terrain.geodetic_up(lat, lon)
    east = np.stack([-np.sin(np.radians(lon)), np.cos(np.radians(lon)),
                     np.zeros_like(lon)], axis=-1)
    beta = np.arctan(0.3)
    expect = np.cos(beta) * up - np.sin(beta) * east
    err = np.abs(n - expect).max()
    assert err < 1e-3, f"tilted DEM normal off by {err:.2e}"


def test_normals_north_slope():
    # plane rising to the NORTH at 30 %: rows grow south, so H decreases with row
    hh, ww = 20, 20
    dem = (0.3 * 26.03 * (hh - 1 - np.arange(hh)))[:, None].repeat(ww, axis=1)
    lon = np.full(dem.shape, 15.4)
    lat = np.full(dem.shape, 47.05)
    n = terrain.surface_normals(dem, 26.03, 26.03, lon, lat)
    up = terrain.geodetic_up(lat, lon)
    la, lo = np.radians(lat), np.radians(lon)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo),
                      np.cos(la)], axis=-1)
    beta = np.arctan(0.3)
    expect = np.cos(beta) * up - np.sin(beta) * north      # normal leans SOUTH
    err = np.abs(n - expect).max()
    assert err < 1e-3, f"north-facing slope sign/normal error {err:.2e}"


def test_psi_matches_grid_incidence():
    G, tg = io.grid()
    P = lla2ecef(G[:, io.LAT], G[:, io.LON], G[:, io.H])
    up = terrain.geodetic_up(G[:, io.LAT], G[:, io.LON])
    S = orbit.position(tg)
    psi = terrain.projected_incidence(up, P, S, up)      # flat normals
    err = np.abs(psi - G[:, io.INC]).max()
    assert err < 0.1, f"psi vs product incidence: max {err:.4f} deg"


def test_horizon_masks_synthetic():
    hh, ww = 60, 120
    dem = np.zeros((hh, ww))
    dem[:, 60] = 1000.0                                   # 1-px wall
    r = (1000.0 - 0.5 * np.arange(ww)[None, :]).repeat(hh, axis=0).astype(float)
    elev = np.full((hh, ww), 30.0)

    lay, sha = terrain.horizon_masks(r, dem, 1.0, 0.0, 26.03, 26.03, 26.03, 100, elev)
    # the outermost row cannot be sampled bilinearly (it needs row+1) -> no info,
    # so masks are False there; test the interior.
    assert sha[1:-1, 20].all(), "shadow missing behind the wall"
    assert not sha[1:-1, 100].any(), "shadow in front of the wall"
    assert not lay[1:-1].any(), "layover on a monotone range field"

    r2 = r.copy()
    r2[:, 63] = 980.0     # a nearer point with a larger range than col 62 (969 m)
    lay2, _ = terrain.horizon_masks(r2, dem, 1.0, 0.0, 26.03, 26.03, 26.03, 100, elev)
    assert lay2[1:-1, 62].all(), "layover not detected behind the range spike"
    assert not lay2[1:-1, 10].any(), "false layover far from the spike"


def test_mask_aoi_real():
    m = terrain.mask_aoi(DEM, GRD, 3500, 1800, 200, 200)
    assert m["layover"].shape == (200, 200) and m["shadow"].shape == (200, 200)
    assert np.isfinite(m["psi"]).all(), "non-finite psi in the AOI"
    frac = m["layover_frac"]
    assert 0.001 < frac < 0.05, f"layover fraction {frac*100:.2f}% outside the plausible range"


if __name__ == "__main__":
    tests = [test_normals_flat_dem, test_normals_tilted_dem, test_normals_north_slope,
             test_psi_matches_grid_incidence,
             test_horizon_masks_synthetic, test_mask_aoi_real]
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