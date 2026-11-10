"""L4 acceptance: ortho driver on a real AOI.

Run:
    python tests/test_l4_ortho.py

Reference implementation (vectorized numpy, 60 iterations) measured:
    geocode_tile vs scalar L3 : line 0.1911 px, pixel 0.01533 px  (same numbers)
    1000x1000 AOI             : 16 s
    200x200 AOI               : ~1 s
    output vs source window   : median DN 176 vs 170 (bilinear + geometry)

Tolerances: tile-vs-scalar < 1e-3 px; linear-ramp sampling < 1e-3;
AOI valid fraction > 0.9; ortho median DN within 15% of the source window median.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rasterio
from rasterio.windows import Window

from sargeo import geo, io, ortho

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM = os.path.join(ROOT, "dem_ellipsoidal.tif")
GRD = os.path.join(
    ROOT,
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
    "measurement",
    "s1d-iw-grd-vv-20261031t165004-20261031t165029-005262-00a00b-001-cog.tiff",
)

_AOI = None


def _aoi():
    global _AOI
    if _AOI is None:
        out = os.path.join(ROOT, "figures", "_l4_test_aoi.tif")
        stats = ortho.run_aoi(DEM, GRD, out, 4500, 2500, 200, 200)
        _AOI = (out, stats)
    return _AOI


def test_geocode_tile_matches_scalar_solver():
    G, _ = io.grid()
    sel = np.arange(0, len(G), 3)
    line, pix = ortho.geocode_tile(G[sel, io.LON], G[sel, io.LAT], G[sel, io.H])
    worst_l = worst_p = 0.0
    for n, k in enumerate(sel):
        g = geo.geolocate(float(G[k, io.LON]), float(G[k, io.LAT]), float(G[k, io.H]))
        worst_l = max(worst_l, abs(float(line[n]) - g["line"]))
        worst_p = max(worst_p, abs(float(pix[n]) - g["pixel"]))
    assert worst_l < 1e-3, f"line mismatch vs scalar solver: {worst_l:.6f} px"
    assert worst_p < 1e-3, f"pixel mismatch vs scalar solver: {worst_p:.6f} px"


def test_bilinear_sampling_on_linear_ramp():
    dn = np.tile(np.arange(500.0), (400, 1))          # value == column index
    rng = np.random.default_rng(0)
    line = rng.uniform(0.0, 398.0, 200)
    pix = rng.uniform(0.0, 498.0, 200)
    val = ortho.sample_bilinear(dn, line, pix)
    err = np.abs(val - pix)
    assert err.max() < 1e-3, f"bilinear ramp error {err.max():.6f} (should be exact)"
    nod = ortho.sample_bilinear(dn, np.array([-1.0, 500.0]), np.array([10.0, 10.0]), nodata=-9)
    assert np.all(nod == -9), f"out-of-bounds not masked: {nod}"


def test_aoi_geotiff_is_written():
    out, stats = _aoi()
    assert os.path.exists(out)
    with rasterio.open(out) as ds:
        assert ds.width == 200 and ds.height == 200
        assert ds.crs.to_epsg() == 32633
        assert ds.dtypes[0] == "uint16"
        arr = ds.read(1)
    assert stats["valid_frac"] > 0.9, f"only {stats['valid_frac']*100:.1f}% valid"
    valid = arr[arr > 0]
    assert valid.size > 0


def test_aoi_preserves_radiometry():
    out, stats = _aoi()
    y0, y1, x0, x1 = stats["src_window"]
    with rasterio.open(GRD) as ds:
        src = ds.read(1, window=Window(x0, y0, x1 - x0, y1 - y0))
    s = src[src > 0].astype(float)
    with rasterio.open(out) as ds:
        o = ds.read(1)
    o = o[o > 0].astype(float)
    med_src, med_out = np.median(s), np.median(o)
    rel = abs(med_out - med_src) / med_src
    assert rel < 0.15, f"median DN drift: source {med_src:.0f} vs ortho {med_out:.0f} ({rel*100:.1f}%)"


if __name__ == "__main__":
    tests = [test_geocode_tile_matches_scalar_solver, test_bilinear_sampling_on_linear_ramp,
             test_aoi_geotiff_is_written, test_aoi_preserves_radiometry]
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