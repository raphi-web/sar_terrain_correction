"""L9 acceptance: the SAFE parser and the CLI reproduce the shipped results.

Run:
    python tests/test_l9_pipeline.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sargeo import io, product, safeparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE = os.path.join(
    ROOT,
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
)
DEM = os.path.join(ROOT, "dem_ellipsoidal.tif")


def test_find_product_both_polarisations():
    for pol in ("vv", "vh"):
        grd, ann, cal = safeparse.find_product(SAFE, pol)
        for p in (grd, ann, cal):
            assert p and os.path.isfile(p), f"{pol}: missing {p}"
            assert f"-{pol}-" in os.path.basename(p)
        assert cal and os.path.basename(cal).startswith("calibration-"), \
            f"{pol}: picked {os.path.basename(cal)} instead of the calibration XML"


def test_prepare_meta_reproduces_shipped_cache():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "meta")
        safeparse.prepare_meta(SAFE, out, verbose=False)

        old_grid, new_grid = np.load(os.path.join(ROOT, "meta", "geolocation.npy")), np.load(os.path.join(out, "geolocation.npy"))
        assert np.abs(old_grid - new_grid).max() == 0.0, "geolocation grid differs"

        old = json.load(open(os.path.join(ROOT, "meta", "s1_meta.json")))
        new = json.load(open(os.path.join(out, "s1_meta.json")))
        assert len(old["orbit"]) == len(new["orbit"])
        a = np.array([c["srgr"] for c in old["coordinateConversion"]])
        b = np.array([c["srgr"] for c in new["coordinateConversion"]])
        assert np.abs(a - b).max() == 0.0, "SRGR coefficients differ"

        t = json.load(open(os.path.join(out, "timing.json")))
        assert t["nlines"] == 16677 and t["nsamp"] == 26540
        assert abs(t["dt_az"] - 1.499065906973919e-03) < 1e-18
        # free physics check: |S(t)-P| == c*tau/2
        assert abs(t["resid_stats"]["max"]) < 1e-3, "range residual above 1 mm"


def test_scene_extent_matches_reference():
    e = product.scene_extent(DEM)
    ref = json.load(open(os.path.join(ROOT, "meta", "ortho_extent.json")))
    assert e["w"] == ref["nx"] and e["h"] == ref["ny"], f'{e["w"]}x{e["h"]} vs {ref["nx"]}x{ref["ny"]}'
    assert e["col0"] == 112 and e["row0"] == 30


def test_cli_product_matches_pipeline():
    import rasterio
    from sargeo import radiometry

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "aoi.tif")
        res = product.write_product(SAFE, DEM, out, pol="vv", aoi=(3500, 1800, 200, 200),
                                    chunk_rows=200, progress=False,
                                    meta_dir=os.path.join(ROOT, "meta"))
        assert res["shape"] == [200, 200] and res["valid_frac"] > 0.9
        with rasterio.open(out) as ds:
            assert ds.count == 3 and ds.crs.to_epsg() == 32633
            assert str(ds.dtypes[0]) == "float32"
            s0 = ds.read(2)
        assert -20 < np.nanmedian(s0) < -5, "sigma0 median outside the plausible range"

        # same AOI through the older single-purpose driver
        ref = radiometry.run_aoi(DEM, safeparse.find_product(SAFE, "vv")[0],
                                 safeparse.find_product(SAFE, "vv")[2], 3500, 1800, 200, 200)
        d = np.nanmedian(s0) - np.nanmedian(ref["sigma0_db"])
        assert abs(d) < 0.01, f"CLI vs pipeline differ by {d:.3f} dB"


def test_meta_cache_builds_reuses_and_repairs():
    import time

    from sargeo import io

    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, "cache")
        d1 = safeparse.ensure_meta(SAFE, "vv", meta_dir=cache, verbose=False)
        assert os.path.isfile(os.path.join(d1, "source.json")), "no stamp written"
        assert os.path.isfile(os.path.join(d1, "timing.json"))

        t1 = os.path.getmtime(os.path.join(d1, "s1_meta.json"))
        time.sleep(0.02)
        d2 = safeparse.ensure_meta(SAFE, "vv", meta_dir=cache, verbose=False)
        assert d1 == d2 and os.path.getmtime(os.path.join(d2, "s1_meta.json")) == t1, "cache not reused"

        os.remove(os.path.join(d2, "timing.json"))       # damage the cache
        d3 = safeparse.ensure_meta(SAFE, "vv", meta_dir=cache, verbose=False)
        assert os.path.isfile(os.path.join(d3, "timing.json")), "cache not repaired"

        io.use_meta(d3)
        assert os.path.abspath(io.meta_dir()) == os.path.abspath(d3)
        assert io.timing()["nlines"] == 16677

    io.use_meta(os.path.join(ROOT, "meta"))              # leave the tests as they were


if __name__ == "__main__":
    tests = [test_find_product_both_polarisations, test_prepare_meta_reproduces_shipped_cache,
             test_scene_extent_matches_reference, test_cli_product_matches_pipeline,
             test_meta_cache_builds_reuses_and_repairs]
    fails = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            fails += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            fails += 1
    print(f"\n{len(tests)-fails}/{len(tests)} green" if fails else "\nAll green~")
    sys.exit(1 if fails else 0)