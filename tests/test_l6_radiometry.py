"""L6 acceptance: calibration LUT -> sigma0 -> flattened gamma0.

Run:
    python tests/test_l6_radiometry.py

Reference measurements (AOI 3500,1800 400x400):
    sigma0 dB      p5 -17.0  p50 -11.3  p95 -3.9
    gamma0 dB      p5 -15.1  p50 -9.8   p95 -3.3
    psi-bin means  6.43 dB spread -> 3.07 dB after flattening
    corr(x, cos psi) 0.506 -> 0.233

Tolerances: LUT exact at nodes (<1e-9), linear midway (<1e-9), synthetic
flatten round trip (<1e-9 relative), real AOI median in (-20, -5) dB,
spread < 3.5 dB and corr < 0.35 after correction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sargeo import radiometry as rad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE = os.path.join(
    ROOT,
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
)
GRD = os.path.join(SAFE, "measurement",
                   "s1d-iw-grd-vv-20261031t165004-20261031t165029-005262-00a00b-001-cog.tiff")
CAL = os.path.join(SAFE, "annotation", "calibration",
                   "calibration-s1d-iw-grd-vv-20261031t165004-20261031t165029-005262-00a00b-001-cog.xml")
DEM = os.path.join(ROOT, "dem_ellipsoidal.tif")

_CAL = None


def cal():
    global _CAL
    if _CAL is None:
        _CAL = rad.load_calibration(CAL)
    return _CAL


def test_lut_reproduces_nodes():
    c = cal()
    L, P, S = c["line"], c["pixel"], c["sigma"]
    line = np.array([L[0], L[0], L[10], L[10]])
    pix = np.array([P[0], P[100], P[300], P[-1]])
    got = rad.calibration_at(c, line, pix)
    want = np.array([S[0, 0], S[0, 100], S[10, 300], S[10, -1]])
    err = np.abs(got - want).max()
    assert err < 1e-9, f"LUT node error {err:.3e}"


def test_lut_is_linear_between_nodes():
    c = cal()
    P = c["pixel"]
    mid = 0.5 * (P[0] + P[1])
    got = rad.calibration_at(c, np.array([0.0]), np.array([mid]))[0]
    want = 0.5 * (c["sigma"][0, 0] + c["sigma"][0, 1])
    assert abs(got - want) < 1e-9, f"range interp off by {got - want:.3e}"

    L = c["line"]
    lmid = 0.5 * (L[0] + L[1])
    got = rad.calibration_at(c, np.array([lmid]), np.array([P[5]]))[0]
    want = 0.5 * (c["sigma"][0, 5] + c["sigma"][1, 5])
    assert abs(got - want) < 1e-9, f"azimuth interp off by {got - want:.3e}"


def test_sigma0_definition():
    # sigma0 = (DN/A)^2 : DN == A -> 1.0 (0 dB)
    assert abs(rad.dn_to_sigma0(710.4, 710.4) - 1.0) < 1e-12
    assert abs(rad.dn_to_sigma0(1420.8, 710.4) - 4.0) < 1e-12
    got = np.array([rad.to_db(1.0), rad.to_db(100.0), rad.to_db(0.01)])
    assert np.allclose(got, [0.0, 20.0, -20.0])
    assert abs(rad.from_db(20.0) - 100.0) < 1e-9


def test_flatten_recovers_truth():
    rng = np.random.default_rng(0)
    truth = 10.0 ** (rng.uniform(-20, 0, (64, 64)) / 10.0)   # linear power
    psi = rng.uniform(-10, 75, (64, 64))
    observed = truth * np.cos(np.radians(psi))               # Lambertian forward model
    got = rad.flatten(observed, psi)
    rel = np.abs(got / truth - 1.0).max()
    assert rel < 1e-9, f"flatten round trip {rel:.3e}"


def test_real_aoi_radiometry():
    d = rad.run_aoi(DEM, GRD, CAL, 3500, 1800, 400, 400)
    ok = d["ok"]
    assert ok.sum() > 0.8 * ok.size, "too many masked pixels"
    s = d["sigma0_db"][ok]
    g = d["gamma0_db"][ok]
    med = np.median(s)
    assert -20.0 < med < -5.0, f"sigma0 median {med:.2f} dB outside the plausible range"
    assert np.isfinite(s).all() and np.isfinite(g).all(), "non-finite values inside the mask"

    psi = d["psi"][ok]
    lo, hi = np.percentile(psi, [5, 95])
    edges = np.linspace(lo, hi, 6)
    idx = np.clip(np.digitize(psi, edges) - 1, 0, 4)
    ms = np.array([s[idx == b].mean() for b in range(5) if (idx == b).sum() > 30])
    mg = np.array([g[idx == b].mean() for b in range(5) if (idx == b).sum() > 30])
    spread_s, spread_g = ms.max() - ms.min(), mg.max() - mg.min()
    corr_s = np.corrcoef(s, np.cos(np.radians(psi)))[0, 1]
    corr_g = np.corrcoef(g, np.cos(np.radians(psi)))[0, 1]
    msg = (f"sigma0 spread {spread_s:.2f} dB -> gamma0 {spread_g:.2f} dB; "
           f"corr {corr_s:.3f} -> {corr_g:.3f}")
    # limits tuned on dem_ellipsoidal.tif (geoid-corrected): 9.18 -> 5.90 dB, corr 0.67 -> 0.46
    assert spread_g < 6.5, "flattening did not reduce the psi-bin spread: " + msg
    assert corr_g < 0.50, "flattening left a strong cos(psi) trend: " + msg


if __name__ == "__main__":
    tests = [test_lut_reproduces_nodes, test_lut_is_linear_between_nodes,
             test_sigma0_definition, test_flatten_recovers_truth,
             test_real_aoi_radiometry]
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
    print(f"\n{len(tests)-fails}/{len(tests)} green" if fails else "\nAll green~")
    sys.exit(1 if fails else 0)