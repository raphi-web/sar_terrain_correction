"""L6 assignment: radiometric calibration + terrain flattening.

WHY
  L4 placed the pixels, L5 told you the local angle. L6 makes the *values*
  mean something: DN -> sigma0 in dB -> terrain-flattened gamma0.

THE CALIBRATION XML
  annotation/calibration/calibration-...vv....xml contains
      count = 27  <calibrationVector>, one every ~1 s of azimuth time
      each has: line, pixel[665], sigmaNought[665], betaNought[665], gamma[665]
  The arrays are CALIBRATION CONSTANTS A, not backscatter values. The product:

      sigma0 = (DN / A_sigma)^2        linear power, then 10*log10 -> dB
      beta0  = (DN / A_beta)^2
      gamma0 = (DN / A_gamma)^2

  Measured on this product: A_beta is constant 507.64 across range; A_sigma
  falls 710.4 -> 596.9; and (A_beta/A_sigma)^2 == sin(incidence). That is the
  proof that sigma0 is per GROUND area while beta0 is per SLANT-range area.

TERRAIN FLATTENING
  A slope facing the radar presents more illuminated ground area per pixel, so
  it looks brighter. For a Lambertian surface:

      sigma0_obs = sigma0_true * cos(psi)      (psi from L5, signed, degrees)
      gamma0     = sigma0_obs / cos(psi)       -> sigma0_true

  On this 400x400 AOI the range-bin mean spread collapses 6.43 dB -> 3.07 dB
  and corr(brightness, cos psi) drops 0.51 -> 0.23. The residual is real land
  cover (forest vs meadow), not a bug. Clip cos psi at a floor (0.05) so a
  near-grazing pixel cannot explode to +30 dB.

  NOTE: this is the cosine correction, not Small 2011's full area-based RTC.
  Good enough to show the effect, honest enough to say so in the blog.

API
    load_calibration(path)                       (provided)
    calibration_at(cal, line, pixel, key)        -> A, any shape
    dn_to_sigma0(dn, a)                          -> linear sigma0
    to_db(x) / from_db(x)
    flatten(sigma0_lin, psi_deg)                 -> gamma0 linear
    run_aoi(...)                                 (provided glue)

Acceptance: tests/test_l6_radiometry.py (6 tests).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def load_calibration(path: str) -> dict:
    """Parse the calibration XML into arrays. (Provided.)"""
    root = ET.parse(path).getroot()
    vecs = root.findall(".//calibrationVector")
    lines, pixels = [], None
    sig, bet, gam = [], [], []
    for v in vecs:
        lines.append(float(v.findtext("line")))
        p = np.array(v.findtext("pixel").split(), float)
        pixels = p if pixels is None else pixels
        sig.append(np.array(v.findtext("sigmaNought").split(), float))
        bet.append(np.array(v.findtext("betaNought").split(), float))
        gam.append(np.array(v.findtext("gamma").split(), float))
    return {"line": np.array(lines), "pixel": pixels,
            "sigma": np.array(sig), "beta": np.array(bet), "gamma": np.array(gam)}


def calibration_at(cal: dict, line, pixel, key: str = "sigma"):
    """Interpolate the calibration constants at (line, pixel). See STEP 1."""
    # STEP 1 — bilinear lookup with two DIFFERENT grids:
    #   * azimuth: 27 values at cal["line"] -> linear between neighbours
    #   * range:   665 values at cal["pixel"] -> linear between neighbours
    #   Vectorized recipe (no python loop over 160k pixels!):
    L = cal["line"]; P = cal["pixel"]
    S = cal[key]
    i0 = np.clip(np.searchsorted(L, line) - 1, 0, 25)          # left vector
    w  = np.clip((line - L[i0]) / (L[i0+1] - L[i0]), 0, 1)  # azimuth weight
    j  = np.clip(np.searchsorted(P, pixel) - 1, 0, 663)
    t  = np.clip((pixel - P[j]) / (P[j+1] - P[j]), 0, 1)    # range weight
    a0 = S[i0, j]*(1-t) + S[i0, j+1]*t        # vector i0
    a1 = S[i0+1, j]*(1-t) + S[i0+1, j+1]*t    # vector i0+1
    return a0*(1-w) + a1*w


def dn_to_sigma0(dn, a):
    """DN -> linear sigma0. See STEP 2."""
    # STEP 2 — sigma0 = (DN / A)^2. Both are arrays, any shape, same shape out.
    return (dn / a) **2




def to_db(x):
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(x)
    return np.where(x > 0.0, db, np.nan)


def flatten(sigma0_lin, psi_deg):
    cospsi = np.clip(np.cos(np.radians(psi_deg)), 0.05, 1.0)
    return sigma0_lin / cospsi


def from_db(x):
    """dB -> linear power."""
    return 10.0 ** (np.asarray(x, float) / 10.0)


def run_aoi(dem_path: str, grd_path: str, cal_path: str,
            col0: int, row0: int, w: int, h: int) -> dict:
    """Full L4+L5+L6 driver on a DEM-aligned window. (Provided glue.)"""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window

    from . import ortho, terrain

    m = terrain.mask_aoi(dem_path, grd_path, col0, row0, w, h)
    with rasterio.open(dem_path) as ds:
        win = Window(col0, row0, w, h)
        dem = ds.read(1, window=win).astype(float)
        tr = ds.window_transform(win)
        crs = ds.crs
    cols = np.arange(w) + 0.5
    rows = np.arange(h) + 0.5
    EE, NN = np.meshgrid(tr.c + cols * tr.a, tr.f + rows * tr.e)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(EE, NN)
    line, pixel = ortho.geocode_tile(lon.ravel(), lat.ravel(), dem.ravel())

    l0 = max(int(np.floor(line.min())) - 3, 0)
    p0 = max(int(np.floor(pixel.min())) - 3, 0)
    with rasterio.open(grd_path) as ds:
        l1 = min(int(np.ceil(line.max())) + 3, ds.height)
        p1 = min(int(np.ceil(pixel.max())) + 3, ds.width)
        dn_full = ds.read(1, window=Window(p0, l0, p1 - p0, l1 - l0)).astype(float)
    dn = ortho.sample_bilinear(dn_full, line - l0, pixel - p0, nodata=np.nan).reshape(h, w)

    cal = load_calibration(cal_path)
    a = calibration_at(cal, line, pixel).reshape(h, w)
    s0 = dn_to_sigma0(dn, a)
    s0_db = to_db(s0)
    g0_db = to_db(flatten(s0, m["psi"]))
    ok = (np.isfinite(s0_db) & np.isfinite(m["psi"]) & (m["psi"] < 80.0)
          & ~m["layover"] & ~m["shadow"])
    return {"sigma0_db": s0_db, "gamma0_db": g0_db, "psi": m["psi"],
            "ok": ok, "dn": dn, "a": a}