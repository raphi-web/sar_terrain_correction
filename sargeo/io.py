"""Cached metadata access — parsing was already done, see meta/README note in OUTLINE.

meta files (repo root):
  s1_meta.json    product/image dicts, 17 orbit state vectors (ECEF m, m/s), 28 coordinateConversion sets
  geolocation.npy (210,8) float64: [tau_s, line, pixel, lat_deg, lon_deg, h_ell_m, inc_deg, elev_deg]
  grid_times.npy  (210,) float64 Unix epoch seconds (azimuthTime per grid point)
  timing.json     t0 (epoch s of line 0), dt_az, nlines, nsamp
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_META = os.path.join(ROOT, "meta")
_META_DIR = os.environ.get("SARGEO_META", _DEFAULT_META)


def meta_dir() -> str:
    """Directory the readers currently use."""
    return _META_DIR


def use_meta(path: str) -> None:
    """Point the readers at another meta directory. Clears the parsed caches."""
    global _META_DIR, _ORB, _CONV
    _META_DIR = os.path.abspath(path)
    _ORB = None
    _CONV = None

TAU, LINE, PIXEL, LAT, LON, H, INC, ELEV = range(8)


def iso2s(t: str) -> float:
    """ISO 8601 -> Unix epoch seconds. Naive strings are UTC (annotation is UTC)."""
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def load_meta() -> dict:
    with open(os.path.join(_META_DIR, "s1_meta.json")) as f:
        return json.load(f)


_ORB = None


def orbit_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, pos, vel): 17 ECEF state vectors, Unix epoch seconds / m / m/s, sorted.

    Cached after the first call — the geolocation solver calls this in tight
    loops, don't re-parse the JSON per call.
    """
    global _ORB
    if _ORB is None:
        m = load_meta()
        t = np.array([iso2s(d["time"]) for d in m["orbit"]])
        pos = np.array([d["pos"] for d in m["orbit"]], dtype=float)
        vel = np.array([d["vel"] for d in m["orbit"]], dtype=float)
        order = np.argsort(t)
        _ORB = (t[order], pos[order], vel[order])
    return _ORB


def grid() -> tuple[np.ndarray, np.ndarray]:
    """((210,8) grid array, (210,) epoch seconds) — the geolocation truth grid."""
    return (np.load(os.path.join(_META_DIR, "geolocation.npy")),
            np.load(os.path.join(_META_DIR, "grid_times.npy")))


def timing() -> dict:
    with open(os.path.join(_META_DIR, "timing.json")) as f:
        return json.load(f)


_CONV = None


def conversions() -> tuple[np.ndarray, list[dict]]:
    """(times (28,) epoch seconds, 28 coefficient sets sorted by time).

    Each set: azimuthTime (ISO), slantRangeTime, sr0 (m), gr0 (m),
              srgr (9 coeffs, SR->GR), grsr (9 coeffs, GR->SR).
    Cached after the first call — safe to call in loops.
    """
    global _CONV
    if _CONV is None:
        ccs = load_meta()["coordinateConversion"]
        t = np.array([iso2s(c["azimuthTime"]) for c in ccs])
        order = np.argsort(t)
        _CONV = (t[order], [ccs[i] for i in order])
    return _CONV
