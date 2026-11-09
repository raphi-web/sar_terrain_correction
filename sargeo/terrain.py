"""L5 assignment: local incidence + layover/shadow masks.

WHY
  L4 told you *where* every map pixel came from. L5 asks *whether the radar
  could see it at all*, and at what angle:
    - local incidence -> the radiometric correction in L6 needs it
    - layover / shadow -> where the image contains no usable ground information

THE GEOMETRY
  For a pixel P (ECEF) and the satellite S at that pixel's zero-Doppler time:
      s  = (S - P) / |S - P|            line of sight, unit
      up = geodetic normal at P         (function below, provided)
      i  = acos(up . s)                 flat-Earth incidence
  On terrain, replace `up` with the surface normal n from the DEM, and work in
  the RANGE PLANE (the plane containing S, P and the Earth centre):

      m  = normalize(s x up)            normal of the range plane
      n1 = normalize(n - m (n.m))       surface normal projected into that plane
      q  = normalize(up - s (up.s))     "up" inside the range plane, perpendicular to s
      psi = atan2(n1.q, n1.s)           SIGNED projected incidence, degrees

  psi is the quantity that matters:
      psi ~ i (30-46 deg)   flat ground, illuminated
      psi < 0               LAYOVER: the surface tilts toward the radar past the
                            line of sight -> range ordering inverts
      psi > 90              back-facing -> SHADOW (self-shadowed)

  That catches self-shadowing, but real radar shadow is often caused by a
  *ridge blocking the view* of ground that faces the right way.  So the mask
  comes from a HORIZON WALK, like SNAP does:

      walk k = 1..kmax steps (step_m metres) along the look direction toward
      the satellite; at each step sample the DEM and the slant range r.
        horizon angle  alpha_k = atan2(h_k - h, k*step_m)
        shadow  = max_k(alpha_k) > elev_sat     (elev_sat = 90 - i, per pixel)
        layover = max_k(r_k) > r                (a nearer point farther in range)
      Use np.fmax, not np.maximum: bilinear sampling returns NaN outside the
      window, and maximum(-inf, NaN) is NaN — one poisoned step would kill the
      whole mask.  (I learned that the hard way, measuring this lesson.)

API
    geodetic_up(lat, lon)                                  (provided)
    surface_normals(dem, d_east, d_north, lon, lat)        -> (H,W,3) ECEF
    projected_incidence(normals, points, sensors, up)      -> (H,W) psi deg
    horizon_masks(r, dem, d_east, d_north, dx, dy,
                  step_m, kmax, elev_sat)                  -> (layover, shadow)
    mask_aoi(...)                                           (provided glue)

Acceptance: tests/test_l5_terrain.py (5 tests).
"""

from __future__ import annotations

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from . import io, orbit, ortho
from .ellipsoid import lla2ecef


def geodetic_up(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Unit ellipsoid normal (ECEF) at lat/lon [deg]. Provided helper.

    """
    la = np.radians(lat)
    lo = np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


def surface_normals(dem, d_east, d_north, lon, lat):
    """
     converts a DEM height field into a 3D unit normal vector in ECEF coordinates.
    """
    gy = -np.gradient(dem, d_north, axis=0)      # rows grow SOUTH
    gx = np.gradient(dem, d_east, axis=1)

    la, lo = np.radians(lat), np.radians(lon)
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], axis=-1)
    north = np.stack([-np.sin(la) * np.cos(lo),
                      -np.sin(la) * np.sin(lo),
                      np.cos(la)], axis=-1)
    up = geodetic_up(lat, lon)

    n = up - gx[..., None] * east - gy[..., None] * north
    return n / np.linalg.norm(n, axis=-1, keepdims=True)

def projected_incidence(normals: np.ndarray, points: np.ndarray,
                        sensors: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Signed projected local incidence psi [deg]."""
    s = sensors - points
    s /= np.linalg.norm(s, axis=-1, keepdims=True)

    m = np.cross(s, up)
    m /= np.linalg.norm(m, axis=-1, keepdims=True)

    n1 = normals - m * np.sum(normals * m, axis=-1, keepdims=True)
    n1 /= np.linalg.norm(n1, axis=-1, keepdims=True)

    q = up - s * np.sum(up * s, axis=-1, keepdims=True)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)

    return np.degrees(np.arctan2(np.sum(n1 * q, axis=-1), np.sum(n1 * s, axis=-1)))



def horizon_masks(r: np.ndarray, dem: np.ndarray, d_east: float, d_north: float,
                  dx: float, dy: float, step_m: float, kmax: int,
                  elev_sat: np.ndarray):
    """Walk along the look direction -> (layover, shadow) booleans. See STEP 3."""
    # STEP 3 — the walk (all steps vectorized over the tile):
    H, W = dem.shape
    rows, cols = np.mgrid[0:H, 0:W]
    dcol = step_m * d_east / dx
    drow = -step_m * d_north / dy
    max_horiz = -90 ; max_near_r = -np.inf
    for k in range(1, kmax):
        hk = ortho.sample_bilinear(dem, rows + k * drow, cols + k * dcol, nodata=np.nan)
        rk = ortho.sample_bilinear(r, rows + k * drow, cols + k * dcol, nodata=np.nan)
        max_horiz  = np.fmax(max_horiz,  np.degrees(np.arctan2(hk - dem, k*step_m)))
        max_near_r = np.fmax(max_near_r, rk)
    layover = max_near_r > r
    shadow  = max_horiz > elev_sat
    return layover, shadow


def mask_aoi(dem_path: str, grd_path: str, col0: int, row0: int, w: int, h: int,
             step_m: float = 52.0, kmax: int = 100) -> dict:
    """Full driver on a DEM-aligned window. (Provided glue.)"""
    with rasterio.open(dem_path) as ds:
        win = Window(col0, row0, w, h)
        dem = ds.read(1, window=win).astype(float)
        transform = ds.window_transform(win)
        crs = ds.crs
    dx, dy = abs(transform.a), abs(transform.e)

    cols = np.arange(w) + 0.5
    rows = np.arange(h) + 0.5
    EE, NN = np.meshgrid(transform.c + cols * transform.a,
                         transform.f + rows * transform.e)
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(EE, NN)

    n = surface_normals(dem, dx, dy, lon, lat)
    line, pixel = ortho.geocode_tile(lon.ravel(), lat.ravel(), dem.ravel())
    tim = io.timing()
    t = tim["t0"] + line * tim["dt_az"]
    P = lla2ecef(lat.ravel(), lon.ravel(), dem.ravel())
    S = orbit.position(t)
    up = geodetic_up(lat.ravel(), lon.ravel())
    psi = projected_incidence(n.reshape(-1, 3), P, S, up).reshape(h, w)

    rng = np.linalg.norm(S - P, axis=1)
    r = rng.reshape(h, w)
    s_unit = (S - P) / rng[:, None]
    inc = np.degrees(np.arccos(np.sum(s_unit * up, axis=1))).reshape(h, w)
    elev_sat = 90.0 - inc

    tf_ecef = Transformer.from_crs("EPSG:4978", crs, always_xy=True)
    Se, Sn, _ = tf_ecef.transform(S[:, 0], S[:, 1], S[:, 2])
    centre_e = transform.c + 0.5 * w * transform.a
    centre_n = transform.f + 0.5 * h * transform.e
    dE = float(np.mean(Se)) - centre_e
    dN = float(np.mean(Sn)) - centre_n
    norm = np.hypot(dE, dN)
    dE, dN = dE / norm, dN / norm

    layover, shadow = horizon_masks(r, dem, dE, dN, dx, dy, step_m, kmax, elev_sat)
    return {"psi": psi, "layover": layover, "shadow": shadow,
            "layover_frac": float(layover.mean()),
            "shadow_frac": float(shadow.mean()),
            "valid": np.isfinite(psi)}