"""Write a multiband product (DN, sigma0, gamma0) from a SAFE plus a DEM.

Pure numpy, chunked over rows so peak memory stays flat. This is the slow path
by design: it is the reference implementation. A Numba variant of the
geolocation kernel gives about 26x end to end, but it is not part of this repo.
"""
from __future__ import annotations

import json
import os

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from . import io, ortho, orbit, radiometry, terrain, safeparse

BANDS = ("dn", "sigma0", "gamma0")


def scene_extent(dem_path: str, pad_px: int = 0) -> dict:
    """DEM-aligned window covering the scene footprint (from the geolocation grid)."""
    G, _ = io.grid()
    with rasterio.open(dem_path) as ds:
        crs, tr = ds.crs, ds.transform
        H, W = ds.height, ds.width

    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    E, N = tf.transform(G[:, io.LON], G[:, io.LAT])
    # the grid samples the interior, so pad by a few pixels for safety
    left = max(float(E.min()) - pad_px * abs(tr.a), tr.c)
    right = min(float(E.max()) + pad_px * abs(tr.a), tr.c + W * tr.a)
    bottom = max(float(N.min()) - pad_px * abs(tr.e), tr.f + H * tr.e)
    top = min(float(N.max()) + pad_px * abs(tr.e), tr.f)

    res = abs(tr.a)
    # snap the bbox to the DEM grid with round(), so the window is centred on the
    # footprint rather than biased by a pixel from floor/ceil
    col0 = max(int(round((left - tr.c) / res)), 0)
    row0 = max(int(round((tr.f - top) / res)), 0)
    w = int(np.ceil((right - left) / res))
    h = int(np.ceil((top - bottom) / abs(tr.e)))
    w = min(w, W - col0)
    h = min(h, H - row0)
    return {"col0": col0, "row0": row0, "w": w, "h": h,
            "E": [float(E.min()), float(E.max())], "N": [float(N.min()), float(N.max())]}


def write_product(safe: str, dem_path: str, out_path: str, pol: str = "vv",
                  aoi: tuple[int, int, int, int] | None = None,
                  chunk_rows: int = 512, bands=BANDS, progress: bool = True,
                  meta_dir: str | None = None, refresh_meta: bool = False) -> dict:
    """Run the pipeline over an AOI (DEM pixel window) or the whole scene.

    aoi = (col0, row0, width, height) in DEM pixels. None -> full scene extent.
    The meta cache is built or reused automatically for this scene.
    """
    grd, ann, cal_path = safeparse.find_product(safe, pol)
    if cal_path is None:
        raise FileNotFoundError("no calibration XML in this product")

    md = safeparse.ensure_meta(safe, pol, meta_dir=meta_dir, refresh=refresh_meta,
                              verbose=progress)
    io.use_meta(md)

    if aoi is None:
        ext = scene_extent(dem_path)
        col0, row0, nx, ny = ext["col0"], ext["row0"], ext["w"], ext["h"]
    else:
        col0, row0, nx, ny = (int(v) for v in aoi)

    with rasterio.open(dem_path) as ds:
        dem_tr, crs = ds.transform, ds.crs
    out_tr = rasterio.windows.transform(Window(col0, row0, nx, ny), dem_tr)

    cal = radiometry.load_calibration(cal_path)
    tim = io.timing()
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    profile = dict(driver="GTiff", width=nx, height=ny, count=len(bands), crs=crs,
                   transform=out_tr, dtype="float32", nodata=np.nan,
                   tiled=True, blockxsize=256, blockysize=256,
                   compress="deflate", predictor=3)

    stats = {"valid": 0, "total": 0, "s0": [], "g0": []}
    n_bands = {"dn": 1, "sigma0": 2, "gamma0": 3}

    with rasterio.open(out_path, "w", **profile) as dst:
        for y in range(0, ny, chunk_rows):
            h = min(chunk_rows, ny - y)
            with rasterio.open(dem_path) as ds:
                dem = ds.read(1, window=Window(col0, row0 + y, nx, h)).astype(np.float64)
            dem = np.where(dem <= -9000, np.nan, dem)

            cols = np.arange(nx) + 0.5
            rows = np.arange(h) + 0.5
            EE, NN = np.meshgrid(out_tr.c + cols * out_tr.a,
                                 out_tr.f + (y + rows) * out_tr.e)
            lon, lat = tf.transform(EE.ravel(), NN.ravel())

            line, pixel = ortho.geocode_tile(lon, lat, dem.ravel())

            l0 = max(int(np.floor(np.nanmin(line))) - 3, 0)
            p0 = max(int(np.floor(np.nanmin(pixel))) - 3, 0)
            with rasterio.open(grd) as ds:
                l1 = min(int(np.ceil(np.nanmax(line))) + 3, ds.height)
                p1 = min(int(np.ceil(np.nanmax(pixel))) + 3, ds.width)
                dn_full = ds.read(1, window=Window(p0, l0, p1 - p0, l1 - l0)).astype(np.float32)

            dn = ortho.sample_bilinear(dn_full, line - l0, pixel - p0, nodata=np.nan)
            valid = np.isfinite(dn) & (dn > 0)

            a = radiometry.calibration_at(cal, line, pixel)
            s0 = radiometry.dn_to_sigma0(np.where(valid, dn, np.nan), a)
            s0_db = radiometry.to_db(s0).reshape(h, nx)

            nrm = terrain.surface_normals(dem, abs(out_tr.a), abs(out_tr.e),
                                          lon.reshape(h, nx), lat.reshape(h, nx))
            t_az = tim["t0"] + line * tim["dt_az"]
            P = terrain.lla2ecef(lat, lon, dem.ravel())
            S = orbit.position(t_az)
            up = terrain.geodetic_up(lat, lon)
            psi = terrain.projected_incidence(nrm.reshape(-1, 3), P, S, up).reshape(h, nx)
            g0_db = radiometry.to_db(radiometry.flatten(s0, psi.ravel())).reshape(h, nx)

            arrays = {"dn": np.where(valid, dn, np.nan).reshape(h, nx),
                      "sigma0": s0_db, "gamma0": g0_db}
            for name in bands:
                dst.write(arrays[name].astype(np.float32), n_bands[name],
                          window=Window(0, y, nx, h))

            ok = np.isfinite(s0_db)
            stats["valid"] += int(ok.sum())
            stats["total"] += ok.size
            if ok.any():
                stats["s0"].append(np.percentile(s0_db[ok], [5, 50, 95]))
                stats["g0"].append(np.percentile(g0_db[ok], [5, 50, 95]))
            if progress:
                frac = 100.0 * stats["valid"] / max(stats["total"], 1)
                print(f"  rows {y:5d}-{y+h-1:5d}  valid {frac:5.1f}%", flush=True)

    res = {"out": out_path, "bands": list(bands), "shape": [ny, nx],
           "valid_frac": stats["valid"] / max(stats["total"], 1)}
    if stats["s0"]:
        res["sigma0_p50"] = float(np.median([p[1] for p in stats["s0"]]))
        res["gamma0_p50"] = float(np.median([p[1] for p in stats["g0"]]))
    return res