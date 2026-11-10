"""Convert the Copernicus DEM from EGM2008 orthometric to WGS84 ellipsoidal heights.

Copernicus DEM heights H are above the EGM2008 geoid; lla2ecef needs heights h
above the WGS84 ellipsoid. h = H + N, with N the geoid height (44.6-47.6 m over
this scene). N comes from PROJ's EGM2008 grid (us_nga_egm08_25.tif), fetched
from the PROJ CDN on first use.

Run once:
    python prepare_dem.py            # dem.tif -> dem_ellipsoidal.tif
"""
import os
import sys

import numpy as np
import pyproj
import rasterio
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "dem.tif")
DST = os.path.join(ROOT, "dem_ellipsoidal.tif")
STEP = 64        # px between geoid samples (~1.7 km); the EGM2008 grid is 2.5' (~4.6 km)

pyproj.network.set_network_enabled(True)
# only_best: raise instead of silently returning N = 0 when the grid is missing
geoid = Transformer.from_crs("EPSG:4326+3855", "EPSG:4979", always_xy=True, only_best=True)

with rasterio.open(SRC) as src:
    profile = src.profile
    tf, crs, nodata = src.transform, src.crs, src.nodata
    H, W = src.height, src.width
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # geoid height N on a coarse grid of pixel centres, covering the full raster
    rows = np.unique(np.r_[np.arange(0, H, STEP), H - 1]) + 0.5
    cols = np.unique(np.r_[np.arange(0, W, STEP), W - 1]) + 0.5
    cc, rr = np.meshgrid(cols, rows)
    e, n = tf * (cc, rr)
    lon, lat = to_ll.transform(e, n)
    N = geoid.transform(lon, lat, np.zeros_like(lon))[2]
    if not (np.all(np.isfinite(N)) and 30 < N.min() and N.max() < 60):
        sys.exit(f"geoid heights look wrong ({N.min():.2f}..{N.max():.2f} m), not writing")
    print(f"geoid height N: {N.min():.2f} .. {N.max():.2f} m")
    interp = RegularGridInterpolator((rows, cols), N)

    profile.update(tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=3)
    with rasterio.open(DST, "w", **profile) as dst:
        all_cols = np.arange(W) + 0.5
        for r0 in range(0, H, 512):
            r1 = min(r0 + 512, H)
            win = rasterio.windows.Window(0, r0, W, r1 - r0)
            dem = src.read(1, window=win)
            rr, cc = np.meshgrid(np.arange(r0, r1) + 0.5, all_cols, indexing="ij")
            Nb = interp(np.column_stack([rr.ravel(), cc.ravel()])).reshape(dem.shape)
            out = np.where(dem == nodata, nodata, dem + Nb).astype(np.float32)
            dst.write(out, 1, window=win)

print("wrote", DST)
