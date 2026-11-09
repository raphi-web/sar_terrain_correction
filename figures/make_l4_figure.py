"""L4 payoff figure: source GRD (radar geometry) vs the ortho output (map geometry) vs DEM.

Run:  python figures/make_l4_figure.py
"""
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sargeo.ortho import run_aoi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM = os.path.join(ROOT, "dem.tif")
GRD = os.path.join(
    ROOT,
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
    "measurement",
    "s1d-iw-grd-vv-20261031t165004-20261031t165029-005262-00a00b-001-cog.tiff",
)
OUT_TIF = os.path.join(ROOT, "figures", "l4_aoi_1000.tif")
COL0, ROW0, W, H = 4500, 2500, 1000, 1000

t0 = time.time()
stats = run_aoi(DEM, GRD, OUT_TIF, COL0, ROW0, W, H)
dt = time.time() - t0

with rasterio.open(OUT_TIF) as ds:
    ortho = ds.read(1)
    transform = ds.transform
    crs = ds.crs

y0, y1, x0, x1 = stats["src_window"]
with rasterio.open(GRD) as ds:
    src = ds.read(1, window=Window(x0, y0, x1 - x0, y1 - y0))

with rasterio.open(DEM) as ds:
    dem = ds.read(1, window=Window(COL0, ROW0, W, H)).astype(float)

gy, gx = np.gradient(dem, 26.03)
slope = np.arctan(np.hypot(gx, gy))
aspect = np.arctan2(-gx, gy)
az, alt = np.radians(315), np.radians(45)
hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)

plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 150,
})

fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6))

def db(a):
    a = a.astype(float)
    a[a <= 0] = np.nan
    d = 20 * np.log10(a)
    return d, np.nanpercentile(d, [2, 98])

d_src, (v0, v1) = db(src)
ax[0].imshow(d_src, cmap="gray", vmin=v0, vmax=v1)
ax[0].set_title(f"source GRD (radar geometry)\n{src.shape[1]}×{src.shape[0]} px window")
ax[0].set_xlabel("range pixel")
ax[0].set_ylabel("azimuth line")

d_orth, (w0, w1) = db(ortho)
ax[1].imshow(d_orth, cmap="gray", vmin=w0, vmax=w1)
ax[1].set_title(f"YOUR ortho output (UTM 33N, 26 m)\n{W}×{H} px in {dt:.1f} s")
ax[1].set_xlabel("northing → easting")
ax[1].set_ylabel("")

ax[2].imshow(hs, cmap="gray")
ax[2].set_title("DEM hillshade (same window)")
ax[2].set_xlabel("")
ax[2].set_ylabel("")

for a in ax:
    a.set_xticks([])
    a.set_yticks([])
    a.grid(False)

fig.suptitle("L4 · the same landscape, three coordinate systems — ridges must line up between the last two",
             fontsize=11, fontweight="bold")
fig.tight_layout()
out_png = os.path.join(ROOT, "figures", "fig7_ortho.png")
fig.savefig(out_png, bbox_inches="tight")
print(f"AOI {W}x{H} in {dt:.1f} s | valid {stats['valid_frac']*100:.1f}% | src window {stats['src_window']}")
print("wrote", out_png)