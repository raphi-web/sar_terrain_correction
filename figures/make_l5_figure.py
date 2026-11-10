"""fig8: local incidence psi + layover/shadow masks (L5)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource
from rasterio.windows import Window

from sargeo import terrain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM = os.path.join(ROOT, "dem_ellipsoidal.tif")
GRD = os.path.join(
    ROOT,
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
    "measurement",
    "s1d-iw-grd-vv-20261031t165004-20261031t165029-005262-00a00b-001-cog.tiff",
)
COL0, ROW0, W, H = 3500, 1800, 400, 400

m = terrain.mask_aoi(DEM, GRD, COL0, ROW0, W, H, step_m=52.0, kmax=100)

with rasterio.open(DEM) as ds:
    dem = ds.read(1, window=Window(COL0, ROW0, W, H)).astype(float)
ls = LightSource(azdeg=315, altdeg=45)
shade = ls.hillshade(dem, vert_exag=1.0, dx=26.03, dy=26.03)

ext = [0, W * 26.03 / 1000, 0, H * 26.03 / 1000]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), dpi=140)

ax = axes[0]
ax.imshow(shade, cmap="gray", extent=ext, vmin=0, vmax=1)
cs = ax.contour(m["psi"], levels=[0.0], colors="#d62728", linewidths=1.2,
                extent=ext, origin="upper")
ax.set_title("hillshade + ψ = 0° contour\n(inside the line: layover)", fontsize=11)
ax.set_xlabel("east [km]"); ax.set_ylabel("north [km]")

ax = axes[1]
ax.imshow(shade, cmap="gray", extent=ext, vmin=0, vmax=1)
from matplotlib.colors import ListedColormap
mask_cmap = ListedColormap(["#d40000"])
ps = ax.imshow(np.ma.masked_where(~m["layover"], m["layover"]), cmap=mask_cmap,
               alpha=0.95, extent=ext, vmin=0, vmax=1, interpolation="nearest")
ax.set_title(f"layover mask — {m['layover_frac']*100:.2f} % of pixels\n"
             "(a nearer point at larger slant range)", fontsize=11)
ax.set_xlabel("east [km]"); ax.set_ylabel("north [km]")

ax = axes[2]
im = ax.imshow(m["psi"], cmap="RdBu_r", extent=ext, vmin=-15, vmax=90)
ax.contour(m["psi"], levels=[0.0], colors="k", linewidths=0.8, extent=ext, origin="upper")
ax.set_title("projected incidence ψ [deg]\n30–46 flat; <0 layover; >90 back-facing", fontsize=11)
ax.set_xlabel("east [km]"); ax.set_ylabel("north [km]")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

fig.suptitle(f"L5 — terrain geometry, AOI {COL0},{ROW0} 400×400 (≈10 × 10 km)  |  "
             f"ψ {np.nanmin(m['psi']):.1f}…{np.nanmax(m['psi']):.1f}°, "
             f"layover {m['layover_frac']*100:.2f} %, shadow {m['shadow_frac']*100:.1f} % "
             f"(satellite 56° up, DEM max slope ~52°)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(ROOT, "figures", "fig8_terrain.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print("psi %.1f..%.1f  layover %.2f%%  shadow %.2f%%" %
      (np.nanmin(m["psi"]), np.nanmax(m["psi"]), m["layover_frac"] * 100, m["shadow_frac"] * 100))