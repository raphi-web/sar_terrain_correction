"""fig9: sigma0 vs terrain-flattened gamma0 (L6)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DEM = os.path.join(ROOT, "dem.tif")
COL0, ROW0, W, H = 3500, 1800, 400, 400

d = rad.run_aoi(DEM, GRD, CAL, COL0, ROW0, W, H)
ok = d["ok"]
s, g, psi = d["sigma0_db"], d["gamma0_db"], d["psi"]

vmin, vmax = np.percentile(s[ok], [2, 98])
ext = [0, W * 26.03 / 1000, 0, H * 26.03 / 1000]

fig = plt.figure(figsize=(16, 5.4), dpi=140)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.28)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(np.ma.masked_where(~ok, s), cmap="gray", extent=ext, vmin=vmin, vmax=vmax)
ax.set_title(f"σ⁰ [dB] — raw calibration\nmedian {np.median(s[ok]):.1f} dB", fontsize=11)
ax.set_xlabel("east [km]"); ax.set_ylabel("north [km]")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(np.ma.masked_where(~ok, g), cmap="gray", extent=ext, vmin=vmin, vmax=vmax)
ax.set_title(f"γ⁰ [dB] — after cos ψ correction\nmedian {np.median(g[ok]):.1f} dB", fontsize=11)
ax.set_xlabel("east [km]"); ax.set_ylabel("north [km]")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

ax = fig.add_subplot(gs[0, 2])
cs = np.cos(np.radians(psi[ok]))
edges = np.linspace(np.percentile(cs, 2), np.percentile(cs, 98), 11)
idx = np.clip(np.digitize(cs, edges) - 1, 0, 9)
ctr = 0.5 * (edges[:-1] + edges[1:])
ms = np.array([s[ok][idx == b].mean() for b in range(10)])
mg = np.array([g[ok][idx == b].mean() for b in range(10)])
ax.plot(ctr, ms, "o-", color="#c0392b", lw=2, label="σ⁰ (raw)")
ax.plot(ctr, mg, "o-", color="#1f77b4", lw=2, label="γ⁰ (flattened)")
ax.set_xlabel("cos ψ  (1 = facing the radar straight on)")
ax.set_ylabel("mean backscatter [dB]")
ax.set_title(f"terrain bias: {ms.max()-ms.min():.2f} dB → {mg.max()-mg.min():.2f} dB", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(frameon=False)
ax.annotate("slopes facing the radar\nlook bright", xy=(ctr[0], ms[0]), xytext=(0.15, 0.25),
            textcoords="axes fraction", fontsize=9, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

fig.suptitle("L6 — radiometric calibration and terrain flattening  |  "
             f"AOI {COL0},{ROW0} 400×400  |  corr with cos ψ: "
             f"{np.corrcoef(s[ok], cs)[0,1]:.2f} → {np.corrcoef(g[ok], cs)[0,1]:.2f}",
             fontsize=11)
fig.savefig(os.path.join(ROOT, "figures", "fig9_radiometry.png"), bbox_inches="tight")
print("wrote figures/fig9_radiometry.png")
print("spread %.2f -> %.2f dB | corr %.3f -> %.3f" % (
    ms.max() - ms.min(), mg.max() - mg.min(),
    np.corrcoef(s[ok], cs)[0, 1], np.corrcoef(g[ok], cs)[0, 1]))