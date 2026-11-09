"""Blog figure: the sub-satellite track in kilometres from the scene centre.

Replaces the lon/lat version: distances are true, no projection, computed from ECEF.

Run:
    conda run -n geo python figures/make_fig1_track_km.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sargeo import io, orbit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures", "fig1_track_km.png")

LAT, LON = 3, 4
FG = "#1f2430"
C1, C2, C3 = "#4C78A8", "#F58518", "#54A24B"

t_orb, p_orb, _ = io.orbit_arrays()
G, _ = io.grid()
lat0, lon0 = G[:, LAT].mean(), G[:, LON].mean()

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(lat0))


def to_km(lat, lon):
    return (lon - lon0) * M_PER_DEG_LON / 1e3, (lat - lat0) * M_PER_DEG_LAT / 1e3


tr = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
lon_o, lat_o, _ = tr.transform(p_orb[:, 0], p_orb[:, 1], p_orb[:, 2])
ox, oy = to_km(lat_o, lon_o)

tm = io.timing()
t_a0, t_a1 = tm["t0"], tm["t0"] + 26540 * tm["dt_az"]     # acquisition window, 39.8 s
tt = np.linspace(t_orb[0], t_orb[-1], 400)
S = orbit.position(tt)
lon_i, lat_i, _ = tr.transform(S[:, 0], S[:, 1], S[:, 2])
tx, ty = to_km(lat_i, lon_i)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": FG, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "axes.grid": True,
    "grid.color": "#d9dde3", "grid.linewidth": 0.6,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(6.2, 5.0))
inside = (tt >= t_a0) & (tt <= t_a1)
ax.plot(tx, ty, color=C1, lw=1.8, zorder=3, label="interpolated track")
ax.plot(tx[inside], ty[inside], color="#a8c6e0", lw=7.0, alpha=0.85, solid_capstyle="round",
        zorder=2, label="acquisition, 40 s")
ax.scatter(ox, oy, s=26, color=C2, zorder=3, label="17 state vectors")
ax.plot([0], [0], marker="x", ms=10, mew=2, color=C3, ls="none", zorder=4)
ax.annotate("scene centre P", xy=(0, 0), xytext=(-30, -55), fontsize=9, color=C3,
            ha="right", va="center",
            arrowprops=dict(arrowstyle="-", color=C3, lw=0.9))

ax.set_xlabel("easting from P [km]")
ax.set_ylabel("northing from P [km]")
ax.set_title("sub-satellite track in kilometres, computed from ECEF", fontsize=10)
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3,
          fontsize=8.5, handlelength=1.8, columnspacing=1.8)
ax.ticklabel_format(useOffset=False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.normpath(OUT), "| track spans",
      f"{np.ptp(tx):.0f} x {np.ptp(ty):.0f} km")