"""Blog figure: the pass in the plane through the scene centre and the satellite.

No lon/lat here. Everything is computed from the ECEF vectors: the up direction at P,
the horizontal look direction, and the satellite arc projected onto that plane.
The Earth's surface is drawn with its real curvature, so the picture shows why a
satellite 876 km away is still 700 km up and 53 degrees above the horizon.

Run:
    conda run -n geo python figures/make_fig1_geometry.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sargeo import io, orbit
from sargeo.ellipsoid import lla2ecef

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures", "fig1_geometry.png")

LAT, LON = 3, 4
FG = "#1f2430"
C1, C2, C3, C4 = "#4C78A8", "#F58518", "#54A24B", "#E45756"
R_EARTH = 6371.0

t_orb, p_orb, _ = io.orbit_arrays()
G, _ = io.grid()
lat0, lon0 = G[:, LAT].mean(), G[:, LON].mean()

P = lla2ecef(lat0, lon0, 0.0) / 1e3                       # scene centre, km
u = lla2ecef(lat0, lon0, 1000.0) / 1e3 - P               # geodetic up, unit-ish
u /= np.linalg.norm(u)

tt = np.linspace(t_orb[0], t_orb[-1], 400)
S = orbit.position(tt) / 1e3
k = int(np.argmin(np.linalg.norm(S - P, axis=1)))

look = S[k] - P
look = look - u * np.dot(look, u)                        # horizontal part
e = look / np.linalg.norm(look)                          # along-track ground direction


def plane(X):
    """ECEF km -> (ground distance, altitude) in the plane of the pass."""
    D = X - P
    return np.dot(D, e), np.dot(D, u)


sx, sy = plane(S)
ox, oy = plane(p_orb / 1e3)

rng = np.linalg.norm(S[k] - P)
elev = np.degrees(np.arcsin(np.dot(S[k] - P, u) / rng))
alt = np.dot(S[k] - P, u)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": FG, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "axes.grid": True,
    "grid.color": "#d9dde3", "grid.linewidth": 0.6,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(6.6, 4.6))

X0, X1 = -160.0, 980.0
xs = np.linspace(X0, X1, 500)
ground = -xs**2 / (2 * R_EARTH)
ax.fill_between(xs, ground - 40, ground, color="#e8ebef", edgecolor="#c3c9d2", lw=0.8, zorder=0)

ax.plot(sx, sy, color=C1, lw=2.4, label="interpolated arc, 160 s of orbit table", zorder=5)
ax.scatter(ox, oy, s=16, color=C2, zorder=6, label="17 orbit state vectors")
ax.plot([0], [0], marker="x", ms=10, mew=2, color=C3, ls="none", zorder=8)
ax.annotate("scene centre P", xy=(0, 0), xytext=(30, -78), fontsize=9, color=C3,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=C3, lw=0.9))

ax.plot([sx[k], 0], [sy[k], 0], color=C4, lw=1.4, ls="--", zorder=4,
        label=f"line of sight {rng:.0f} km, elevation {elev:.0f}°")

ax.plot([sx[k], sx[k]], [sy[k], ground_at_k := -sx[k]**2 / (2 * R_EARTH)],
        color="#9aa3ad", lw=0.9, ls=":", zorder=3)
ax.annotate(f"altitude {alt:.0f} km", xy=(sx[k], alt / 2), xytext=(sx[k] - 120, alt / 2 - 120),
            fontsize=8.5, color="#6b7280",
            arrowprops=dict(arrowstyle="-", color="#9aa3ad", lw=0.8))
ax.annotate("sub-satellite point", xy=(sx[k], ground_at_k), xytext=(sx[k] + 40, -75),
            fontsize=8.5, color="#6b7280",
            arrowprops=dict(arrowstyle="-", color="#9aa3ad", lw=0.8))

ax.set_xlabel("ground distance along the pass [km]")
ax.set_ylabel("altitude above the ellipsoid [km]")
ax.set_title("the pass in the plane of the orbit, computed from the ECEF vectors")
ax.set_xlim(X0, X1)
ax.set_ylim(-95, 880)
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {os.path.normpath(OUT)} | range {rng:.0f} km, elevation {elev:.1f} deg, altitude {alt:.0f} km")