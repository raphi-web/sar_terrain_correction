"""Figures for the SAR course: L1 orbit, L1 Hermite, L2 SRGR, L3 Doppler, scene, errors.

Run:
    python figures/make_figures.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sargeo import io
from sargeo.ellipsoid import lla2ecef

import rasterio
from rasterio.enums import Resampling
from pyproj import Transformer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
C0 = 299_792_458.0
TAU, LINE, PIX, LAT, LON, H, INC, ELEV = range(8)

BG = "#ffffff"
FG = "#1f2430"
C1, C2, C3, C4, C5 = "#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": FG, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "axes.grid": True,
    "grid.color": "#d9dde3", "grid.linewidth": 0.6,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 150,
})

t_orb, p_orb, v_orb = io.orbit_arrays()
G, tg = io.grid()
sr_true = C0 * G[:, TAU] / 2.0
gr_px = G[:, PIX] * 10.0
t0 = io.timing()["t0"]
dt_az = io.timing()["dt_az"]


# ---------------------------------------------------------------- helpers
def hermite(tt, t, p, v):
    tt = np.atleast_1d(tt).astype(float)
    i = np.clip(np.searchsorted(t, tt) - 1, 0, len(t) - 2)
    h = (t[i + 1] - t[i])[:, None]
    s = ((tt - t[i]) / (t[i + 1] - t[i]))[:, None]
    p0, p1, v0, v1 = p[i], p[i + 1], v[i] * h, v[i + 1] * h
    return (2 * s**3 - 3 * s**2 + 1) * p0 + (s**3 - 2 * s**2 + s) * v0 + \
           (-2 * s**3 + 3 * s**2) * p1 + (s**3 - s**2) * v1


def solve_doppler(P):
    lo, hi = float(t_orb[0]), float(t_orb[-1])

    def f(tt):
        S = hermite(tt, t_orb, p_orb, v_orb)[0]
        V = _hermite_vel(tt)[0]
        return float(np.dot(S - P, V))

    f_lo = f(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (f_lo > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _hermite_vel(tt):
    tt = np.atleast_1d(tt).astype(float)
    i = np.clip(np.searchsorted(t_orb, tt) - 1, 0, len(t_orb) - 2)
    h = (t_orb[i + 1] - t_orb[i])
    s = (tt - t_orb[i]) / h
    p0, p1, v0, v1 = p_orb[i], p_orb[i + 1], v_orb[i] * h[:, None], v_orb[i + 1] * h[:, None]
    d00 = 6 * s**2 - 6 * s
    d10 = 3 * s**2 - 4 * s + 1
    d01 = -6 * s**2 + 6 * s
    d11 = 3 * s**2 - 2 * s
    return (d00[:, None] * p0 + d10[:, None] * v0 + d01[:, None] * p1 + d11[:, None] * v1) / h[:, None]


# ============================================================ 1. orbit
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

tr = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
lon_o, lat_o, alt_o = tr.transform(p_orb[:, 0], p_orb[:, 1], p_orb[:, 2])
tt = np.linspace(t_orb[0], t_orb[-1], 400)
S = hermite(tt, t_orb, p_orb, v_orb)
lon_i, lat_i, alt_i = tr.transform(S[:, 0], S[:, 1], S[:, 2])

ax[0].plot(lon_i, lat_i, color=C1, lw=1.6, label="interpolated track")
ax[0].scatter(lon_o, lat_o, s=22, color=C2, zorder=3, label="17 orbit state vectors")
ax[0].scatter([G[:, LON].mean()], [G[:, LAT].mean()], marker="x", s=60, color=C3, label="scene centre")
ax[0].set_xlabel("longitude [deg]")
ax[0].set_ylabel("latitude [deg]")
ax[0].set_title("L1 · sub-satellite ground track (25 s)")
ax[0].legend(loc="lower right")
ax[0].ticklabel_format(useOffset=False)

sec = tt - t0
ax2 = ax[1]
ax2.plot(sec, alt_i / 1e3, color=C1, lw=1.8, label="altitude above ellipsoid")
ax2.set_xlabel("seconds from first line")
ax2.set_ylabel("altitude [km]", color=C1)
ax2.tick_params(axis="y", labelcolor=C1)
ax2b = ax2.twinx()
ax2b.plot(sec, np.linalg.norm(_hermite_vel(tt), axis=1) / 1e3, color=C3, lw=1.6, ls="--")
ax2b.set_ylabel("speed [km/s]", color=C3)
ax2b.tick_params(axis="y", labelcolor=C3)
ax2b.grid(False)
ax2.set_title("L1 · orbit is nearly circular")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_orbit.png"), bbox_inches="tight")
plt.close(fig)

# ============================================================ 2. hermite
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
s = np.linspace(0, 1, 200)
ax[0].plot(s, 2 * s**3 - 3 * s**2 + 1, color=C1, lw=1.8, label="h00 · p_i")
ax[0].plot(s, s**3 - 2 * s**2 + s, color=C2, lw=1.8, label="h10 · h·v_i")
ax[0].plot(s, -2 * s**3 + 3 * s**2, color=C3, lw=1.8, label="h01 · p_next")
ax[0].plot(s, s**3 - s**2, color=C4, lw=1.8, label="h11 · h·v_next")
ax[0].axhline(0, color=FG, lw=0.8)
ax[0].set_xlabel("s = (t − t_i) / h")
ax[0].set_ylabel("weight")
ax[0].set_title("L1 · Hermite blending functions")
ax[0].legend(loc="center right", fontsize=8.5)

k = 5
ti, tn = t_orb[k], t_orb[k + 1]
tts = np.linspace(ti, tn, 300)
dev = np.linalg.norm(hermite(tts, t_orb, p_orb, v_orb) -
                     0.5 * (p_orb[k] + p_orb[k + 1]), axis=1) if False else None
chord = p_orb[k][None, :] + (p_orb[k + 1] - p_orb[k])[None, :] * \
        ((tts - ti) / (tn - ti))[:, None]
dev = np.linalg.norm(hermite(tts, t_orb, p_orb, v_orb) - chord, axis=1)
ax[1].plot((tts - ti) * 1e3, dev, color=C1, lw=1.8)
ax[1].fill_between((tts - ti) * 1e3, dev, color=C1, alpha=0.12)
imax = int(np.argmax(dev))
ax[1].scatter([(tts[imax] - ti) * 1e3], [dev[imax]], color=C2, zorder=3)
ax[1].annotate(f"max: {dev[imax]:.1f} m",
               xy=((tts[imax] - ti) * 1e3, dev[imax]),
               xytext=(0.52, 0.40), textcoords="axes fraction", fontsize=9,
               arrowprops=dict(arrowstyle="-", color=FG, lw=0.8))
ax[1].set_xlabel("milliseconds into the 10 s segment")
ax[1].set_ylabel("distance from linear chord [m]")
ax[1].set_title("L1 · chord error over one 10 s segment")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_hermite.png"), bbox_inches="tight")
plt.close(fig)

# ============================================================ 3. SRGR
T, sets = io.conversions()
c = sets[len(sets) // 2]
gr = np.linspace(0, 265_390, 4000)
sr = np.polynomial.polynomial.polyval(gr, c["grsr"])

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax[0].plot(gr / 1e3, sr / 1e3, color=C1, lw=1.8)
ax[0].scatter(gr_px / 1e3, sr_true / 1e3, s=10, color=C2, label="geolocation grid")
ax[0].set_xlabel("ground range [km]")
ax[0].set_ylabel("slant range [km]")
ax[0].set_title("L2 · ground → slant mapping (grsr polynomial)")
ax[0].legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)
ax[0].annotate(f"near edge: sr = {sr[0]/1e3:.3f} km", (gr[0], sr[0] / 1e3),
               textcoords="offset points", xytext=(8, 16), fontsize=8.5)
ax[0].annotate(f"far edge: sr = {sr[-1]/1e3:.3f} km", (gr[-1] / 1e3, sr[-1] / 1e3),
               textcoords="offset points", xytext=(-118, -18), fontsize=8.5)

slope = np.gradient(sr, gr)
ax[1].plot(gr / 1e3, slope, color=C1, lw=1.8, label="d(sr)/d(gr) from polynomial")
ax[1].scatter(gr_px / 1e3, np.sin(np.radians(G[:, INC])), s=12, color=C2,
              label="sin(incidence) from grid")
ax[1].set_xlabel("ground range [km]")
ax[1].set_ylabel("d(sr)/d(gr)  =  sin i")
ax[1].set_title("L2 · slope = sin(incidence): the geometry is baked into the fit")
ax[1].legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_srgr.png"), bbox_inches="tight")
plt.close(fig)

# ============================================================ 4. doppler
idx = 100
P = lla2ecef(G[idx, LAT], G[idx, LON], G[idx, H])
ts = np.linspace(t_orb[0], t_orb[-1], 800)
f = np.array([np.dot(hermite(tv, t_orb, p_orb, v_orb)[0] - P, _hermite_vel(tv)[0]) for tv in ts])
tstar = solve_doppler(P)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax[0].plot((ts - t0), f / 1e6, color=C1, lw=1.8)
ax[0].axhline(0, color=FG, lw=0.9)
ax[0].axvline(tstar - t0, color=C2, lw=1.2, ls="--", label="solved t*")
ax[0].set_xlabel("seconds from first line")
ax[0].set_ylabel("(S − P)·V  [10⁶ m²/s]")
ax[0].set_title("L3 · zero-Doppler function, monotone through one root")
ax[0].legend()

zoom = np.linspace(tstar - 1e-3, tstar + 1e-3, 400)
fz = np.array([np.dot(hermite(tv, t_orb, p_orb, v_orb)[0] - P, _hermite_vel(tv)[0]) for tv in zoom])
ax[1].plot((zoom - tstar) * 1e3, fz / 1e4, color=C1, lw=1.8)
ax[1].axhline(0, color=FG, lw=0.9)
ax[1].axvline(0, color=C2, lw=1.2, ls="--")
tgrid = tg[idx]
ax[1].axvline((tgrid - tstar) * 1e3, color=C4, lw=1.2, ls=":")
ax[1].set_xlabel("milliseconds from solved t*")
ax[1].set_ylabel("(S − P)·V  [10⁴ m²/s]")
ax[1].set_title("L3 · zoom ±1 ms: solved t* vs the product's grid time")
ax[1].annotate(f"Δt* − t_grid = {(tgrid - tstar)*1e3:.3f} ms\n(the line residual 0.19 px comes from the\ngrid's line↔time rounding, not the solver)",
               ((tgrid - tstar) * 1e3, 0), textcoords="offset points", xytext=(12, 30), fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_doppler.png"), bbox_inches="tight")
plt.close(fig)

# ============================================================ 5. scene
safe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
                    "S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE",
                    "measurement",
                    "s1d-iw-grd-vv-20261031t165004-20261031t165029-005262-00a00b-001-cog.tiff")
h, w = 800, 1260
with rasterio.open(safe) as ds:
    dn = ds.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
    nlines, nsamp = ds.height, ds.width
db = 20 * np.log10(np.maximum(dn, 1))
db[dn <= 0] = np.nan          # GRD padding / nodata: paint as background, not black
vmin, vmax = np.nanpercentile(db, [2, 98])

dem_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dem.tif")
with rasterio.open(dem_path) as ds:
    dem = ds.read(1, out_shape=(800, 1092), resampling=Resampling.bilinear).astype(float)
    bounds = ds.bounds
gy, gx = np.gradient(dem, 26.03)
slope = np.arctan(np.hypot(gx, gy))
aspect = np.arctan2(-gx, gy)
az, alt = np.radians(315), np.radians(45)
hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax[0].imshow(db, cmap="gray", vmin=vmin, vmax=vmax, extent=[0, nsamp, nlines, 0])
ax[0].set_facecolor(BG)
ax[0].scatter(G[:, PIX], G[:, LINE], s=6, color=C2, alpha=0.85, label="geolocation grid")
ax[0].set_title("L0 · Sentinel-1 IW GRD VV, DN (dB)")
ax[0].set_xlabel("range pixel"); ax[0].set_ylabel("azimuth line")
ax[0].legend(loc="lower right", fontsize=8)
ax[0].grid(False)

ax[1].imshow(hs, cmap="gray", extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
ax[1].set_title("L0 · DEM (26 m) with scene footprint")
ax[1].set_xlabel("UTM 33N easting [m]"); ax[1].set_ylabel("northing [m]")
ax[1].ticklabel_format(useOffset=False, style="plain")
ax[1].grid(False)
foot = dict(left=385_642, right=674_883, bottom=5_073_954, top=5_287_134)
ax[1].plot([foot["left"], foot["right"], foot["right"], foot["left"], foot["left"]],
           [foot["bottom"], foot["bottom"], foot["top"], foot["top"], foot["bottom"]],
           color=C2, lw=1.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_scene.png"), bbox_inches="tight")
plt.close(fig)

# ============================================================ 6. errors
e_line, e_pix, e_r, dop = [], [], [], []
for n in range(len(G)):
    t = solve_doppler(lla2ecef(G[n, LAT], G[n, LON], G[n, H]))
    S_ = hermite(t, t_orb, p_orb, v_orb)[0]
    r = float(np.linalg.norm(S_ - lla2ecef(G[n, LAT], G[n, LON], G[n, H])))
    gr_ = float(np.polynomial.polynomial.polyval(r - sets[int(np.argmin(np.abs(T - t)))]["sr0"],
                                                  sets[int(np.argmin(np.abs(T - t)))]["srgr"]))
    e_line.append((t - t0) / dt_az - G[n, LINE])
    e_pix.append(gr_ / 10.0 - G[n, PIX])
    e_r.append(r - sr_true[n])
    dop.append(np.dot(S_ - lla2ecef(G[n, LAT], G[n, LON], G[n, H]), _hermite_vel(t)[0]))
e_line, e_pix, e_r, dop = map(np.array, (e_line, e_pix, e_r, dop))

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
ax[0].scatter(G[:, PIX], e_line, s=10, color=C1)
ax[0].set_xlabel("range pixel"); ax[0].set_ylabel("line error [px]")
ax[0].set_title(f"L3 · azimuth: max {np.abs(e_line).max():.3f} px")
ax[0].axhline(0, color=FG, lw=0.7)
ax[1].scatter(G[:, H], e_pix, s=10, color=C3)
ax[1].set_xlabel("grid height [m]"); ax[1].set_ylabel("pixel error [px]")
ax[1].set_title(f"L3 · range: max {np.abs(e_pix).max():.4f} px")
ax[1].axhline(0, color=FG, lw=0.7)
ax[2].hist(np.abs(dop), bins=30, color=C4, alpha=0.8)
ax[2].set_xlabel("|(S−P)·V| at solution [m²/s]")
ax[2].set_ylabel("grid points")
ax[2].set_title(f"L3 · Doppler residual: max {np.abs(dop).max():.1f} m²/s")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_errors.png"), bbox_inches="tight")
plt.close(fig)

print("done; files:")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print(" ", f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")