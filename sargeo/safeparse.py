"""Parse a Sentinel-1 COG SAFE into the meta/ cache.

Everything the pipeline needs from the annotation XML lives here:
    meta/s1_meta.json     orbit state vectors + SRGR/GRSR polynomial sets
    meta/geolocation.npy  (n,8): [slantRangeTime_s, line, pixel, lat, lon, h, inc, elev]
    meta/grid_times.npy   (n,)  Unix epoch seconds of each grid point
    meta/timing.json      t0, dt_az, nlines, nsamp

COG products ship no XML namespaces, but this parser also accepts namespaced
(older) annotation files by stripping the namespace on read.

Times: annotation timestamps are UTC but written without a zone. They are
parsed as UTC explicitly, so the cache does not depend on the machine's
timezone. (An earlier version used naive parsing, which shifted every epoch by
the local offset; internally consistent, but not portable.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np

C0 = 299_792_458.0


def iso2s(t: str) -> float:
    """'2026-10-31T16:50:04.048046' -> Unix epoch seconds, interpreted as UTC."""
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _hermite_positions(t, t_orb, p_orb, v_orb):
    """Cubic Hermite S(t), same basis as sargeo.orbit.position, no meta dependency."""
    t = np.atleast_1d(np.asarray(t, float))
    i = np.clip(np.searchsorted(t_orb, t) - 1, 0, len(t_orb) - 2)
    h = t_orb[i + 1] - t_orb[i]
    s = (t - t_orb[i]) / h
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return (h00[:, None] * p_orb[i] + h10[:, None] * (h[:, None] * v_orb[i])
            + h01[:, None] * p_orb[i + 1] + h11[:, None] * (h[:, None] * v_orb[i + 1]))


def find_product(safe: str, pol: str = "vv"):
    """Locate (measurement_tiff, annotation_xml, calibration_xml) inside a SAFE.

    Accepts either the .SAFE directory itself or a parent directory containing it.
    Handles the COG layout where the SAFE is nested once more.
    """
    hits = glob.glob(os.path.join(safe, "**", "measurement", f"*grd-{pol}-*.tiff"), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(safe, "**", f"*grd-{pol}-*.tiff"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no GRD {pol.upper()} measurement under {safe}")
    grd = sorted(hits, key=len)[0]

    ann_dir = os.path.join(os.path.dirname(os.path.dirname(grd)), "annotation")
    ann = [p for p in glob.glob(os.path.join(ann_dir, f"*grd-{pol}-*.xml"))
           if os.path.basename(p).startswith("s1") or os.path.basename(p).startswith("s2")]
    if not ann:
        raise FileNotFoundError(f"no annotation XML under {ann_dir}")
    cal = [p for p in glob.glob(os.path.join(ann_dir, "calibration", f"*grd-{pol}-*.xml"))
           if os.path.basename(p).startswith("calibration-")]
    return grd, ann[0], (cal[0] if cal else None)


def prepare_meta(safe: str, out_dir: str | None = None, pol: str = "vv",
                 verbose: bool = True) -> dict:
    """Write the meta/ cache for one SAFE. Returns a summary dict."""
    grd, ann_path, cal_path = find_product(safe, pol)
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(grd)), "..", "..", "..", "meta")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    root = _strip_ns(ET.parse(ann_path).getroot())

    img = root.find(".//imageInformation")
    if img is None:
        raise ValueError("annotation has no imageInformation block")
    t0 = iso2s(img.findtext("productFirstLineUtcTime"))
    dt_az = float(img.findtext("azimuthTimeInterval"))
    nlines = int(img.findtext("numberOfLines"))
    nsamp = int(img.findtext("numberOfSamples"))

    orbit = []
    for o in root.findall(".//orbitList/orbit"):
        orbit.append({
            "time": o.findtext("time"),
            "frame": o.findtext("frame"),
            "pos": [float(o.findtext("position/x")), float(o.findtext("position/y")),
                    float(o.findtext("position/z"))],
            "vel": [float(o.findtext("velocity/x")), float(o.findtext("velocity/y")),
                    float(o.findtext("velocity/z"))],
        })

    conv = []
    for c in root.findall(".//coordinateConversionList/coordinateConversion"):
        conv.append({
            "azimuthTime": c.findtext("azimuthTime"),
            "slantRangeTime": float(c.findtext("slantRangeTime")),
            "sr0": float(c.findtext("sr0")),
            "gr0": float(c.findtext("gr0")),
            "srgr": [float(v) for v in c.findtext("srgrCoefficients").split()],
            "grsr": [float(v) for v in c.findtext("grsrCoefficients").split()],
        })

    grid, gtimes = [], []
    for g in root.findall(".//geolocationGridPointList/geolocationGridPoint"):
        grid.append([
            float(g.findtext("slantRangeTime")), float(g.findtext("line")),
            float(g.findtext("pixel")), float(g.findtext("latitude")),
            float(g.findtext("longitude")), float(g.findtext("height")),
            float(g.findtext("incidenceAngle")), float(g.findtext("elevationAngle")),
        ])
        gtimes.append(iso2s(g.findtext("azimuthTime")))

    if len(orbit) < 2:
        raise ValueError(f"expected more than one orbit state vector, got {len(orbit)}")
    if not conv:
        raise ValueError("no coordinateConversion sets in the annotation")

    meta = {
        "product": {
            "pass": root.findtext(".//productInformation/pass"),
            "timelinessCategory": root.findtext(".//productInformation/timelinessCategory"),
            "projection": root.findtext(".//productInformation/projection"),
            "polarisation": pol.upper(),
        },
        "image": {
            "productFirstLineUtcTime": img.findtext("productFirstLineUtcTime"),
            "productLastLineUtcTime": img.findtext("productLastLineUtcTime"),
            "azimuthTimeInterval": dt_az,
            "numberOfLines": nlines,
            "numberOfSamples": nsamp,
        },
        "orbit": orbit,
        "coordinateConversion": conv,
        "wgs84": {"a": 6378137.0, "b": 6356752.314245, "f": 0.003352810664774516},
        "c0": C0,
    }

    with open(os.path.join(out_dir, "s1_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    np.save(os.path.join(out_dir, "geolocation.npy"), np.asarray(grid, float))
    np.save(os.path.join(out_dir, "grid_times.npy"), np.asarray(gtimes, float))

    # free sanity check: |S(t) - P| must equal c*tau/2 at every grid point.
    # Hermite is inlined here on purpose: prepare_meta must not depend on whatever
    # meta directory sargeo.io happens to have active.
    t_orb = np.array([iso2s(o["time"]) for o in orbit])
    p_orb = np.array([o["pos"] for o in orbit])
    v_orb = np.array([o["vel"] for o in orbit])
    order = np.argsort(t_orb)
    t_orb, p_orb, v_orb = t_orb[order], p_orb[order], v_orb[order]
    g = np.asarray(grid, float)

    from .ellipsoid import lla2ecef

    P = lla2ecef(g[:, 3], g[:, 4], g[:, 5])
    S = _hermite_positions(np.array(gtimes), t_orb, p_orb, v_orb)
    r_geom = np.linalg.norm(S - P, axis=1)
    r_meas = C0 * g[:, 0] / 2.0
    resid = r_geom - r_meas

    timing = {"t0": t0, "dt_az": dt_az, "nlines": nlines, "nsamp": nsamp,
              "resid_stats": {"min": float(resid.min()), "max": float(resid.max()),
                              "mean": float(resid.mean()), "std": float(resid.std())}}
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=1)

    summary = {"meta_dir": out_dir, "measurement": grd, "annotation": ann_path,
               "calibration": cal_path, "orbit": len(orbit), "grid": len(grid),
               "conversions": len(conv), "resid_max_m": float(np.abs(resid).max())}
    if verbose:
        print(f"meta written to {out_dir}")
        print(f"  orbit {len(orbit)} | grid {len(grid)} | conversion sets {len(conv)}")
        print(f"  range residual vs c*tau/2: max {np.abs(resid).max()*1000:.3f} mm")
    return summary


def product_id(safe: str) -> str:
    """Folder name of the SAFE, without the extension: stable cache key."""
    base = os.path.basename(os.path.abspath(safe.rstrip("/")))
    return base[:-5] if base.endswith(".SAFE") else base


def default_cache_dir(safe: str, pol: str = "vv") -> str:
    """Per-scene cache under the user cache dir, so scenes never collide."""
    root = os.environ.get("SARGEO_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "sargeo"))
    return os.path.join(root, f"{product_id(safe)}-{pol}")


def _stamp_path(d: str) -> str:
    return os.path.join(d, "source.json")


def _cache_ok(d: str, grd: str) -> bool:
    need = ("s1_meta.json", "geolocation.npy", "grid_times.npy", "timing.json", "source.json")
    if not all(os.path.isfile(os.path.join(d, f)) for f in need):
        return False
    try:
        st = json.load(open(_stamp_path(d)))
    except (OSError, ValueError):
        return False
    return (st.get("measurement") == os.path.abspath(grd)
            and st.get("size") == os.path.getsize(grd)
            and st.get("mtime") == os.path.getmtime(grd))


def ensure_meta(safe: str, pol: str = "vv", meta_dir: str | None = None,
                refresh: bool = False, verbose: bool = True) -> str:
    """Return a meta directory for this scene, building it if missing or stale."""
    grd, _, _ = find_product(safe, pol)
    d = os.path.abspath(meta_dir) if meta_dir else default_cache_dir(safe, pol)

    if not refresh and _cache_ok(d, grd):
        if verbose:
            print(f"meta: reusing {d}")
        return d

    if verbose:
        print(f"meta: {'rebuilding' if refresh else 'building'} {d}")
    prepare_meta(safe, d, pol, verbose=verbose)
    with open(_stamp_path(d), "w") as f:
        json.dump({"measurement": os.path.abspath(grd),
                   "size": os.path.getsize(grd),
                   "mtime": os.path.getmtime(grd),
                   "pol": pol}, f, indent=1)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the meta/ cache from a Sentinel-1 COG SAFE.")
    ap.add_argument("safe", help="path to a .SAFE folder (or its parent)")
    ap.add_argument("--out", default=None, help="output meta directory (default: <repo>/meta)")
    ap.add_argument("--pol", default="vv", choices=["vv", "vh", "hh", "hv"])
    a = ap.parse_args(argv)
    prepare_meta(a.safe, a.out, a.pol)


if __name__ == "__main__":
    sys.exit(main())