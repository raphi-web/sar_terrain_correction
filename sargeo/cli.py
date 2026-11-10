"""Command line entry point.

    python -m sargeo.cli prepare-meta --safe <SAFE> [--pol vv] [--out meta]
    python -m sargeo.cli run --safe <SAFE> --dem dem_ellipsoidal.tif --out product.tif \
        [--pol vv] [--aoi COL ROW W H] [--chunk 512]
"""
from __future__ import annotations

import argparse
import os
import sys

from . import product, safeparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sargeo", description="Sentinel-1 terrain correction from scratch.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("prepare-meta", help="parse the annotation XML into meta/")
    pm.add_argument("safe", help=".SAFE folder (or a parent containing it)")
    pm.add_argument("--pol", default="vv", choices=["vv", "vh", "hh", "hv"])
    pm.add_argument("--out", default=None, help="output meta directory (default <repo>/meta)")

    rn = sub.add_parser("run", help="orthorectify, calibrate and flatten to a GeoTIFF")
    rn.add_argument("--safe", required=True, help=".SAFE folder (or a parent containing it)")
    rn.add_argument("--dem", required=True, help="DEM GeoTIFF in the target UTM zone")
    rn.add_argument("--out", required=True, help="output GeoTIFF")
    rn.add_argument("--pol", default="vv", choices=["vv", "vh", "hh", "hv"])
    rn.add_argument("--aoi", nargs=4, type=int, metavar=("COL", "ROW", "W", "H"),
                    help="DEM pixel window; omit to process the whole scene")
    rn.add_argument("--chunk", type=int, default=512, help="rows per strip (default 512)")
    rn.add_argument("--bands", default="dn,sigma0,gamma0",
                    help="comma separated subset of dn,sigma0,gamma0")
    rn.add_argument("--no-progress", action="store_true")
    rn.add_argument("--meta", default=None,
                    help="meta directory (default: per-scene cache under ~/.cache/sargeo)")
    rn.add_argument("--refresh-meta", action="store_true",
                    help="rebuild the meta cache even if it looks current")

    a = ap.parse_args(argv)

    if a.cmd == "prepare-meta":
        safeparse.prepare_meta(a.safe, a.out, a.pol)
        return 0

    bands = tuple(b.strip() for b in a.bands.split(",") if b.strip())
    bad = [b for b in bands if b not in product.BANDS]
    if bad:
        ap.error(f"unknown band(s) {bad}; choose from {product.BANDS}")

    res = product.write_product(a.safe, a.dem, a.out, pol=a.pol,
                                aoi=tuple(a.aoi) if a.aoi else None,
                                chunk_rows=a.chunk, bands=bands,
                                progress=not a.no_progress,
                                meta_dir=a.meta, refresh_meta=a.refresh_meta)
    print(f"\nwrote {res['out']}  {res['shape'][1]}x{res['shape'][0]}  bands={res['bands']}")
    print(f"valid {res['valid_frac']*100:.1f}%", end="")
    if "sigma0_p50" in res:
        print(f"  | sigma0 p50 {res['sigma0_p50']:.2f} dB  gamma0 p50 {res['gamma0_p50']:.2f} dB")
    else:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())