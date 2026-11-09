#!/usr/bin/env python3
"""Build the meta/ cache from a Sentinel-1 COG SAFE.

    python prepare_meta.py path/to/S1D_..._COG.SAFE
    python prepare_meta.py path/to/S1D_..._COG.SAFE --pol vh --out meta_vh

Thin wrapper around sargeo.safeparse, so the same code can be used from Python.
"""
import sys

from sargeo.safeparse import main

if __name__ == "__main__":
    sys.exit(main())