# SAR terrain correction from scratch

Turns a Sentinel-1 IW GRD product into an
orthorectified, radiometrically calibrated, terrain-flattened GeoTIFF. Pure Python,
NumPy plus rasterio and pyproj, no SNAP, no GAMMA.

The result can be checked: every S1 product carries a 210-point geolocation grid, and this pipeline
reproduces it to sub-millimeter range.

| Check (against the product's own grid) | Result |
|---|---|
| Slant range | **0.2 mm** max |
| Azimuth line | **0.191 px** max |
| Range pixel | **0.0153 px** max |

The 0.19 px azimuth residual is the grid's own time quantization (0.28 ms), not the solver.

## What is implemented

| Module | Function |
|---|---|
| `sargeo/ellipsoid.py` | geodetic to ECEF, WGS84 |
| `sargeo/orbit.py` | cubic Hermite interpolation of the 17 orbit state vectors |
| `sargeo/srgr.py` | slant range <-> ground range via the 28 per-second polynomial sets |
| `sargeo/geo.py` | zero-Doppler acquisition time (bisection on `(S-P)·V`) |
| `sargeo/ortho.py` | per map pixel inverse geocoding, bilinear resampling, GeoTIFF output |
| `sargeo/terrain.py` | DEM surface normals, signed projected incidence, layover and shadow masks (horizon walk) |
| `sargeo/radiometry.py` | calibration lookup, DN to sigma0, cosine terrain flattening |

Each module has a test file with tolerances measured against the product.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 or newer. Tested with numpy 2.5, rasterio 1.5, pyproj 3.7.

## Data

Two inputs, both public:

1. **Sentinel-1 IW GRD** in the COG SAFE layout, from [Copernicus Data Space](https://dataspace.copernicus.eu/).
   Unzip it so the `.SAFE` folder sits next to `sargeo/`.
2. **Copernicus DEM GLO-30**, reprojected to the UTM zone of the scene, saved as `dem.tif`.

```
<repo>/
  dem.tif
  S1D_IW_GRDH_..._COG.SAFE/
  meta/            pre-parsed annotation for the demo scene (60 KB)
  sargeo/
  tests/
```

`meta/` holds the orbit, geolocation grid, SRGR coefficients and timing extracted from
the annotation XML, so the tests run without touching the 1.2 GB measurement files for
the geometry checks. It is specific to this scene; regenerating it for another product
is on the roadmap.

## Run the tests

```bash
python tests/test_l1_orbit.py     # 5 tests
python tests/test_l2_srgr.py      # 5
python tests/test_l3_geo.py       # 5
python tests/test_l4_ortho.py     # 4
python tests/test_l5_terrain.py   # 6
python tests/test_l6_radiometry.py # 5
```

A small orthorectified AOI:

```python
from sargeo import ortho

stats = ortho.run_aoi("dem.tif", "measurement/....tiff", "out.tif",
                      col0=3500, row0=1800, w=1000, h=1000)
```

## Validation numbers, for the curious

- Terrain flattening on a 10 x 10 km Alpine AOI: mean backscatter spread over
  incidence bins falls from **7.43 dB to 3.50 dB**, correlation with `cos(psi)` from
  **0.51 to 0.23**.
- Layover covers 0.5 to 1.8 percent of that area. Shadow is zero, and that is correct:
  at 34 degree incidence the satellite stands 56 degrees above the horizon, while the
  26 m DEM peaks near 52 degrees of slope.

## Known limitations

- Cosine terrain correction only, not the full area based RTC from Small (2011).
- No thermal noise removal, no antenna pattern, no TOPS deramping.
- Ellipsoid heights throughout, no geoid.
- GRD only: no phase, so no polarimetric decomposition and no interferometry.
- No speckle filtering in the products; multi-looking only helps if the output grid is
  also coarsened.
- Tests are pinned to the included demo scene.

## Related

A Numba accelerated version of the geocoding kernel lives in a separate working copy
and is not part of this repository.

## License

GPL-2.0. ESA SNAP (GPL) was used as a behavioural reference while writing this code;
no SNAP source code was copied.
