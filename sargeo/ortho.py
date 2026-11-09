from __future__ import annotations

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from . import io, orbit  # noqa: F401  (your code will need io.conversions() / io.timing())
from .ellipsoid import lla2ecef
from .geo import zero_doppler_time

DX = 10.0


def geocode_tile(lon, lat, h):
    """
    Convert a tile of geographic coordinates into SAR image coordinates.

    For every input ground point:
        1. Convert geodetic coordinates (lon, lat, h) to ECEF coordinates.
        2. Find the azimuth acquisition time using the zero-Doppler condition:

              (S(t) - P) · V(t) = 0

           where:
              S(t) = satellite position
              V(t) = satellite velocity
              P   = ground point

           This determines the time when the radar was looking exactly at
           the target point.

        3. Compute the satellite-to-ground distance (slant range).
        4. Convert slant range to ground range using the
           SAR range conversion polynomial.
        5. Convert acquisition time and ground range into SAR image
           coordinates (line, pixel).

    The implementation is fully vectorized: each input point is solved
    independently, allowing an entire image tile to be geocoded at once.

    Args:
        lon: Longitude values in degrees, shape (N,)
        lat: Latitude values in degrees, shape (N,)
        h:   Ellipsoid height in meters, shape (N,)

    Returns:
        line: SAR azimuth line coordinates, shape (N,)
        pixel: SAR range pixel coordinates, shape (N,)
    """

    # Convert all geographic points into Earth Centered Earth Fixed (ECEF)
    # coordinates.
    # Output:
    #   P = [[X,Y,Z],
    #        [X,Y,Z],
    #        ...]
    #
    # shape: (N,3)
    P = lla2ecef(lat, lon, h)

    # The zero-Doppler time can only exist between the first and last
    # available orbit state vectors.
    #
    # Instead of one interval:
    #
    #       [lo ---------------- hi]
    #
    # we maintain one interval per pixel:
    #
    #       pixel 1: [lo1 -> hi1]
    #       pixel 2: [lo2 -> hi2]
    #
    # This allows all points to be solved simultaneously.

    t_orb, _, _ = io.orbit_arrays()

    lo = np.full(len(P), float(t_orb[0]))
    hi = np.full(len(P), float(t_orb[-1]))

    # The zero-Doppler condition is:
    # D(t) = (S(t)-P) · V(t) = 0
    #
    # this means that the radar line-of-sight vector is
    # perpendicular to the satellite flight direction.
    #
    # A positive value means the satellite is moving toward one side of  closest approach
    # a negative value means the opposite side.
    #
    # The root of this function gives the acquisition time.
    #
    # Because we process N pixels simultaneously:
    #
    #       S -> (N,3)
    #       V -> (N,3)
    #       output -> (N,)
    #
    # np.sum(..., axis=1) performs a separate dot product for every point.
    def f(t):
        S = orbit.position(t)
        V = orbit.velocity(t)

        return np.sum((S - P) * V, axis=1)

    # Evaluate the Doppler function at the lower boundary.
    # Bisection only needs the sign of the function at the boundaries.
    f_lo = f(lo)

    # Vectorized bisection search for zero-Doppler time.
    # Each iteration halves the search interval for every pixel.
    # Scalar version:
    #
    #       if f(mid) has same sign as f(lo):
    #             move lo -> mid
    #       else:
    #             move hi -> mid

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)

        # Pixels where midpoint is still on the same side of the root
        # as the lower boundary.
        same = (fm > 0) == (f_lo > 0)
        lo = np.where(same, mid, lo) # Move only those lower boundaries upward.
        hi = np.where(same, hi, mid) # For the remaining pixels the root lies below midpoint.

        # Keep stored lower-bound function values synchronized.
        f_lo = np.where(same, fm, f_lo)

    # The final interval midpoint is our zero-Doppler acquisition time.
    t = 0.5 * (lo + hi)

    # Compute slant range.
    S = orbit.position(t)
    r = np.linalg.norm(S - P, axis=1)

    # Convert slant range -> ground range.
    # The conversion polynomial changes with azimuth time because the
    # satellite geometry changes along the orbit.
    # Every pixel needs its own conversion coefficients.
    #
    # The metadata provides conversion polynomials at discrete times:
    #       time0 -> polynomial0
    #       time1 -> polynomial1
    #       ...
    # Select the closest polynomial for every pixel.
    T, sets = io.conversions()

    sr0 = np.array([s["sr0"] for s in sets])
    C = np.array([s["srgr"] for s in sets])

    # Find nearest conversion time independently for every pixel.
    idx = np.argmin(
        np.abs(T[None, :] - t[:, None]),
        axis=1
    )

    # Shift range origin for numerical stability.
    x = r - sr0[idx]

    # Evaluate:
    #     ground_range = a0 + a1*x + a2*x² + ...
    # This loop evaluates the polynomial terms
    gr = np.zeros_like(x)
    for i in range(9):
        gr += C[idx, i] * x ** i

    # Convert physical coordinates into SAR image coordinates.
    # Azimuth:
    #
    #       line = (acquisition_time - image_start_time)
    #              / time_between_lines
    #       pixel = ground_range / 10 m
    tim = io.timing()

    line = (t - tim["t0"]) / tim["dt_az"]
    pixel = gr / 10.0

    return line, pixel


def sample_bilinear(dn, line, pixel, nodata=0.0):
    # perform bilinear interpolation of sar to dem grid
    hh, ww = dn.shape
    l0 = np.floor(line).astype(np.int64)       # row above
    p0 = np.floor(pixel).astype(np.int64)      # column left
    dl = line - l0                             # 0..1 down the cell
    dp = pixel - p0                            # 0..1 across the cell

    ok = (l0 >= 0) & (l0 + 1 < hh) & (p0 >= 0) & (p0 + 1 < ww)
    l0c = np.clip(l0, 0, hh - 2)               # safe indices for the gather
    p0c = np.clip(p0, 0, ww - 2)

    v00 = dn[l0c, p0c].astype(float)           # top-left
    v01 = dn[l0c, p0c + 1].astype(float)       # top-right
    v10 = dn[l0c + 1, p0c].astype(float)       # bottom-left
    v11 = dn[l0c + 1, p0c + 1].astype(float)   # bottom-right

    out = (v00 * (1 - dl) * (1 - dp) + v01 * (1 - dl) * dp +
           v10 * dl * (1 - dp) + v11 * dl * dp)
    out[~ok] = nodata
    return out
def run_aoi(
    dem_path: str,
    grd_path: str,
    out_path: str,
    col0: int,
    row0: int,
    w: int,
    h: int,
    nodata: int = 0
) -> dict:
    """
    Geocode and sample a SAR image for an area of interest (AOI).

    The output raster is aligned to the DEM grid. For every DEM pixel:
        DEM pixel -> lon/lat/height -> SAR geolocation -> SAR line/pixel coordinates -> Bilinear SAR sampling -> output raster

    Args:
        dem_path:
            DEM GeoTIFF used as the target grid.

        grd_path:
            SAR intensity image.

        out_path:
            Output GeoTIFF.

        col0,row0:
            Upper-left pixel of the DEM window.

        w,h:
            Size of the DEM processing tile.

        nodata:
            Output value for invalid SAR locations.

    Returns:
        Dictionary containing processing statistics.
    """

    # SAR is sensitive to height because the radar line-of-sight
    # intersects the terrain at different positions depending on
    # elevation.
    # ---------------------------------------------------------------
    with rasterio.open(dem_path) as ds:
        win = Window(col0, row0, w, h)
        dem = ds.read(1, window=win).astype(float)
        transform = ds.window_transform(win)
        crs = ds.crs



    # Generate DEM pixel center coordinates.
    cols = np.arange(w) + 0.5
    rows = np.arange(h) + 0.5


    # Convert pixel coordinates into projected coordinates.
    # x = origin_x + column * pixel_width
    # y = origin_y + row    * pixel_height
    E = transform.c + cols * transform.a
    N = transform.f + rows * transform.e
    # (Easting, Northing, height)
    EE, NN = np.meshgrid(E, N)

    # Transform DEM coordinates into longitude/latitude.
    #
    # SAR orbit calculations are performed in ECEF/WGS84 coordinates,
    # therefore we need geographic coordinates before converting them
    # to ECEF inside geocode_tile().
    tf = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True
    )

    lon, lat = tf.transform(
        EE.ravel(),
        NN.ravel()
    )

    # Convert every DEM pixel into SAR coordinates.
    line, pixel = geocode_tile(
        lon,
        lat,
        dem.ravel()
    )

    # Remove invalid geolocation results.
    # Invalid values can occur if:
    #  point is outside SAR acquisition
    #  - orbit does not cover the point
    #  - zero Doppler search fails

    ok = (
        np.isfinite(line)
        &
        np.isfinite(pixel)
    )

    if not ok.any():
        raise RuntimeError(
            "no valid geolocation in this window"
        )

    # Reading the complete SAR image would be very expensive.
    # we find the bounding box of all required SAR pixels.

    y0 = max(
        int(np.floor(np.min(line[ok]))) - 2,
        0
    )

    y1 = int(
        np.ceil(np.max(line[ok]))
    ) + 3

    x0 = max(
        int(np.floor(np.min(pixel[ok]))) - 2,
        0
    )

    x1 = int(
        np.ceil(np.max(pixel[ok]))
    ) + 3

    # Read only the required SAR subset.
    with rasterio.open(grd_path) as ds:
        y1 = min(y1, ds.height)
        x1 = min(x1, ds.width)

        dn = ds.read(
            1,
            window=Window(
                x0,
                y0,
                x1 - x0,
                y1 - y0
            )
        )

    # Sample SAR values at the geocoded positions.
    # line/pixel currently refer to the full SAR image.
    # shift coordinates relative to that window:
    #       global pixel 12345
    # becomes:
    #       local pixel 25
    val = sample_bilinear(
        dn,
        line - y0,
        pixel - x0,
        nodata=nodata
    )

    # geocode_tile worked on flattened arrays because vector operations
    out = val.reshape(h, w).astype(np.uint16)
    # Write the SAR-derived values using the DEM geometry.
    # The output is now aligned with the DEM:
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:

        dst.write(out, 1)

    valid = out > 0

    return {
        "valid_frac": float(valid.mean()),
        "src_window": (y0, y1, x0, x1)
    }