---
layout: default
title: "I am scared of SAR"
permalink: /
---

# I am scared of SAR 

And I wanted to change that. For that I picked a over the top exercise that cost me some headache along the way. But I did learn and it did make Synthetic Apture radar less scarry.
**So let show you how you can terrain correct your own Sentinel-1 GRD products from scratch using Python, Numpy and Rasterio!**

## Why do we need Terrain correction anyway? 

The scene I am working with is a ascending orbit Sentinel-1 GRD Product. Downloaded from ESA-Copernicus.
GRD means that the radar is already projected onto a flat Earth ellipsoid model (WGS84). 
It has also been Multi-Looked which means adjacent pixels where averaged together to smooth out speckle noise.  


The exact file i am working with:  
`S1D_IW_GRDH_1SDV_20261031T165004_20261031T165029_005262_00A00B_5EE3_COG.SAFE`

The figure below shows the S1-Image on the left and a Digital Elevation model on the right.
Notice that the mountains look kinda distorted? Thats because of the side looking nature of SAR. 
With SAR we are not looking straigh down (orthagonal) on the landscape, we are looking from the side. 
This causes **Layover** and **Foreshortening** effects. My goal is to use the DEM to correct the these effects.

![Uncorreced Image](figures/fig5_scene.png)


## Refresher in Radar Geometrie

All comes down to solving doppler Equation
`(S-P) * V = 0`

SAR does not have the classical X,Y coordinates, we are working with azimuth and range¹. 

¹For a flat terrain there are actually two ranges:
- Slant Range: the straight line from object to radar-antenna (what actually gets measured)
- Ground Range: the horizontal distance fom the nadir

Azimuth really is a time not a distance. In a naive sense its the time the doppler equation equals zero.
Range is the distance from the antenna to the target, obtained from the delay of the echo. 

**In our little space ship we don't actually record a targets azimuth time and slant range, we record the echoes. The radar transmits a pulse and after each one samples the returning energy.**   
**Fast time (within a pulse): gives the delay which lets us calculate the slant range**  
**Slow time (from pulse to pulse): gives the Doppler history so we can calculate the azimuth**

## Parsing the XML

A SAFE folder is mostly XML. The measurement tiff holds the numbers, the annotation holds everything that gives those numbers a position.

```
S1D_..._COG.SAFE/
  annotation/
    s1d-iw-grd-vv-....xml             geometry: orbit, grid, polynomials
    calibration/calibration-...xml    the DN to sigma0 lookup
  measurement/s1d-iw-grd-vv-....tiff  the pixels, with no georeferencing at all
```

Four blocks carry everything I need:

- `imageInformation`: `productFirstLineUtcTime` (the time of line 0), `azimuthTimeInterval`, `numberOfLines`, `numberOfSamples`
- `orbitList`: 17 state vectors, position and velocity in Earth Centered Earth Fixed, 10 seconds apart. 
- `coordinateConversionList`: 28 polynomial sets mapping slant range to ground range, one per second of azimuth time
- `geolocationGridPointList`: 210 points with latitude, longitude, height, line, pixel and incidence. ESA ships them for calibration. I use them as a test suite.

Then the trap that cost me an hour: **all timestamps are UTC, but written without a timezone.** `datetime.fromisoformat()` returns a naive datetime, and `.timestamp()` then reads it as local time. Nothing crashes. The entire scene slides by your UTC offset, and because every timestamp slides together the result still looks perfect on your own machine. It breaks the day somebody in another timezone builds the same cache. I now attach UTC explicitly, and the shipped `meta/` was rebuilt after fixing it.

The reward for parsing carefully is that the product validates itself. Every grid point carries a `slantRangeTime`. Multiply by c/2 and you have a measured range. Interpolate the orbit to the same instant and you have a geometric range. They must agree:

```python
r_meas = 299_792_458.0 * tau / 2.0        # from the echo delay
r_geom = np.linalg.norm(S(t_az) - P)      # from the orbit and the ellipsoid
print(np.abs(r_geom - r_meas).max())      # 0.000219 m
```

0.2 millimetres, over 210 points and 800 km of range. That number told me the parsing was correct, and that every bug after it would be mine and not the file's.
All of it goes into a small cache, `meta/`: one JSON with the orbit and the polynomials, one `.npy` for the grid, one for the times. 60 KB, and the tests stop needing 1.2 GB of tiff.


## Unit Conversion Time, what fun

We start with `ellipsoid.py`, because everything later needs the same thing: a ground point expressed as a vector in the frame the orbit already uses.

The `orbitList` from the annotation is already in ECEF, Earth Centred Earth Fixed, in metres. Our DEM is not. It speaks UTM easting and northing plus a height, so it has to be converted before anything can be compared: UTM → lon/lat → ECEF.

$$a = 6\,378\,137\ \text{m}, \qquad b = 6\,356\,752.314\ \text{m}, \qquad e^{2} = 1 - \frac{b^{2}}{a^{2}}$$


The radius of curvature in the prime vertical, which is a length of about 6.4 million metres. 

$$N(\varphi) = \frac{a}{\sqrt{1 - e^{2}\sin^{2}\varphi}}$$

Then the point itself:

$$x = (N + h)\cos\varphi\cos\lambda$$

$$y = (N + h)\cos\varphi\sin\lambda$$

$$z = \big(N(1 - e^{2}) + h\big)\sin\varphi$$

with latitude $\varphi$, longitude $\lambda$, and $h$ the **ellipsoidal** height, all in metres.

The $N$ in $x$ and $y$, but $N(1-e^{2})$ in $z$: Geodetic latitude is defined by the surface normal, and on a squashed ellipsoid that normal does not pass through the centre. Get it wrong and the point lands consistently wrong, which is the worst kind of geodesy bug, because nothing crashes and the picture still looks fine.

In code that is one call. The algebra above is the entire function.

```python
P = lla2ecef(lat, lon, h)     # (N, 3) ECEF, metres
```

One caveat, the 210 geolocation points carry ellipsoidal heights. 
The Copernicus DEM carries orthometric heights, roughly 47 m higher than the 
ellipsoid over Austria. Feed those in unchanged and every point lands 47 m low, 
which shifts it about 67 m horizontally at 35° incidence. 
That is out of scope for this project, and it explains why a product can be geometrically correct 
and still sit a couple of pixels off.








