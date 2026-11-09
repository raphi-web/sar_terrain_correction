# Blog outline — "Terrain correction from scratch"

Goal: portfolio piece + LinkedIn reach. Audience: geospatial devs who know rasters
but not radar geometry; SAR-curious scientists who use SNAP as a black box.

Angle (the differentiator): **not "I implemented it" but "I verified it."** The
hook is that a from-scratch implementation reproduces ESA's own 210-point
geolocation grid to 0.2 mm in range and 0.19 px in azimuth. Everything else
serves that claim.

## Series plan (two posts)

**Part 1 — Geometry (ready after L5).** "The radar image has no coordinates."
**Part 2 — Radiometry (after L6).** "Why the same forest looks bright on one
slope and dark on the other." → second LinkedIn post, seconds the first.

## Part 1 structure (~1800–2200 words)

1. **Hook (150 w).** The COG GeoTIFF opens in QGIS with `CRS: None`. No
   geotransform. A radar image that doesn't know where it is. Everything needed
   to place it lives in the annotation XML.
2. **The five-sentence physics (200 w).** Slant range from round-trip time
   (`r = c·τ/2`); azimuth from Doppler; GRD stretches columns by ~1/sin(i) on a
   *reference* surface; mountains break that assumption → foreshortening,
   layover, shadow. (Fig: fig5 scene — the ridges are already visible.)
3. **The pipeline (600 w).** One paragraph + one figure per step:
   - Orbit: 17 state vectors → cubic Hermite (fig1). Why linear sags 102 m (fig2).
   - Ellipsoid: geodetic → ECEF, and the `N(φ)` warm-up.
   - SRGR: the 28 polynomial sets and the **name inversion** (fig3 shows the
     slope = sin i).
   - Range-Doppler: the zero-Doppler root (fig4), monotone through one crossing.
   - Ortho: map pixel → (lat, lon, h) → (line, pixel) → bilinear sample.
4. **Verification (300 w).** The 210-point grid as free ground truth; the table
   of residuals (0.2 mm range, 0.19 px azimuth, 0.015 px range pixel); why the
   0.19 px is the *grid's* quantization, not the solver's. (Fig6.)
5. **The traps that cost hours (500 w).** The shareable part — see list below.
6. **What this does and doesn't do (200 w).** Honest: geometry only; no
   radiometric normalization yet; shadow = 0 % in this scene because 34°
   incidence puts the satellite 56° up and the DEM peaks at 52°; no speckle
   filtering; Numba still pending.
7. **Code + reproduce (150 w).** Repo link, the `meta/` cache approach, the
   test harness as the teaching device. Data: the public S1D COG product +
   Copernicus DEM.
8. **CTA (100 w).** Questions; what readers want next (Numba? radiometry?
   SNAP comparison?).

## Ready-made numbers (copy into the post)

| Quantity | Value | How |
|---|---|---|
| slant range vs grid | **0.2 mm** max | Hermite orbit + `c·τ/2` |
| azimuth line | **0.191 px** max | zero-Doppler solve |
| range pixel | **0.0153 px** max | SRGR inversion |
| SRGR forward | **0.0000 m** | polynomial exactness |
| AOI runtime | 27 s / 1 MP | vectorized numpy |
| scene | S1D IW GRD VV, 2026-10-31, asc | COG product |
| DEM | 26 m, UTM 33N | Copernicus |
| world time, 1 MP | 16.4 s (numpy) → 0.62 s (Numba) | 26.5× speedup |
| full scene, 442 MP | ~2 h → **~4.6 min** | extrapolated |

## Part 3 (optional, short) — "make it fast without making it wrong"

The hook writes itself: *the same code, 100 times faster, and the output is
bit-identical.* Points worth making (they are the difference between a
benchmark and an engineering lesson):

- The numpy path stays as the **oracle**. Tests assert the kernel matches it to
  1e-10 px. Speed without a reference is just plausible output, faster.
- `fastmath=False` on purpose: reassociation changes the last bits. Speed came
  from the loop, not from the algebra.
- `@njit` cannot see dicts or strings: pre-pack the 28 SRGR sets into `(28,9)`
  and `(28,)` arrays and pass them in.
- `prange` over pixels, `cache=True`: 12 threads, compile once.
- The result: 102× on the kernel, 26.5× end-to-end. The gap between those two
  numbers is rasterio I/O — and *that* is the next bottleneck, not the physics.
- One trap: the first call costs ~2.4 s of compilation. Warm it before timing,
  or your "benchmark" measures LLVM.

## The trap list (sell this section hard — it's what people remember)

1. **The COG XML has no namespaces** — `endswith('}tag')` parsers silently fail.
2. **The tiff has no georeferencing** — the geography lives in the annotation.
3. **`srgr` vs `grsr` are (variable, function)**, not (function, variable): the
   leading slopes (1.977 vs 0.506) are the proof. Swap them and you're hundreds
   of km off.
4. **The 28 SRGR sets are fits, not physical constants** — interpolating
   coefficients between them gives 3 px error; nearest-set gives 0.015 px.
5. **A helper module named `io` shadows the stdlib** — `import io` gets Python's
   file-stream module and an `AttributeError` at 2 a.m.
6. **`np.dot((N,3), (N,3))` is a matrix product** — you want `np.sum((S-P)*V, axis=-1)`.
7. **`np.maximum(-inf, NaN) = NaN`** — one bilinear step off the edge poisoned
   every mask until I switched to `np.fmax`.
8. **Layover has a direction** — a nearer point at larger range. Put the test
   spike on the wrong side and you "prove" the mask is broken.

## LinkedIn post template (Part 1)

> A Sentinel-1 image usually arrives with coordinates baked in.
> This one had **none** — `CRS: None`, no geotransform, just pixels.
>
> So I rebuilt terrain correction from scratch in Python: orbit interpolation,
> the range-Doppler equations, slant↔ground conversion, DEM resampling.
>
> The check that mattered: ESA ships a 210-point geolocation grid inside the
> product. My pipeline reproduces it to **0.2 mm in range and 0.19 px in
> azimuth**.
>
> Three things that cost me the most time:
> • the annotation calls the two conversion polynomials `srgr`/`grsr` — and they
>   mean the opposite of what you'd guess
> • the 28 polynomials are per-second *fits*: interpolate them and you're 3 px
>   off; use the nearest and you're at 0.015 px
> • the measurement grid is your free ground truth — use it as a test, not a
>   sanity check
>
> Full write-up + code: [link]
>
> #SAR #Sentinel1 #RemoteSensing #Geospatial #EarthObservation #Python

## Publishing checklist

- [ ] Personal blog with a canonical URL (Hugo is fine — you already run Hugo
      sites), then link from LinkedIn rather than posting a LinkedIn article
      (better SEO, your own domain).
- [ ] Figures: export the four best (fig1, fig3, fig4, fig7) at 2× resolution.
- [ ] Code: MIT license, strip nothing — the test files are the pedagogy.
- [ ] Citation: name the product (`S1D_IW_GRDH_...`) and the DEM source.
- [ ] Credit SNAP as the reference implementation, without copying its code
      (GPL — reading is fine, pasting is not).
- [ ] Ask one question at the end — comments are the reach multiplier.
- [ ] Part 2 teaser: "next: why brightness lies on slopes."