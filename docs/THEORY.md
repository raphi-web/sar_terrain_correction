# Terrain correction, why and how (L0)

## 1. What SAR actually measures
SAR is a **ranging** sensor. One measurement = two things:
- **Slant range** from round-trip time: `r = c·τ/2`, c = 299 792 458 m/s.
  This scene: τ_near = 5.3349222738 ms → r = **799 684.7 m**, satellite at ~691 km altitude above the ellipsoid (a = 6 378 137 m, b = 6 356 752.314245 m).
- **Azimuth** via Doppler history along track (f_az ≈ 486.5 Hz, PRF ≈ 1717 Hz). A GRD line maps to one azimuth time: `t(line) = t0 + line·Δt`, Δt = 1.4990659 ms.

So the "natural" image coordinate system is **(azimuth time, slant range)** — a curved,
25°–46° off-nadir cylinder around the orbit, NOT a map. Every pixel is the intersection of
a range sphere `|S(t) − P| = r` with the zero-Doppler plane (the plane through `S(t)` with
normal `S(t)×V(t)` — i.e., `P` coplanar with satellite position and velocity).

## 2. What the GRD product did to geometry
The processor applied **SRGR** (Slant Range to Ground Range): it stretched each column so that
range is expressed as ground range on a *smooth reference surface* (their coarse DEM: 115–1997 m
for this scene). That's why the range-stretch slope ≈ `1/sin(incidence)` (near range: 1/sin 30.3° = 1.98).
GRD is great for flat land. It is **wrong** wherever the real ground deviates from that surface —
which is the entire Alps. Hence: we re-derive slant range from the metadata polynomials and redo
the geometry with the real DEM. The geometry lives in the annotation, not the tiff — this COG has
no geotransform at all, hehe~

## 3. The three terrain artifacts (all at once, no filter)
For a slope of local incidence θ_loc vs flat-terrain incidence θ_0:
- **Foreshortening**: facing slopes are squeezed — the whole slope projects into Δr·sin spacing.
  Steep facing slopes (θ_loc → 90°… well, toward grazing the other way) merge into one bright line.
- **Layover**: the *top* of a steep facing slope is nearer to the sensor than its foot → it is
  imaged first. Radar builds "inside-out" cliffs: roofs over write street. In this scene (inc 30–46°,
  slopes up to ~45° in the Dachstein/Tauern area) expect severe layover on range-facing ridges.
- **Shadow**: slopes tilted away beyond grazing → no backscatter returns, only speckle-level darkness.
  Shadow is *informative* (it measures slope directly) but radiometrically empty.

## 4. Radiometry inherits the mess
DN → σ0 needs the illuminated ground area per pixel. On slopes, GRD-area assumptions are wrong:
facing slopes get over-bright (same energy, "smaller" assumed area), shadowed get garbage.
Fix = terrain normalization: divide by the actual illuminated area factor (γ0 / "flat earth").
We'll do geometry first (L1–L4), masks next (L5), radiometry last (L6).

## 5. The pipeline we are building
```
meta/ (cached XML) ──► orbit s(t) [L1] ──► range-Doppler solver [L3]
DEM (lon,lat,h) ─────►                 (t, r) per map pixel ─► (line, pixel) via SRGR⁻¹ [L2]
                                                        └► bilinear sample GRD DN [L4]
Layover/shadow from local geometry [L5] · σ0→γ0 [L6]
```
Ground truth we own for free: the 210-point geolocation grid (satisfies every equation above
to 0.2 mm — verified). Our implementation must hit it independently.

### Symbols (our conventions)
`S(t), V(t)` ECEF satellite m, m/s · `P` ECEF target · `r = |S−P|` slant range ·
`τ` two-way time · `i` incidence angle (between LOS `P−S` and local ellipsoid normal, or terrain normal) ·
`e2 = 1−(b/a)² ≈ 6.7395e−3`, `N(φ) = a/√(1−e2 sin²φ)` (radius of curvature in the prime vertical).
