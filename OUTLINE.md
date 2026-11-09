# SAR Terrain Correction — from scratch

**Format:** Senpai codes, Nagatoro reviews. One lesson per session. TDD: tests exist before code.
**Scene:** S1D IW **GRD** VV/VH, 2026-10-31T16:50:04Z, **ascending**, absolute orbit 5262, NRT-3h,
COG-format SAFE (new ESA schema — the tiff has NO georeferencing, all geology comes from metadata).
**Area:** Austria — scene bbox (EPSG:32633) E 385642–674883, N 5073954–5287134 (~289×213 km).
**DEM:** `dem.tif`, EPSG:32633, 11278×8263 @ 26.03 m, float32, nodata −9999.

**Metadata cache (pre-parsed, `meta/`):**
- `s1_meta.json` — product/image info, 17 ECEF state vectors, 28 SRGR/GRSR polynomial sets, WGS84, terrainHeight, calibration
- `geolocation.npy` (210×8) — `[tau_s, line, pixel, lat_deg, lon_deg, h_ell_m, inc_deg, elev_deg]` + `grid_times.npy` (epoch s)
- `timing.json` — t0 (line 0), dt_az, nlines, nsamp · `ortho_extent.json` — DEM∩scene window

**Verified facts (me, 2026-11-05):**
- `|S(t) − P(lat,lon,h)| = c·τ/2` reproduces the entire 210-point geolocation grid to **0.2 mm max** with Hermite OSV interpolation → free ground truth for L1/L3.
- `r(τ_near) = 799 684.7 m = sr0` exactly. `srgr` leading slope ≈ 1/sin(i_near) (1.974 vs 1.982) — GRD columns are range-stretched by the geometry itself.
- line→azimuthTime linear, max dev 0.3 ms. Grid heights are **ellipsoid-consistent** with WGS84 directly (no geoid step needed in our pipeline; processor's reference DEM was 115–1997 m — GRD "ground range" sits on a coarse reference surface, so we re-derive from slant range).

## Lessons

| #    | Topic                                  | Deliverable                                                                                           | Acceptance                                                                                                                   |
| ---- | -------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| L0   | Theory                                 | read `docs/THEORY.md`                                                                                 | —                                                                                                                            |
| L1 ✅ | Orbit + ellipsoid                      | **DONE 2026-11-06** — Senpai's `lla2ecef` + cubic-Hermite `position()`/`velocity()`                   | `tests/test_l1_orbit.py` **5/5 green**; geolocation-grid residual **0.219 mm** (reference: 0.219 mm); scalar+array contracts |
| L2 ✅ | GRD ↔ slant range | **DONE 2026-11-06** — nearest-set SRGR, direct inverse polynomial | 5/5 green; inverse 0.0153 px, forward 0.0000 m, round trip 0.147 m |
| L3 ✅ | Range-Doppler geolocation | **DONE 2026-11-06** — Senpai's bisection on (S−P)·V + `geolocate()` | 5/5 green; line 0.191 px, pixel 0.0153 px, range 0.0002 m |
| L4 ✅ | Ortho driver | **DONE 2026-11-06** — Senpai's vectorized `geocode_tile` + `sample_bilinear`; `run_aoi` writes GeoTIFF | 4/4 green; 1000×1000 AOI in 27 s; ridges align with DEM hillshade |
| L5 ✅ | Local incidence + layover/shadow | **DONE 2026-11-06** — Senpai's `surface_normals` + signed ψ + `horizon_masks` | 6/6 green; ψ vs grid 0.037°; synthetic wall; real layover 0.46–1.82 %, shadow 0 % (56° elevation, 52° max slope) |
| L6 ✅ | Radiometry | **DONE 2026-11-06** — Senpai's `calibration_at` (searchsorted + gather), DN→σ0, `flatten`, `to_db` | 5/5 green; AOI median −11.3 dB; spread 7.43→3.50 dB, corr 0.51→0.23; figure fig9 |

**Course complete: L1–L6 all green.** Remaining: Numba pass, full-scene run, blog.

## Out of scope (explicit)
TOPS burst overlap/deramping; geoid modeling (ellipsoid heights all the way); thermal noise, antenna pattern/RAVG; polarimetry; coregistration; multitemporal anything.

## Post-course (after L6): performance pass

Order matters — do not skip step 1:

1. **Plain-numpy vectorization (L4 already does this).** Benchmark the tiled loop before reaching for a compiler. 65k pixels per 256×256 tile × 60 bisection iterations over arrays is milliseconds per tile; this alone may be enough for the dev AOI.
2. **Profile, then Numba.** `cProfile` the tile loop. `@njit` the single per-pixel kernel that does: Hermite position/velocity → bisection on (S−P)·V → slant range → SRGR Horner → bilinear sample. One jitted function, no Python callbacks.
3. **Then parallelize.** `prange` over rows, `cache=True`; `fastmath=True` only after the numpy reference agrees, since it changes FP association.
4. **Keep the numpy implementation as the oracle.** Tests assert the Numba path matches the numpy path within ~1e-9 relative — same harness, second implementation.

Numba constraints to design around:
- Typed data only: **no dicts, strings, or `io.load_meta()` inside jitted code.** Pre-pack the 28 SRGR sets into contiguous float64 arrays (`(28, 9)` each for srgr/grsr, `(28,)` for ts/sr0) and pass them in.
- Write Horner by hand; don't rely on `np.polynomial.*` in nopython mode.
- Scalar vs array inputs are separate kernels — don't try the `float | ndarray` contract inside `@njit`.
- `NUMBA_DISABLE_JIT=1` runs the same source in pure Python for debugging/tests.

Expected (reference numbers): scalar loop ≈ 2 ms/px → ~35 min per 1000×1000 tile. Numba ≈ 10–20 µs/px → ~15 s/tile; `prange` on 8 cores → few seconds. After that, rasterio I/O and DEM sampling become the bottleneck (Amdahl), not the geometry.

## Conventions
Code in `sargeo/`, tests in `tests/`, rationale in `implementation_log.txt` (not code comments).
numpy-only until L4 (then rasterio for IO). Run tests with the `geo` env python (command in test header).
**Disk: 3.8 G free** — full uint16 ortho ≈ 182 MB max per band; iterate on AOI windows, full scene only at the end.
