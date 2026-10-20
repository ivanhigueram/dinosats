# Vantor (Maxar) imagery for Siaya: processed data

This covers the nine Vantor Level 1B strips that NASA CSDA delivered for the
Egger et al. (2022) Siaya households: what was done to them, where the outputs
are, and which strips are before or after treatment. The code is in
`~/projects/remote_outcomes`. The longer write-ups are in that repo:
`docs/data/csda-maxar.md` (why these strips), `docs/data/vantor-l1b.md`
(processing) and `docs/experiments/embedding-change.md` (analysis).

The imagery is licensed under the CSDA agreement. Don't publish it, or
anything rendered from it, outside the project.

## Terms

- **Strip.** One satellite pass, identified by its catalog ID. Strips overlap,
  so one household can appear in two or three strips within an epoch.
- **Part.** A piece of a strip as delivered. A strip has 4 to 13 parts, and
  only the parts that cover a household were processed.
- **Epoch.** `pre` or `endline`, set by the acquisition window the strip was
  ordered for.
- **Copy.** One household as seen in one strip.
- **Chosen copy.** The single copy per household and epoch used in analysis.
- **Stratum.** The pair of strips a household's two chosen copies come from,
  written `<pre catalog id>_<endline catalog id>`.

## Treatment timing

The order used windows taken from the survey dates in `households.parquet`.

| Event | Date |
|---|---|
| Census | from 2014-08-18 |
| Transfers start | 2014-09-01 |
| Endline interviews | 2016-05-30 to 2017-05-25 |

- **Pre** window: 2013-01-01 to 2014-08-17. All three pre strips are from
  2013-03-02, about 18 months before the first transfer. Every pre copy is
  pre-treatment.
- **Endline** window: 2016-05-30 to 2017-05-25. All six endline strips fall in
  it (2016-06-03 to 2017-01-24), so every endline copy is post-treatment and
  overlaps the endline survey.

No scene from 2014 or early 2015 covers a household, so there is no image at
the baseline survey.

This is cleaner than the Esri Wayback imagery used before. There, about a third
of the `endline` epoch is chips dated 2010 to 2013, which have to be filtered
out with `endline_is_post`. That filter doesn't apply to the Vantor data, so
run `embedding_change.py` with `--no-wayback-filter`.

## The strips

| Epoch | Date | Sensor | Catalog ID | Pan GSD (m) | Off-nadir | Parts processed | Clip km² | Copies | Chosen | Final sample |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| pre | 2013-03-02 | WV02 | `1030010020286200` | 0.55 | 24.3° | 4 | 536 | 40,464 | 34,766 | 32,937 |
| pre | 2013-03-02 | WV02 | `1030010020944E00` | 0.53 | 22.3° | 4 | 300 | 22,318 | 22,204 | 19,333 |
| pre | 2013-03-02 | WV02 | `103001001F4B3700` | 0.58 | 27.6° | 4 | 168 | 13,470 | 8,410 | 8,304 |
| endline | 2016-06-03 | WV03 | `104001001DC23800` | 0.35 | 20.3° | 1 | 35 | 2,490 | 2,268 | 2,257 |
| endline | 2016-08-21 | GE01 | `1050010005D53300` | 0.45 | 17.7° | 3 | 73 | 6,135 | 2,846 | 2,740 |
| endline | 2016-08-29 | GE01 | `1050010006160300` | 0.42 | 10.6° | 4 | 344 | 27,125 | 26,883 | 26,711 |
| endline | 2016-09-11 | WV03 | `104001002267FA00` | 0.37 | 26.6° | 4 | 402 | 30,933 | 12,223 | 10,102 |
| endline | 2016-09-11 | WV03 | `104001002246F500` | 0.33 | 14.9° | 4 | 217 | 16,583 | 13,652 | 11,499 |
| endline | 2017-01-24 | WV02 | `1030010062CF2100` | 0.50 | 17.4° | 2 | 184 | 16,046 | 7,265 | 7,265 |

- **Copies:** households with a valid pixel in the strip.
- **Chosen:** households whose chosen copy for that epoch comes from the strip.
- **Final sample:** households in the change analysis. They need a chosen copy
  in both epochs, and the endline copy must be clear of cloud. Each epoch
  column sums to 60,574.

## Processing

`download/vantor_l1b.py` turns each L1B part into orthorectified, pansharpened
top-of-atmosphere reflectance at 0.5 m. It follows the Polar Geospatial
Center's GDAL recipe, reimplemented rather than copied because their code is
GPL.

1. **DEM.** Copernicus GLO-30, converted from EGM2008 to ellipsoid heights.
   The geoid shift was −16.3 to −16.9 m in every part. Without the EGM2008 grid,
   GDAL skips this shift silently, so the script stops if the shift is under
   5 m.
2. **Orthorectify** Pan and MUL with the RPCs and the DEM (`gdalwarp -rpc`,
   cubic), clipped to 200 m around each household. All strips share one UTM 36N
   grid (EPSG:32636) snapped to 2 m, so they line up pixel for pixel.
3. **Calibrate** DN to TOA reflectance using Maxar's 2018v0 absolute
   calibration, Thuillier 2003 solar irradiance, and the sun angle and time
   from each part's IMD.
4. **Pansharpen** with GDAL's weighted Brovey, using PGC's band weights for
   WV02 and WV03 and equal weights for GE01.
5. **Write** a COG with bands B, G, R, NIR, plus an 8-bit RGB VRT on top of it.

The steps don't include atmospheric correction or co-registration. Pixels sit
wherever the RPCs and GLO-30 put them. By eye on one part, the offset between
strips is within a few metres, but it hasn't been measured.

The run was on 2026-10-17: 30 parts, 2,260 km² clipped, 42 GB, 88 minutes on
20 cores.

### Picking one copy per household

`download/vantor_coverage.py` lists every copy and picks one per household and
epoch. Clear copies come first, then copies with a full 134 m chip, then the
lowest off-nadir angle. A copy is clear if it is more than 100 m from Vantor's
cloud polygons. The polygons don't cover cloud shadow. The pick never crosses
epochs, so a pre chip is never filled from an endline strip.

| Epoch | Households with a copy | Clear | Inside a cloud polygon | Copies per household |
|---|---:|---:|---:|---:|
| pre | 65,380 | 65,380 | 0 | 1.17 |
| endline | 65,137 | 60,577 | 3,427 | 1.52 |

The census has 65,383 households. 3,310 endline households are under cloud in
every strip that covers them. They are flagged, not filled.

### Embeddings

These are DINOv3 ViT-L/16 features (SAT-493M checkpoint, frozen) from the RGB
VRT, so NIR is dropped. Each chip is 100 m across, resized to 208 px (about
0.5 m per pixel, the native resolution). There is one run per strip, covering
the households whose chosen copy that strip holds, so each household is
embedded once per epoch.

## Layout

All paths are under `/mnt/drive/`. None of it is in git.

```
vantor_nasa_data/                       294 GB  raw L1B delivery, as received
  <catalog_id>/                                 one directory per strip
    <order_id>-ord-request.json                 the original order (system-ready, 2026-10-02)
    <order_id>-ord_manifest.json
    <product_id>/<product_id>_P0NN_{PAN,MUL}/   parts: NTF + RPB + IMD + XML

remote_outcomes/
  siaya_hh_data/households.parquet              65,383 census households (point, village, treat, eligible)

  csda/
    vantor_request.csv                          the nine strips ordered: window, date, sensor, GSD, off-nadir, cloud
    vantor_browse/<catalog_id>.cloud.json       Vantor cloud polygons (no shadow)
    reorder/<catalog_id>.json                   draft re-order requests (map-ready, ACOMP, 16-bit, DRA off)

  vantor/
    ard/                                42 GB   processed imagery
      <part>_toa.tif                            COG, UInt16, B G R NIR, reflectance x 10,000, nodata 0
      <part>_rgb.vrt                            8-bit RGB: 1 + 254 * reflectance / 0.3, clipped
      strip_<catalog_id>_toa.vrt                mosaic of a strip's parts (reflectance)
      strip_<catalog_id>_rgb.vrt                mosaic of a strip's parts (8-bit RGB, what DINOv3 reads)
      parts.csv                                 per part: catalog ID, sensor, time, clip km2, geoid shift, band means
    ard_run.log
    coverage.parquet                            175,564 rows, one per household x strip copy
    embed_ids/<epoch>_<catalog_id>.parquet      hhid_key + stratum: who is embedded from each strip
    compare/                                    side-by-side review pages (served on localhost:8787)

  embeddings/vantor/                    1.2 GB
    <epoch>_<catalog_id>.parquet                one row per household
    <epoch>_<catalog_id>.yaml                   the full Hydra config of that run
    run.log

  probes/embedding_change_vantor/               change-analysis results on the Vantor embeddings
    village_probe.csv  components.csv  dose_response.csv  ate.csv  households_center.parquet
  probes/embedding_change/                      the same on Wayback, for comparison
```

`<part>` is the Pan filename without `-P1BS-`, for example
`13MAR02082157-200014472558_01_P002`. The part prefix gives its date
(`13MAR02` is 2013-03-02).

### Column reference

`vantor/coverage.parquet`

| Column | Meaning |
|---|---|
| `hhid_key` | household ID, joins to `households.parquet` |
| `epoch` | `pre` or `endline` |
| `catalog_id`, `date`, `collection`, `off_nadir`, `gsd_pan` | the strip |
| `cloud_distance_m` | distance to the nearest cloud polygon. 0 inside one, `inf` if the strip has none |
| `full` | all four corners of a 134 m chip are valid |
| `clear` | `cloud_distance_m > 100` |
| `chosen` | the copy used for this household and epoch |
| `raster` | path to the strip's `_toa.vrt`. Swap the suffix to `_rgb.vrt` for 8-bit |

`embeddings/vantor/<epoch>_<catalog_id>.parquet`

| Column | Meaning |
|---|---|
| `hhid_key` | household ID |
| `cls` | CLS token, 1024-d, float16 |
| `patch` | mean of all patch tokens over the 100 m chip |
| `center` | mean of patch tokens inside a 20 m disc on the household point |
| `center_r40`, `center_r60` | the same for 40 m and 60 m discs |
| `raster` | the strip VRT it was read from |
| `stratum` | `<pre catalog id>_<endline catalog id>` |

To get a household's change vector, read all nine files, concatenate within
each epoch, and join pre to endline on `hhid_key`. Each household is in
exactly one pre file and one endline file. `analysis/embedding_change.py`
raises if a household repeats.

## Strata

There are ten strip pairs. Treatment share varies by pair, so any regression
needs a stratum fixed effect.

| Stratum (pre_endline) | Households | Treated village share | Eligible share |
|---|---:|---:|---:|
| `1030010020286200_1050010006160300` | 21,147 | 0.53 | 0.31 |
| `1030010020944E00_104001002246F500` | 11,499 | 0.55 | 0.32 |
| `1030010020286200_104001002267FA00` | 8,840 | 0.43 | 0.32 |
| `103001001F4B3700_1050010006160300` | 5,564 | 0.50 | 0.35 |
| `1030010020944E00_1030010062CF2100` | 5,058 | 0.44 | 0.35 |
| `103001001F4B3700_1050010005D53300` | 2,740 | 0.62 | 0.41 |
| `1030010020286200_1030010062CF2100` | 2,207 | 0.42 | 0.41 |
| `1030010020944E00_104001001DC23800` | 1,514 | 0.43 | 0.41 |
| `1030010020944E00_104001002267FA00` | 1,262 | 0.39 | 0.32 |
| `1030010020286200_104001001DC23800` | 743 | 0.60 | 0.47 |

## Caveats

- **Strips differ in brightness on the same ground.** On clear households
  seen by two strips, two WV02 strips from the same day differ by +0.085 in
  red (0 to 1 RGB scale). WV02 `…CF2100` (January 2017) is 0.171 darker in red
  than WV03 `…67FA00`. GE01 matches WV03 within 0.016, so the variation follows
  the strip rather than the sensor. TOA calibration keeps haze and view-angle
  effects, which is why the analysis demeans within stratum.
- **Endline dates span seven months**, June 2016 to January 2017, so endline
  strips also differ by season. That is absorbed by the stratum effect too.
- **Iron roofs saturate** in the 8-bit RGB at reflectance 0.3. The
  `_toa.tif` keeps the full range.
- **About 7 pixels per million** have one band at 0, from cubic undershoot
  next to bright roofs. GDAL treats that band as nodata there. Three
  households lose their only pre copy this way.
- **Cloud shadow isn't masked.** At 65° sun elevation it falls roughly 0.5 to
  1 km from the cloud, so some "clear" endline copies are shaded.
- **A re-order is drafted but not placed.** It asks Vantor for map-ready
  orthorectification with ACOMP atmospheric correction, which should reduce the
  strip-to-strip differences. The requests are in `csda/reorder/`.
