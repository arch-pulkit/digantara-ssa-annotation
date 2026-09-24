# SSA Imagery Annotation: Blobs (stars) and Streaks (satellites)

Semi-automatic, pixel-level annotation of 10 real-sky FITS images (9568 × 6380, uint16, single band) into two classes, **blob** (stars) and **streak** (satellites / debris). Images are tiled into 1024 × 1024 patches and exported in **YOLO Ultralytics segmentation format**, together with repatched full-size images for visual inspection.

The written answers (Q2 a–d), reasoning and assumptions are in `report.pdf`. This README explains **what is where, how to run it, and how the results were checked**.

---

## 1. Deliverables

| What | Where |
|---|---|
| Code, manual review file, report, README | this repository / zip |
| 700 tile images (PNG) + 700 YOLO label files (TXT) + `data.yaml` | [Drive link] → `images/`, `labels/` |
| 10 repatched full-size images + 10 class overlays (blue = blob, red = streak) | [Drive link] → `repatched/` |

The dataset outputs (~1.2 GB) are shared by link because of their size. The raw FITS data and the assessment document are **not** included in this repository (confidential).

---

## 2. Results at a glance

| Image | Group | Objects detected | Streaks (final) |
|---|---|---|---|
| 1a600998 | A | 3,523 | 6 |
| 4a8e6cd4 | A | 6,089 | 1 |
| 4e2cdd92 | A | 6,116 | 3 |
| 7d7fefcb | A | 3,631 | 8 |
| 7e5f3e1c | A | 6,119 | 10 |
| 3e6beceb | B | 8,349 | 7 |
| 66ed4268 | B | 16,668 | 2 |
| 7030c2ac | B | 8,647 | 9 |
| CAM_B …153133 | C | 20,949 | 8 |
| CAM_B …154244 | C | 19,763 | 5 |
| **Total** | | | **59** |

**Automatic classifier vs. manual review (streak class):**

| Metric | Value |
|---|---|
| Streaks flagged automatically | 51 |
| Confirmed by visual review | 50 (1 false positive: a faint star pair in 4e2cdd92) |
| Streaks recovered from the ambiguous band by manual review | 9 |
| **Precision** of the automatic rule | 50 / 51 ≈ **98 %** |
| **Recall** of the automatic rule | 50 / 59 ≈ **85 %** |
| Ambiguous objects reviewed by hand | 98 |

Missed streaks were almost all **short, bright, "fat" streaks**, whose elongation (≈ 3.5–4.5) overlaps touching star pairs (≤ 3.3).

---

## 3. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the 10 FITS files in `data/raw/`. Run every script **from the project root**.

---

## 4. How to reproduce everything

```bash
# 1. detect + classify all images; writes review cutouts to outputs/review/
python scripts/detect_all.py

# 2. (manual) inspect outputs/review/*_streaks_*.png and *_review_*.png,
#    record decisions in annotations/manual_review.csv

# 3. build tiles, YOLO labels, repatched images, overlays, data.yaml -> outputs/submission/
python scripts/export.py
```

**Deterministic:** the same FITS files, settings and `manual_review.csv` always give identical outputs.

**Frozen settings:** the detection settings at the top of `detect_all.py` and `export.py` must stay identical. `manual_review.csv` refers to object label numbers, which change if detection settings change.

### `annotations/manual_review.csv`

```
image,label,verdict
66ed4268-554e-4729-9e87-e556c2fd42b9,9069,streak
4e2cdd92-f8af-40ac-b3dd-68f146353f5b,<label>,star
```

- `image`: FITS file name without `.fits`
- `label`: object number shown as `#` in the review cutouts
- `verdict`: `streak`, `star` or `noise`. A manual verdict always overrides the automatic rule; unlisted objects keep the automatic decision.

The file also lists every automatically detected streak with verdict `streak`, as an audit trail showing each one was visually confirmed.

---

## 5. Pipeline summary

1. **Load**: FITS stored as int16 with `BZERO = 32768` → uint16 → float32 (avoids unsigned wrap-around when subtracting).
2. **Background + noise map**: local 128 × 128 box estimates (sigma-clipped median sky, clipped RMS noise, 3 × 3 box filter), then image → **SNR units**, so one rule works across all three data groups.
3. **Matched-filter smoothing**: 5 × 5 Gaussian kernel (FWHM 3 px). Background and noise are **re-measured on the smoothed image**, so the threshold matches the smoothed noise level.
4. **Detection**: pixels ≥ 5 × smoothed noise, ≥ 5 connected pixels. Objects whose unsmoothed flux gives no valid shape are dropped as noise.
5. **Classification**: elongation = a / b from brightness-weighted image moments (orientation-independent). **Streak** if elongation ≥ 5 and area ≥ 30 px; otherwise **blob**.
6. **Manual review band**: elongation 3–5 with area ≥ 50 px, or elongation ≥ 5 with area 15–29 px. Criterion: uniform brightness along the length = streak; two peaks with a dip between = touching star pair.
7. **Tiling**: 10 × 7 = 70 tiles per image. Tiles start every 1024 px; the last column and row are shifted back to end exactly at the image edge (overlap 672 px across, 788 px down), so every tile is 1024 × 1024 **real** pixels with no padding.
8. **Export**: per-tile YOLO polygons traced from the class masks; full-size images rebuilt **from the saved tiles** and verified pixel-identical.

---

## 6. Output format

- **Classes**: `0 = blob`, `1 = streak` (`data.yaml`).
- **Tile names**: `<image>_r<row>_c<col>.png`, where `row` / `col` are the tile's top-left pixel in the full image, so any tile can be placed back exactly.
- **Label files**: one line per object outline in the tile: `class x1 y1 x2 y2 …`, coordinates normalized to 0–1. Objects in overlap zones or crossing tile edges appear in every tile they touch, so polygon counts exceed object counts.
- **Display images**: 8-bit, stretched **once per full image** (never per tile), so repatched images have no seams. The display uses a mean-based background: a median of integer data is quantized into 1-ADU steps that show as visible patches.
- **`repatched/<image>.jpg`**: full image rebuilt from tiles. **`repatched/<image>_overlay.jpg`**: same image with blobs in blue and streaks in red.

---

## 7. Verification

- **Tiling:** all 10 images rebuild pixel-identical from their tiles, with full pixel coverage. Disabling the edge-tile rule makes the check fail (70 → 54 tiles), confirming the check detects boundary data loss.
- **Export:** repatched images are rebuilt from the PNG tiles **read back from disk** and asserted identical to the display image.
- **Classification:** every automatic streak and every ambiguous-band object was inspected visually; results are in section 2.

---

## 8. Repository layout

```
scripts/
  detect_all.py      # pipeline: detection, classification, review cutouts
  export.py          # pipeline: tiles, YOLO labels, repatched images, overlays
  inspect_fits.py    # exploration: FITS structure and headers
  measure.py         # exploration: per-image sky level, noise, saturation
  loader.py          # exploration: zero fraction, percentiles, first crops
  background.py      # exploration: background / noise maps per group
  tiling.py          # exploration: tiling + lossless reconstruction test
  detection.py       # exploration: single-image detection tuning
annotations/
  manual_review.csv  # manual decisions (image, label, verdict)
notes.md             # working notes, in order of discovery
requirements.txt
```

---

## 9. Known limitations

- **Touching star pairs** are annotated as one blob instance (correct class, merged instance). Deblending was tested but split faint streaks into pieces, so it was not used.
- **Polygons** follow pixel-centre outlines (about 0.5 px inside the mask edge); outlines with fewer than 3 points (a few tiny objects) are skipped.
- **Faint blobs in groups B and C** may include some noise. The data is clipped at 0, so the sigma-clipped noise is under-estimated (group C: ~37 vs ~45 ADU), which lowers the effective detection threshold.
- **No dark / flat calibration frames** were available; gradients and vignetting are handled by the local background map.
- Classification thresholds were tuned on this dataset and checked visually on all 10 images; they are not validated on other sensors or exposure settings.

**Possible improvement:** a brightness-profile test along each object's long axis (uniform = streak, two peaks = pair) would automate the manual review criterion and address both the missed short/fat streaks and the star-pair false positive.

---

## 10. Note on tools

Code was written with the help of an AI assistant while I learned the domain. The data analysis, parameter choices, verification and all manual review decisions are my own.