from pathlib import Path
import csv
import numpy as np
import cv2
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.convolution import convolve
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.background import Background2D, MedianBackground, MeanBackground

# same settings as detect_all.py (never change these, or the review labels break)
detect_threshold = 5.0
min_pixels = 5
streak_min_elongation = 5.0
streak_min_area = 30
tile_size = 1024

out = Path("outputs/submission")
for folder in ["images", "labels", "repatched"]:
    (out / folder).mkdir(parents=True, exist_ok=True)
kernel = make_2dgaussian_kernel(3.0, size=5)


def measure_background(img):
    return Background2D(img, box_size=128, filter_size=3,
                        sigma_clip=SigmaClip(sigma=3.0),
                        bkg_estimator=MedianBackground())


def tile_starts(length):
    starts = list(range(0, length - tile_size + 1, tile_size))
    if starts[-1] + tile_size < length:
        starts.append(length - tile_size)
    return starts


# your manual decisions
manual = {}
with open("annotations/manual_review.csv") as f:
    for row in csv.DictReader(f, skipinitialspace=True):
        manual[(row["image"].strip(), int(row["label"]))] = row["verdict"].strip()

for path in sorted(Path("data/raw").glob("*.fits")):
    name = path.stem

    # STEP 1: detection (same as detect_all.py)
    image = fits.getdata(path).astype(np.float32)
    bg = measure_background(image)
    signal = image - bg.background
    snr = signal / bg.background_rms
    smoothed = convolve(signal, kernel)
    bg_smooth = measure_background(smoothed)
    smoothed_snr = (smoothed - bg_smooth.background) / bg_smooth.background_rms
    segments = detect_sources(smoothed_snr, threshold=detect_threshold, n_pixels=min_pixels)
    table = SourceCatalog(signal, segments).to_table(columns=["label", "area", "elongation"])

    # STEP 2: give every object a class  (0 = sky or dropped, 1 = blob, 2 = streak)
    class_of = np.zeros(segments.data.max() + 1, dtype=np.uint8)
    for label, area, elong in zip(table["label"], np.array(table["area"]),
                                  np.array(table["elongation"])):
        if np.isnan(elong):
            continue                                   # no valid shape: noise, dropped
        verdict = manual.get((name, int(label)))
        if verdict == "streak":
            class_of[label] = 2
        elif verdict == "star":
            class_of[label] = 1
        elif verdict == "noise":
            continue
        elif elong >= streak_min_elongation and area >= streak_min_area:
            class_of[label] = 2
        else:
            class_of[label] = 1
    mask = class_of[segments.data]                     # full-size mask: 0 / 1 / 2 per pixel

    # STEP 3: viewable image. Display-only background uses the MEAN (smooth, no
    # whole-number steps). Detection above is unchanged, so review labels stay valid.
    view_bg = Background2D(image, box_size=128, filter_size=3,
                           sigma_clip=SigmaClip(sigma=3.0),
                           bkg_estimator=MeanBackground())
    view_snr = (image - view_bg.background) / view_bg.background_rms
    view = (np.clip((view_snr + 1) / 9, 0, 1) * 255).astype(np.uint8)

    # STEP 4: cut image + mask into tiles, write one YOLO label file per tile
    rebuilt = np.zeros_like(view)
    n_blobs = n_streaks = 0
    for r in tile_starts(view.shape[0]):
        for c in tile_starts(view.shape[1]):
            tile_name = f"{name}_r{r:04d}_c{c:04d}"
            tile_path = out / "images" / f"{tile_name}.png"
            cv2.imwrite(str(tile_path), view[r:r + tile_size, c:c + tile_size])

            tile_mask = mask[r:r + tile_size, c:c + tile_size]
            lines = []
            for mask_value, yolo_class in [(1, 0), (2, 1)]:      # blob -> class 0, streak -> class 1
                outlines, _ = cv2.findContours((tile_mask == mask_value).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for outline in outlines:
                    if len(outline) < 3:
                        continue                                 # a polygon needs 3+ corners
                    points = outline.reshape(-1, 2) / tile_size  # scale to 0..1
                    lines.append(f"{yolo_class} " +
                                 " ".join(f"{x:.6f} {y:.6f}" for x, y in points))
                    if yolo_class == 0:
                        n_blobs += 1
                    else:
                        n_streaks += 1
            (out / "labels" / f"{tile_name}.txt").write_text("\n".join(lines))

            # STEP 5 (part): read the saved tile back and put it in place
            rebuilt[r:r + tile_size, c:c + tile_size] = cv2.imread(str(tile_path),
                                                                  cv2.IMREAD_GRAYSCALE)

    # STEP 5: repatched full-size images, rebuilt FROM THE SAVED TILES
    identical = np.array_equal(rebuilt, view)
    cv2.imwrite(str(out / "repatched" / f"{name}.jpg"), rebuilt, [cv2.IMWRITE_JPEG_QUALITY, 95])
    overlay = cv2.cvtColor(rebuilt, cv2.COLOR_GRAY2BGR)
    overlay[mask == 1] = (255, 120, 0)     # blobs blue   (OpenCV order is Blue, Green, Red)
    overlay[mask == 2] = (0, 0, 255)       # streaks red
    cv2.imwrite(str(out / "repatched" / f"{name}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"{name[:12]}  rebuilt identical: {identical}  blob polygons: {n_blobs}  streak polygons: {n_streaks}")

# STEP 6: the small file YOLO needs to know the class names
(out / "data.yaml").write_text("path: .\ntrain: images\nval: images\nnames:\n  0: blob\n  1: streak\n")
print("done")