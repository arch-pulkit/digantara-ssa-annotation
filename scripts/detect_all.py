from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # save figures to files instead of opening windows
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.convolution import convolve
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog

detect_threshold = 5.0        # x smoothed flicker
min_pixels = 5                # touching pixels per object
streak_min_elongation = 5.0   # clear streak: long / wide
streak_min_area = 30          # clear streak: pixels
review_min_elongation = 3.0   # review band starts here
review_min_area = 50          # review band: big enough to judge

raw_folder = Path("data/raw")
review_folder = Path("outputs/review")
review_folder.mkdir(parents=True, exist_ok=True)
kernel = make_2dgaussian_kernel(3.0, size=5)


def measure_background(img):
    return Background2D(img, box_size=128, filter_size=3,
                        sigma_clip=SigmaClip(sigma=3.0),
                        bkg_estimator=MedianBackground())


def save_cutouts(snr, rows, title, out_path):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for a in axes.flat:
        a.axis("off")
    for a, row in zip(axes.flat, rows[:16]):
        x = int(round(float(row["x_centroid"])))
        y = int(round(float(row["y_centroid"])))
        cutout = snr[max(y - 60, 0):y + 60, max(x - 60, 0):x + 60]
        a.imshow(cutout, cmap="gray", vmin=-1, vmax=6)
        a.set_title(f"#{row['label']} e={float(row['elongation']):.1f} "
                    f"area={row['area'].value:.0f}", fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


for path in sorted(raw_folder.iterdir()):
    if path.suffix.lower() != ".fits":
        continue
    name = path.stem

    # background, noise, SNR
    image = fits.getdata(path).astype(np.float32)
    bg = measure_background(image)
    signal = image - bg.background
    snr = signal / bg.background_rms

    # smooth, re-measure flicker, detect
    smoothed = convolve(signal, kernel)
    bg_smooth = measure_background(smoothed)
    smoothed_snr = (smoothed - bg_smooth.background) / bg_smooth.background_rms
    segments = detect_sources(smoothed_snr, threshold=detect_threshold, n_pixels=min_pixels)

    # measure shapes
    table = SourceCatalog(signal, segments).to_table(
        columns=["label", "x_centroid", "y_centroid", "area", "elongation", "orientation"])
    elongation = np.array(table["elongation"])
    area = np.array(table["area"])

    # classify
        # classify
    valid = ~np.isnan(elongation)
    is_streak = valid & (elongation >= streak_min_elongation) & (area >= streak_min_area)
    needs_review = valid & ~is_streak & (
        ((elongation >= review_min_elongation) & (area >= review_min_area)) |
        ((elongation >= streak_min_elongation) & (area >= 15))
    )

    streaks = table[is_streak]
    streaks.sort("elongation", reverse=True)
    review = table[needs_review]
    review.sort("elongation", reverse=True)

    print(f"{name[:12]}  objects={len(table)}  no-shape={(~valid).sum()}  "
          f"streaks={len(streaks)}  to-review={len(review)}")

    # save all cutouts, 16 per page
    for start in range(0, len(streaks), 16):
        save_cutouts(snr, streaks[start:start + 16], f"{name} - streaks",
                     review_folder / f"{name}_streaks_{start // 16 + 1}.png")
    for start in range(0, len(review), 16):
        save_cutouts(snr, review[start:start + 16], f"{name} - REVIEW",
                     review_folder / f"{name}_review_{start // 16 + 1}.png")