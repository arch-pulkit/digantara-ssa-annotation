import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.convolution import convolve
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import (detect_sources, deblend_sources,
                                    make_2dgaussian_kernel, SourceCatalog)

file = "data/raw/1a600998-b97c-4307-8632-6fcf573621ff.fits"
snr_threshold = 3.0
min_pixels = 5

# background, noise, SNR
image = fits.getdata(file).astype(np.float32)
bg = Background2D(image, box_size=128, filter_size=3,
                  sigma_clip=SigmaClip(sigma=3.0),
                  bkg_estimator=MedianBackground())
signal = image - bg.background
snr = signal / bg.background_rms

# smoothing: average each pixel with its neighbors (star-sized blur)
kernel = make_2dgaussian_kernel(3.0, size=5)
smoothed = convolve(signal, kernel)

# re-measure sky and flicker AFTER smoothing, so the ruler matches the image
bg_smooth = Background2D(smoothed, box_size=128, filter_size=3,
                         sigma_clip=SigmaClip(sigma=3.0),
                         bkg_estimator=MedianBackground())
smoothed_snr = (smoothed - bg_smooth.background) / bg_smooth.background_rms

# detection: pixels at least 5x the NEW flicker, at least 5 touching
detect_threshold = 5.0
segments = detect_sources(smoothed_snr, threshold=detect_threshold, n_pixels=min_pixels)
print("objects found:", segments.n_labels)

# deblending: split objects that contain two separate peaks
#segments = deblend_sources(smoothed, segments, n_pixels=min_pixels,
#                           n_levels=32, contrast=0.01, progress_bar=False)
#print("objects after deblending:", segments.n_labels)

# measure shapes
catalog = SourceCatalog(signal, segments)
table = catalog.to_table(columns=["label", "x_centroid", "y_centroid",
                                  "area", "semimajor_axis", "semiminor_axis",
                                  "elongation", "orientation", "max_value"])

elongation = np.array(table["elongation"])
area = np.array(table["area"])
print("elongation percentiles 50/90/99:", np.nanpercentile(elongation, [50, 90, 99]))
print("objects with no valid shape (nan):", np.isnan(elongation).sum())

print("objects with elongation > 2:", (elongation > 2).sum())

table.sort("elongation", reverse=True)

# plots: all objects
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].hist(elongation, bins=100, log=True)
ax[0].set_xlabel("elongation (a / b)")
ax[0].set_ylabel("number of objects")
ax[1].scatter(area, elongation, s=2, alpha=0.3)
ax[1].set_xscale("log")
ax[1].set_xlabel("area (pixels)")
ax[1].set_ylabel("elongation")

# streak candidates: big enough to trust AND elongated
area_all = np.array(table["area"])
elongation_all = np.array(table["elongation"])
candidates = table[(area_all >= 15) & (elongation_all >= 2.5)]
print("streak candidates:", len(candidates))
print(candidates["label", "x_centroid", "y_centroid", "area", "elongation", "orientation"])

# cutouts around each candidate
half = 60
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for a in axes.flat:
    a.axis("off")
for a, row in zip(axes.flat, candidates[6:22]):
    x = int(round(float(row["x_centroid"])))
    y = int(round(float(row["y_centroid"])))
    cutout = snr[max(y - half, 0):y + half, max(x - half, 0):x + half]
    a.imshow(cutout, cmap="gray", vmin=-1, vmax=6)
    a.set_title(f"#{row['label']}  e={float(row['elongation']):.1f}  "
                f"area={row['area'].value:.0f}", fontsize=9)
plt.tight_layout()
plt.show()