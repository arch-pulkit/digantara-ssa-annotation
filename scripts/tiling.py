from pathlib import Path
import numpy as np
from astropy.io import fits

TILE_SIZE = 1024

def tile_starts(length):
    starts = list(range(0, length - TILE_SIZE + 1, TILE_SIZE))
    if starts[-1] + TILE_SIZE < length:
        starts.append(length - TILE_SIZE)
    return starts

folder = Path("data/raw")

for path in sorted(folder.iterdir()):
    if path.suffix.lower() == ".fits":
        image = fits.getdata(path)
        height, width = image.shape

        row_starts = tile_starts(height)
        col_starts = tile_starts(width)

        patches = []
        for row in row_starts:
            for col in col_starts:
                patch = image[row:row + TILE_SIZE, col:col + TILE_SIZE]
                patches.append((row, col, patch))

        rebuilt = np.zeros_like(image)
        covered = np.zeros(image.shape, dtype=bool)
        for row, col, patch in patches:
            rebuilt[row:row + TILE_SIZE, col:col + TILE_SIZE] = patch
            covered[row:row + TILE_SIZE, col:col + TILE_SIZE] = True

        count_ok = len(patches) == len(row_starts) * len(col_starts)
        size_ok = all(patch.shape == (TILE_SIZE, TILE_SIZE) for row, col, patch in patches)
        covered_ok = covered.all()
        equal_ok = np.array_equal(rebuilt, image)

        print(path.name[:12],
              "patches:", len(patches),
              "count:", count_ok,
              "size:", size_ok,
              "covered:", covered_ok,
              "identical:", equal_ok)

print("row starts (down): ", row_starts)
print("col starts (across): ", col_starts)