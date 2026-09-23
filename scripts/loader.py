import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
from astropy.visualization import ImageNormalize, ZScaleInterval

#Dividing files per group based on data from measure.py
#3 groups, 1 file per group

names = ["1a600998-b97c-4307-8632-6fcf573621ff.fits", 
         "3e6beceb-b453-4e49-a222-b661eaaac4d0.fits",
         "CAM_B_20260815T153133_manual_f000207.fits"]

for name in names:
    data = fits.getdata("data/raw/" + name)
    print(name[:12])

    print("fraction of pixels that are exactly ): ", round((data == 0).mean(), 3))
    print("percentiles 1/25/50/75/99: ", np.percentile(data[::4, ::4], [1, 25, 50, 75, 99]))
    print("max: ", data.max())

    print("pixels equal to max:", (data == data.max()).sum())
    p50, p84 = np.percentile(data[::4, ::4], [50, 84])
    print("noise estimate (p84 - median):", p84 - p50)

    corp = data[3000:4024, 4000:5024]
    norm = ImageNormalize(corp, interval=ZScaleInterval())
    plt.figure(figsize=(8, 8))
    plt.imshow(corp, cmap="gray", norm=norm)
    plt.title(name[:12])

plt.show()