import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground


names = ["1a600998-b97c-4307-8632-6fcf573621ff.fits", 
         "3e6beceb-b453-4e49-a222-b661eaaac4d0.fits",
         "CAM_B_20260815T153133_manual_f000207.fits"]

for name in names:
    data = fits.getdata("data/raw/" + name).astype(np.float32)

    bg = Background2D(data, box_size=128, filter_size=3, sigma_clip=SigmaClip(sigma=3.0),
                      bkg_estimator=MedianBackground())

    snr = (data - bg.background) / bg.background_rms

    print(name[:12])

    print("background min/max: ", bg.background_mesh.min(), bg.background_mesh.max())
    print("noise min/max: ", bg.background_rms_mesh.min(), bg.background_rms_mesh.max())
    print("Signal Noise Ratio percentiles 1/50/99: ", np.percentile(snr[::4, ::4], [1, 50, 99]))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    im0 = ax[0].imshow(bg.background_mesh, cmap="viridis")
    ax[0].set_title(name[:12] + "background (box grid)")

    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(bg.background_rms_mesh, cmap="viridis")

    ax[1].set_title("noise (box grid)")
    fig.colorbar(im1, ax=ax[1])

    ax[2].imshow(snr[3000:4024, 4000:5024], cmap="gray", vmin=-2, vmax=5)
    ax[2].set_title("SNR crop (same center square as before)")

plt.show()