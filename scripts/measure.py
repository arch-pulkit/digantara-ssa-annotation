from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

folder = Path("data/raw")

for image in sorted(folder.iterdir()):
    if image.suffix.lower() == ".fits":
        data = fits.getdata(image)
        hdr = fits.getheader(image)

        mean, median, sigma = sigma_clipped_stats(data[::4, 4::], sigma=3.0)
        saturated = (data == 65535).sum()
        """
        median = brightness of sky
        sigma = noise of sky
        saturated = number of pixels at max, i.e. 65535
        """
        print(image.name[:12], hdr["SERIAL"], hdr["DATE-OBS"][:10],
                "median =", round(median, 1),
                "sigma =", round(sigma, 1),
                "saturated =", saturated )  