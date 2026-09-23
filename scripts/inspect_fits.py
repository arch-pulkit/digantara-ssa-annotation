from astropy.io import fits
from pathlib import Path

path = "data/raw/1a600998-b97c-4307-8632-6fcf573621ff.fits"
fits.info(path)

folder = Path("data/raw")
i = 0
for image in folder.iterdir():
    if image.suffix.lower() == ".fits":
        i += 1
        hdr = fits.getheader(image)
        print(f"\n\n{i}. {image.name}\n")
        print(repr(hdr))

