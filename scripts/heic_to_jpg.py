# Run from project root directory

from pathlib import Path
from PIL import Image, ImageOps
import pillow_heif
import os

pillow_heif.register_heif_opener()
src=Path(os.environ['IMAGES'])
dst=Path(os.environ['IMAGES_JPG'])
dst.mkdir(exist_ok=True)

for p in src.iterdir():
    try:
        im=Image.open(p); im=ImageOps.exif_transpose(im)
        im.convert("RGB").save(dst/(p.stem+".jpg"), "JPEG", quality=95)
    except Exception as e:
        print("skip", p, e)
print("done")