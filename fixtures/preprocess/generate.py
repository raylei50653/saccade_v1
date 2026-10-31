"""Offline fixture generator; never imported by the native runtime/tests."""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent

def image(width, height):
    pixels = bytes(v for y in range(height) for x in range(width)
                   for v in (160 + (3*x+y) % 64,
                             64 + (x+2*y) % 64,
                             8 + (2*x+3*y) % 32))
    return Image.frombytes("RGB", (width, height), pixels)

base = image(35, 27)
base.save(root / "baseline.jpg", quality=95, subsampling=2, optimize=False)
image(48, 32).save(root / "alternate.jpg", quality=95, subsampling=2, optimize=False)
base.convert("L").save(root / "gray.jpg", quality=95, optimize=False)
base.save(root / "progressive.jpg", quality=95, subsampling=2, progressive=True, optimize=False)
base.convert("CMYK").save(root / "cmyk.jpg", quality=95, optimize=False)
base.save(root / "444.jpg", quality=95, subsampling=0, optimize=False)
