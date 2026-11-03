# Writes two synthetic 1920x1080 baseline 4:2:0 JPEGs with different content.
import numpy as np
from PIL import Image

y, x = np.mgrid[0:1080, 0:1920]
a = np.stack([160 + (3 * x + y) % 64, 64 + (x + 2 * y) % 64, 8 + (2 * x + 3 * y) % 32], -1)
b = np.stack([32 + (x // 8 + y // 8) % 2 * 160, 200 - (x + y) % 128, 16 + (x * y) % 200], -1)
for name, rgb in (("a.jpg", a), ("b.jpg", b)):
    Image.fromarray(rgb.astype(np.uint8), "RGB").save(name, quality=95, subsampling=2, optimize=False, progressive=False)
