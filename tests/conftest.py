import numpy as np
from PIL import Image, ImageDraw


def structured_image(size: int = 200) -> Image.Image:
    """Random ellipses on a gray field, gives real edges for focus metrics to measure."""
    rng = np.random.default_rng(42)
    im = Image.new("L", (size, size), 120)
    d = ImageDraw.Draw(im)
    for _ in range(30):
        x, y, r = rng.integers(10, size - 10), rng.integers(10, size - 10), rng.integers(5, 25)
        d.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.integers(0, 256)))
    return im
