import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from fovea.core.score.classical import metrics, rank_burst, to_gray

rng = np.random.default_rng(42)


def structured_image(size: int = 200) -> Image.Image:
    """Random ellipses on a gray field, gives real edges for the metrics to measure."""
    im = Image.new("L", (size, size), 120)
    d = ImageDraw.Draw(im)
    for _ in range(30):
        x, y, r = rng.integers(10, size - 10), rng.integers(10, size - 10), rng.integers(5, 25)
        d.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.integers(0, 256)))
    return im


@pytest.mark.parametrize("metric", ["brenner", "tenengrad", "edge_sharpness"])
def test_blur_lowers_every_metric(metric: str) -> None:
    sharp = structured_image()
    prev = metrics(to_gray(sharp))[metric]
    for radius in (1, 3, 6):
        cur = metrics(to_gray(sharp.filter(ImageFilter.GaussianBlur(radius))))[metric]
        assert cur < prev, f"{metric} did not decrease at radius {radius}"
        prev = cur


def test_metrics_on_flat_patch_are_zero() -> None:
    flat = np.full((100, 100), 128, dtype=np.float32)
    m = metrics(flat)
    assert m["brenner"] == 0.0 and m["tenengrad"] == 0.0 and m["edge_sharpness"] == 0.0


def test_rank_burst() -> None:
    assert rank_burst([10.0, 30.0, 20.0]) == [0.0, 1.0, 0.5]
    assert rank_burst([5.0]) == [0.5]
