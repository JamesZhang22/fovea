import numpy as np
from PIL import Image


def to_gray(im: Image.Image) -> np.ndarray:
    """PIL image to float32 grayscale array."""
    return np.asarray(im.convert("L"), dtype=np.float32)


def brenner(gray: np.ndarray) -> float:
    """Mean squared 2-pixel difference, fast and good for within-burst ranking."""
    dx = gray[:, 2:] - gray[:, :-2]
    dy = gray[2:, :] - gray[:-2, :]
    return float(np.mean(dx**2) + np.mean(dy**2))


def tenengrad(gray: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude."""
    gx = _sobel(gray)
    gy = _sobel(gray.T).T
    return float(np.mean(gx**2 + gy**2))


def edge_sharpness(gray: np.ndarray) -> float:
    """Contrast-normalized gradient strength on strong edges, less content-dependent
    than raw gradient metrics because local contrast divides out."""
    gx = _sobel(gray)
    gy = _sobel(gray.T).T
    mag = np.sqrt(gx**2 + gy**2)
    strong = mag > np.percentile(mag, 99)
    if not strong.any():
        return 0.0
    contrast = _local_range(gray)[strong]
    widths = np.maximum(contrast, 1.0) / mag[strong]
    return float(1.0 / np.mean(widths))


def metrics(gray: np.ndarray) -> dict[str, float]:
    """All classical focus metrics for one patch."""
    return {
        "brenner": brenner(gray),
        "tenengrad": tenengrad(gray),
        "edge_sharpness": edge_sharpness(gray),
    }


def rank_burst(values: list[float]) -> list[float]:
    """Percentile rank of each value within its burst, 1.0 is sharpest, 0.5 for a singleton."""
    n = len(values)
    if n == 1:
        return [0.5]
    order = np.argsort(np.argsort(values))
    return [float(r) / (n - 1) for r in order]


def _sobel(g: np.ndarray) -> np.ndarray:
    """Horizontal Sobel response for the interior, zero-padded to input shape."""
    out = np.zeros_like(g)
    out[1:-1, 1:-1] = (
        (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[1:-1, :-2] + g[2:, :-2])
    ) / 8.0
    return out


def _local_range(g: np.ndarray) -> np.ndarray:
    """Max minus min over a 3x3 neighborhood, a cheap local contrast estimate."""
    stacked = np.stack(
        [
            g[1 + dy : g.shape[0] - 1 + dy, 1 + dx : g.shape[1] - 1 + dx]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
    )
    out = np.zeros_like(g)
    out[1:-1, 1:-1] = stacked.max(axis=0) - stacked.min(axis=0)
    return out
