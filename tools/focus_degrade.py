"""Synthetic degradation for the focus model: known-radius defocus and linear motion.

The label axis is blur radius in native pixels, which is what makes the trained score
comparable across images by construction. Kernels are supersampled so sub-pixel radii
are real, and mild sensor noise is added after blurring because real soft frames are
noisy soft frames.

Usage: uv run python tools/focus_degrade.py preview   # strip image for eyeballing
"""

import sys
from pathlib import Path

import numpy as np

from fovea.core.score.model import DEFOCUS_RADII_PX  # canonical ordinal levels

MOTION_LENGTHS_PX = [3.0, 6.0, 10.0, 16.0]  # equivalent severity label = length / 2
SUPERSAMPLE = 8
NOISE_STD_DN = 1.5  # sensor-ish noise applied after blur


def disc_kernel(radius_px: float) -> np.ndarray:
    """Anti-aliased circular aperture PSF."""
    if radius_px <= 0:
        return np.ones((1, 1), dtype=np.float32)
    half = int(np.ceil(radius_px)) + 1
    n = 2 * half + 1
    ss = SUPERSAMPLE
    coords = (np.arange(n * ss) - (n * ss - 1) / 2) / ss
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    fine = (yy**2 + xx**2 <= radius_px**2).astype(np.float32)
    kernel = fine.reshape(n, ss, n, ss).mean(axis=(1, 3))
    if kernel.sum() == 0:  # radius below the supersample pitch, effectively no blur
        return np.ones((1, 1), dtype=np.float32)
    return kernel / kernel.sum()


def motion_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    """Line segment PSF for linear motion during exposure."""
    half = int(np.ceil(length_px / 2)) + 1
    n = 2 * half + 1
    ss = SUPERSAMPLE
    coords = (np.arange(n * ss) - (n * ss - 1) / 2) / ss
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    rad = np.deg2rad(angle_deg)
    along = xx * np.cos(rad) + yy * np.sin(rad)
    across = -xx * np.sin(rad) + yy * np.cos(rad)
    fine = ((np.abs(along) <= length_px / 2) & (np.abs(across) <= 0.5)).astype(np.float32)
    kernel = fine.reshape(n, ss, n, ss).mean(axis=(1, 3))
    if kernel.sum() == 0:
        return np.ones((1, 1), dtype=np.float32)
    return kernel / kernel.sum()


def apply_kernel(gray: np.ndarray, kernel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """FFT convolution with edge padding, plus post-blur sensor noise."""
    if kernel.shape == (1, 1):
        out = gray.astype(np.float32)
    else:
        pad = kernel.shape[0] // 2
        padded = np.pad(gray.astype(np.float32), pad, mode="reflect")
        fk = np.fft.rfft2(kernel, padded.shape)
        out = np.fft.irfft2(np.fft.rfft2(padded) * fk, padded.shape)
        shift = kernel.shape[0] - 1
        out = out[shift : shift + gray.shape[0], shift : shift + gray.shape[1]]
    out = out + rng.normal(0.0, NOISE_STD_DN, out.shape)
    return np.clip(out, 0, 255).astype(np.float32)


def preview() -> None:
    import json

    from PIL import Image

    corpus = Path("data/sharp-eyes")
    rejected = set(json.loads((corpus / "rejected.json").read_text()))
    kept = [r for r in json.loads((corpus / "index.json").read_text()) if r["id"] not in rejected]
    rng = np.random.default_rng(7)

    rows = []
    for rec in (kept[0], kept[len(kept) // 2], kept[-1]):
        gray = np.asarray(
            Image.open(corpus / "patches" / f"{rec['id']}.png").convert("L"), dtype=np.float32
        )
        cells = [apply_kernel(gray, disc_kernel(r), rng) for r in DEFOCUS_RADII_PX]
        cells += [
            apply_kernel(gray, motion_kernel(length, 30.0), rng) for length in MOTION_LENGTHS_PX
        ]
        rows.append(np.concatenate(cells, axis=1))
    strip = np.concatenate(rows, axis=0).astype(np.uint8)
    out = Path("data/sharp-eyes/degrade_preview.png")
    Image.fromarray(strip).resize((strip.shape[1] * 2, strip.shape[0] * 2), Image.NEAREST).save(out)
    print(f"columns: defocus r={DEFOCUS_RADII_PX} then motion L={MOTION_LENGTHS_PX} @30deg")
    print(f"open {out}")


if __name__ == "__main__":
    {"preview": preview}[sys.argv[1]]()
