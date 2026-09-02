from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from fovea.core.ingest.decode import roi_native

# SimCC geometry, tools/eye_net.py trains against these same values.
INPUT_PX = 192
BINS_PER_AXIS = INPUT_PX * 2  # half-pixel resolution.
TARGET_SIGMA_BINS = 8.0  # gaussian spread of the soft training target.
BOX_PAD_FRACTION = 0.15  # square padded crop matching the training distribution.

DEFAULT_MODEL = Path("models/eye.onnx")


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _peak() -> float:
    """Peak probability of a perfect soft target, normalizes confidence to ~1."""
    offsets = np.arange(-40, 41)
    return float(1.0 / np.exp(-(offsets**2) / (2 * TARGET_SIGMA_BINS**2)).sum())


PEAK = _peak()


def decode_simcc(logits_x: np.ndarray, logits_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Soft-argmax coordinates in input pixels and confidence in [0, 1], batched."""
    coords, confs = [], []
    for logits in (logits_x, logits_y):
        p = _softmax(logits)
        bins = np.arange(BINS_PER_AXIS, dtype=np.float32)
        coords.append((p * bins).sum(axis=-1) * (INPUT_PX / BINS_PER_AXIS))
        confs.append(p.max(axis=-1))
    xy = np.stack(coords, axis=-1)
    conf = np.minimum(confs[0], confs[1]) / PEAK
    return xy, np.clip(conf, 0.0, 1.0)


def square_region(box: dict) -> tuple[int, int, int, int]:
    """Padded square around a bird box in full-resolution pixels."""
    w, h = box["x1"] - box["x0"], box["y1"] - box["y0"]
    side = max(w, h) * (1 + 2 * BOX_PAD_FRACTION)
    cx, cy = (box["x0"] + box["x1"]) / 2, (box["y0"] + box["y1"]) / 2
    return (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))


class EyeLocator:
    """Eye keypoint inference on bird crops, onnxruntime CPU."""

    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def locate(self, jpeg: bytes, box: dict) -> dict:
        """Eye position in full-resolution pixels plus confidence for one bird box."""
        x0, y0, x1, y1 = square_region(box)
        crop = roi_native(jpeg, (x0, y0, x1, y1)).resize((INPUT_PX, INPUT_PX), Image.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        lx, ly = self.session.run(None, {"image": arr})
        xy, conf = decode_simcc(lx[0], ly[0])
        return {
            "x": x0 + float(xy[0]) / INPUT_PX * (x1 - x0),
            "y": y0 + float(xy[1]) / INPUT_PX * (y1 - y0),
            "confidence": round(float(conf), 3),
        }
