from pathlib import Path

import numpy as np

# ordinal blur levels in native pixels, tools/focus_degrade.py trains against these
DEFOCUS_RADII_PX = [0.0, 0.5, 1.0, 1.5, 2.25, 3.25, 4.5, 6.5, 9.0, 12.0]
N_LEVELS = len(DEFOCUS_RADII_PX)
PATCH_PX = 96  # native-resolution model input

# published radius -> score curve, <=1px is one "critically sharp" band because the
# corpus anchors carry camera optics + processing blur (see context/learnings.md)
CURVE_RADII_PX = [0.0, 1.0, 2.0, 3.25, 4.5, 6.5, 12.0]
CURVE_SCORES = [100.0, 85.0, 70.0, 55.0, 40.0, 20.0, 0.0]

MIN_CONFIDENCE = 0.15  # entropy confidence below this abstains, calibrated in M5 step 6
DEFAULT_MODEL = Path("models/focus.onnx")


def index_to_radius(index: np.ndarray) -> np.ndarray:
    """Continuous level index back to blur radius via piecewise-linear interpolation."""
    radii = np.array(DEFOCUS_RADII_PX)
    idx = np.clip(index, 0, len(radii) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(radii) - 1)
    frac = idx - lo
    return radii[lo] * (1 - frac) + radii[hi] * frac


def radius_to_score(radius_px: float) -> float:
    """The fixed 0-100 mapping, comparable across images because radius is."""
    return float(np.interp(radius_px, CURVE_RADII_PX, CURVE_SCORES))


def decode_ordinal(logits: np.ndarray) -> tuple[float, float]:
    """Soft-argmax blur radius and entropy confidence in [0, 1] for one sample."""
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    index = float((p * np.arange(N_LEVELS)).sum())
    entropy = float(-(p * np.log(p + 1e-9)).sum())
    confidence = 1.0 - entropy / np.log(N_LEVELS)
    return float(index_to_radius(np.array([index]))[0]), confidence


class FocusScorer:
    """Ordinal defocus inference on native-res gray eye patches, onnxruntime CPU."""

    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def score(self, patch_gray: np.ndarray) -> dict:
        """Blur radius, 0-100 score, and confidence. Score is None when abstaining."""
        x = (patch_gray / 255.0).astype(np.float32)[None, None]
        (logits,) = self.session.run(None, {"patch": x})
        radius, confidence = decode_ordinal(logits[0])
        abstain = confidence < MIN_CONFIDENCE
        return {
            "radius_px": round(radius, 2),
            "score": None if abstain else round(radius_to_score(radius), 1),
            "confidence": round(confidence, 3),
        }
