import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_PX = 224  # BioCLIP 2 input size, squash-resized without crop like its training
CROP_PAD_FRACTION = 0.1  # context around the bird box helps CLIP-style models
TOP_K = 3

# CLIP normalization constants, must match the training preprocessing
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def merge_groups(probs: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum probability mass over label rows sharing a display name (taxonomic synonyms)."""
    merged = np.zeros(n_groups, dtype=probs.dtype)
    np.add.at(merged, group_idx, probs)
    return merged


class SpeciesClassifier:
    """BioCLIP 2 zero-shot bird ID against precomputed Aves text embeddings."""

    def __init__(self, model_path, labels_path) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        z = np.load(labels_path)
        self.emb = z["emb"]
        self.logit_scale = float(z["logit_scale"])
        self.scientific = z["scientific"].astype(str)
        self.common = z["common"].astype(str)
        self.family = z["family"].astype(str)
        keys = np.where(self.common != "", np.char.lower(self.common), self.scientific)
        _, self.rep_rows, self.group_idx = np.unique(keys, return_index=True, return_inverse=True)

    def classify(self, crop: Image.Image) -> list[dict]:
        """Top-K species for a padded bird crop, confidences merged across synonyms."""
        im = crop.convert("RGB").resize((IMAGE_PX, IMAGE_PX), Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        (emb,) = self.session.run(None, {"image": x})
        logits = self.logit_scale * (emb[0] @ self.emb.T)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        merged = merge_groups(probs, self.group_idx, len(self.rep_rows))
        top = np.argsort(-merged)[:TOP_K]
        return [
            {
                "common": self.common[r] or None,
                "scientific": self.scientific[r],
                "family": self.family[r] or None,
                "confidence": round(float(merged[g]), 3),
            }
            for g in top
            for r in [self.rep_rows[g]]
        ]
