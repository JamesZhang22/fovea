import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_PX = 224  # BioCLIP 2 input size, squash-resized without crop like its training
CROP_PAD_FRACTION = 0.1  # context around the bird box helps CLIP-style models
TOP_K = 3

# CLIP normalization constants, must match the training preprocessing
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# region key -> IOC breeding-range codes, species without codes always pass the filter
REGIONS = {
    "north-america": {"NA", "MA"},
    "south-america": {"SA"},
    "eurasia": {"EU"},
    "africa": {"AF"},
    "south-asia": {"OR"},
    "australasia": {"AU"},
}
OCEANIC_CODES = {"AN", "AO", "PO", "IO", "TrO", "TO", "NO", "SO"}  # pelagics pass everywhere


def region_mask(regions: np.ndarray, region: str | None) -> np.ndarray:
    """Boolean row mask for a region key, fail-open for unknown or oceanic ranges."""
    if region is None:
        return np.ones(len(regions), dtype=bool)
    allowed = REGIONS[region] | OCEANIC_CODES
    return np.array(
        [not r or bool(set(r.split(",")) & allowed) for r in regions.astype(str)], dtype=bool
    )


def label_names(labels_path) -> list[str]:
    """Sorted display names (common, scientific when unnamed) for autocomplete."""
    z = np.load(labels_path)
    common = z["common"].astype(str)
    scientific = z["scientific"].astype(str)
    return sorted(set(np.where(common != "", common, scientific)))


def species_info(labels_path, name: str) -> tuple[str | None, str | None]:
    """Scientific name and family for a display name, (None, None) when unknown."""
    z = np.load(labels_path)
    common = z["common"].astype(str)
    scientific = z["scientific"].astype(str)
    idx = np.flatnonzero(np.where(common != "", common, scientific) == name)
    if not len(idx):
        return None, None
    row = idx[0]
    return scientific[row], z["family"].astype(str)[row] or None


def merge_groups(probs: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum probability mass over label rows sharing a display name (taxonomic synonyms)."""
    merged = np.zeros(n_groups, dtype=probs.dtype)
    np.add.at(merged, group_idx, probs)
    return merged


class SpeciesClassifier:
    """BioCLIP 2 zero-shot bird ID against precomputed Aves text embeddings."""

    def __init__(self, model_path, labels_path, region: str | None = None) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        z = np.load(labels_path)
        self.emb = z["emb"]
        self.logit_scale = float(z["logit_scale"])
        self.scientific = z["scientific"].astype(str)
        self.common = z["common"].astype(str)
        self.family = z["family"].astype(str)
        self.mask = region_mask(z["regions"], region) if "regions" in z else None
        keys = np.where(self.common != "", np.char.lower(self.common), self.scientific)
        _, self.rep_rows, self.group_idx = np.unique(keys, return_index=True, return_inverse=True)

    def classify(self, crop: Image.Image) -> list[dict]:
        """Top-K species for a padded bird crop, confidences merged across synonyms."""
        im = crop.convert("RGB").resize((IMAGE_PX, IMAGE_PX), Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        (emb,) = self.session.run(None, {"image": x})
        logits = self.logit_scale * (emb[0] @ self.emb.T)
        if self.mask is not None:
            logits[~self.mask] = -np.inf  # softmax renormalizes over the allowed rows
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
