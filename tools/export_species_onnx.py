"""Export the BioCLIP 2 image encoder to ONNX and verify numerically against torch.

Only the visual tower ships, text embeddings are precomputed by build_species_labels.py.
The graph takes CLIP-normalized 224px images and returns L2-normalized embeddings.

Usage: uv run python tools/export_species_onnx.py [--out models/species.onnx]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

MODEL_STR = "hf-hub:imageomics/bioclip-2"
IMAGE_PX = 224
TOLERANCE = 1e-3


class VisualEncoder(torch.nn.Module):
    """BioCLIP image tower + L2 normalize, matching pybioclip's create_image_features."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(image), dim=-1)


def export(out: Path) -> None:
    model, _ = open_clip.create_model_from_pretrained(MODEL_STR)
    model.eval()
    encoder = VisualEncoder(model)

    # fixed batch 1: the stage scores one crop per burst, and the dynamo exporter
    # bakes batch-1 reshapes into ViT attention that break dynamic axes anyway
    example = torch.randn(1, 3, IMAGE_PX, IMAGE_PX)
    torch.onnx.export(
        encoder,
        (example,),
        str(out),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=18,
    )
    size_mb = sum(f.stat().st_size for f in out.parent.glob(f"{out.name}*")) / 1e6
    print(f"exported {out} ({size_mb:.0f} MB)")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    diff = 0.0
    for _ in range(4):
        x = torch.randn(1, 3, IMAGE_PX, IMAGE_PX)
        with torch.no_grad():
            ref = encoder(x).numpy()
        (got,) = sess.run(None, {"image": x.numpy()})
        diff = max(diff, float(np.abs(got - ref).max()))
    print(f"CPU EP max diff {diff:.2e} | {'OK' if diff < TOLERANCE else 'DIVERGED'}")
    if diff >= TOLERANCE:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/species.onnx")
    args = parser.parse_args()
    export(Path(args.out))
