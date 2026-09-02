"""Export the focus model to ONNX and verify numerically against torch.

Usage: uv run python tools/export_focus_onnx.py [--weights models/focus_v1.pt]
           [--out models/focus.onnx]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent))
from focus_net import PATCH_PX, FocusNet

TOLERANCE = 1e-4


def export(weights: Path, out: Path) -> None:
    model = FocusNet()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    example = torch.randn(1, 1, PATCH_PX, PATCH_PX)
    torch.onnx.export(
        model,
        (example,),
        str(out),
        input_names=["patch"],
        output_names=["logits"],
        dynamic_axes={"patch": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {out} ({out.stat().st_size / 1e3:.0f} KB)")

    batch = torch.randn(16, 1, PATCH_PX, PATCH_PX)
    with torch.no_grad():
        ref = model(batch).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    (got,) = sess.run(None, {"patch": batch.numpy()})
    diff = float(np.abs(got - ref).max())
    print(f"CPU EP max diff {diff:.2e} | {'OK' if diff < TOLERANCE else 'DIVERGED'}")
    if diff >= TOLERANCE:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/focus_v1.pt")
    parser.add_argument("--out", default="models/focus.onnx")
    args = parser.parse_args()
    export(Path(args.weights), Path(args.out))
