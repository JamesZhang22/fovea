"""Export eye model weights to ONNX and verify outputs numerically.

Usage: uv run python tools/export_eye_onnx.py [--weights models/eye_own_v1.pt]
           [--out models/eye.onnx]

Verifies torch vs onnxruntime CPU and CoreML outputs on random inputs, silent
post-conversion divergence is a documented failure mode.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eye_net import INPUT_PX, EyeNet

TOLERANCE = 1e-3  # max abs logit difference allowed vs torch


def export(weights: Path, out: Path) -> None:
    model = EyeNet(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    example = torch.randn(1, 3, INPUT_PX, INPUT_PX)
    torch.onnx.export(
        model,
        (example,),
        str(out),
        input_names=["image"],
        output_names=["logits_x", "logits_y"],
        dynamic_axes={"image": {0: "batch"}, "logits_x": {0: "batch"}, "logits_y": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort

    batch = torch.randn(8, 3, INPUT_PX, INPUT_PX)
    with torch.no_grad():
        ref_x, ref_y = model(batch)

    provider_sets = (
        ["CPUExecutionProvider"],
        ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    for providers in provider_sets:
        sess = ort.InferenceSession(str(out), providers=providers)
        t0 = time.perf_counter()
        ox, oy = sess.run(None, {"image": batch.numpy()})
        dt = (time.perf_counter() - t0) / len(batch) * 1000
        dx = float(np.abs(ox - ref_x.numpy()).max())
        dy = float(np.abs(oy - ref_y.numpy()).max())
        used = sess.get_providers()[0]
        status = "OK" if max(dx, dy) < TOLERANCE else "DIVERGED"
        print(f"{used}: max diff x={dx:.2e} y={dy:.2e} | {dt:.1f} ms/img | {status}")
        if status == "DIVERGED" and used == "CPUExecutionProvider":
            sys.exit(1)  # CPU is the runtime EP, CoreML results are informational


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/eye_own_v1.pt")
    parser.add_argument("--out", default="models/eye.onnx")
    args = parser.parse_args()
    export(Path(args.weights), Path(args.out))
