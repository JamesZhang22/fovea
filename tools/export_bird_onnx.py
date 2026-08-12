"""Export RF-DETR-Nano to ONNX and verify detections against the torch model.

Usage: uv run --group ml python tools/export_bird_onnx.py [--out models/bird.onnx]

The packaged app runs detection through onnxruntime, torch never ships.
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path


def export(out: Path) -> None:
    import numpy as np
    from rfdetr import RFDETRNano

    from fovea.core.detect.bird import OnnxBirdDetector
    from fovea.core.ingest import cr3
    from fovea.core.ingest.decode import decode_scaled

    model = RFDETRNano()
    with tempfile.TemporaryDirectory() as tmp:
        model.export(output_dir=tmp)
        exported = next(Path(tmp).glob("*.onnx"))
        out.parent.mkdir(exist_ok=True)
        shutil.copy(exported, out)
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB)")

    sample = sorted(Path("data/labeling-set").glob("*.CR3"))
    if not sample:
        print("no local CR3s, skipping numeric verification")
        return

    onnx_det = OnnxBirdDetector(out)
    agree = 0
    for p in sample[:10]:
        im = decode_scaled(cr3.read_range(p, cr3.read_previews(p).full), 1740)
        torch_boxes = model.predict(im, threshold=0.4)
        torch_birds = [
            b for b, c in zip(torch_boxes.xyxy, torch_boxes.class_id, strict=False) if c == 16
        ]
        t0 = time.perf_counter()
        onnx_birds = onnx_det.detect(im)
        dt = (time.perf_counter() - t0) * 1000
        ok = len(torch_birds) == len(onnx_birds) and all(
            float(np.abs(np.array([b.x0, b.y0, b.x1, b.y1]) - tb).max()) < 3.0
            for b, tb in zip(onnx_birds, torch_birds, strict=False)
        )
        agree += ok
        print(
            f"{p.name}: torch={len(torch_birds)} onnx={len(onnx_birds)} "
            f"{'OK' if ok else 'MISMATCH'} ({dt:.0f} ms)"
        )
    print(f"{agree}/10 frames agree")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/bird.onnx")
    args = parser.parse_args()
    export(Path(args.out))
