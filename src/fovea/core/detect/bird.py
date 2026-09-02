from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image

BIRD_COCO_ID = 16  # original COCO category id for "bird", what predict returns.
DETECT_WIDTH_PX = 1740  # decode width fed to the detector, it resizes internally anyway.
DEFAULT_THRESHOLD = 0.4  # minimum detection confidence.


@dataclass(frozen=True)
class BirdBox:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

    def scaled(self, factor: float) -> BirdBox:
        return BirdBox(
            self.x0 * factor,
            self.y0 * factor,
            self.x1 * factor,
            self.y1 * factor,
            self.confidence,
        )


class BirdDetector:
    """RF-DETR-Nano filtered to COCO birds, box coordinates in input-image pixels.

    predict returns original COCO category ids, bird is 16 there.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        # deferred: rfdetr pulls torch, which must never be a runtime import.
        from rfdetr import RFDETRNano

        self.model = RFDETRNano()
        self.threshold = threshold

    def detect(self, im: Image.Image) -> list[BirdBox]:
        det = self.model.predict(im, threshold=self.threshold)
        return [
            BirdBox(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
            for b, cls, c in zip(det.xyxy, det.class_id, det.confidence, strict=True)
            if cls == BIRD_COCO_ID
        ]


class OnnxBirdDetector:
    """Same detections through onnxruntime, what the packaged app ships instead of torch."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_path, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_px = self.session.get_inputs()[0].shape[2]
        self.threshold = threshold

    def _resize_no_antialias(self, im: Image.Image):
        """Plain bilinear sampling matching torch resize(antialias=False) at export time."""
        src = np.asarray(im, dtype=np.float32)
        h, w = src.shape[:2]
        n = self.input_px
        # torchvision maps output centers to input coords as (i + 0.5) * scale - 0.5.
        ys = np.clip((np.arange(n) + 0.5) * (h / n) - 0.5, 0, h - 1)
        xs = np.clip((np.arange(n) + 0.5) * (w / n) - 0.5, 0, w - 1)
        y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
        y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
        wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
        return (
            src[y0][:, x0] * (1 - wy) * (1 - wx)
            + src[y0][:, x1] * (1 - wy) * wx
            + src[y1][:, x0] * wy * (1 - wx)
            + src[y1][:, x1] * wy * wx
        )

    def detect(self, im: Image.Image) -> list[BirdBox]:
        arr = self._resize_no_antialias(im) / 255.0
        arr = (arr - self.IMAGENET_MEAN) / self.IMAGENET_STD
        arr = arr.transpose(2, 0, 1)[None].astype(np.float32)

        dets, labels = self.session.run(None, {self.input_name: arr})
        scores = 1.0 / (1.0 + np.exp(-labels[0]))  # sigmoid over class logits.
        bird_scores = scores[:, BIRD_COCO_ID]
        keep = bird_scores > self.threshold

        boxes = []
        for (cx, cy, w, h), score in zip(dets[0][keep], bird_scores[keep], strict=True):
            boxes.append(
                BirdBox(
                    float((cx - w / 2) * im.width),
                    float((cy - h / 2) * im.height),
                    float((cx + w / 2) * im.width),
                    float((cy + h / 2) * im.height),
                    float(score),
                )
            )
        return sorted(boxes, key=lambda b: -b.confidence)
