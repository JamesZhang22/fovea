from dataclasses import dataclass

from PIL import Image

BIRD_COCO_ID = 16  # original COCO category id for "bird", what predict returns
DETECT_WIDTH_PX = 1740  # decode width fed to the detector, it resizes internally anyway
DEFAULT_THRESHOLD = 0.4  # minimum detection confidence


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
