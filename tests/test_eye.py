import numpy as np

from fovea.core.detect.eye import BINS_PER_AXIS, INPUT_PX, TARGET_SIGMA_BINS, decode_simcc


def gaussian_logits(center_bin: float, sharpness: float = 1.0) -> np.ndarray:
    bins = np.arange(BINS_PER_AXIS, dtype=np.float32)
    return -((bins - center_bin) ** 2) / (2 * TARGET_SIGMA_BINS**2) * sharpness


def test_decode_recovers_coordinates() -> None:
    lx, ly = gaussian_logits(200.0), gaussian_logits(100.0)
    xy, conf = decode_simcc(lx, ly)
    assert abs(xy[0] - 200.0 * INPUT_PX / BINS_PER_AXIS) < 0.5
    assert abs(xy[1] - 100.0 * INPUT_PX / BINS_PER_AXIS) < 0.5
    assert conf > 0.9


def test_uniform_logits_give_low_confidence() -> None:
    flat = np.zeros(BINS_PER_AXIS, dtype=np.float32)
    _, conf = decode_simcc(flat, flat)
    assert conf < 0.1


def test_confidence_clipped_to_one() -> None:
    spike = np.full(BINS_PER_AXIS, -1e9, dtype=np.float32)
    spike[50] = 0.0
    _, conf = decode_simcc(spike, spike)
    assert conf == 1.0
