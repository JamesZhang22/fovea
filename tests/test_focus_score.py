import numpy as np

from fovea.core.score.model import decode_ordinal, index_to_radius, radius_to_score


def test_curve_endpoints_and_monotonic() -> None:
    assert radius_to_score(0.0) == 100.0
    assert radius_to_score(1.0) == 85.0
    assert radius_to_score(12.0) == 0.0
    assert radius_to_score(20.0) == 0.0
    scores = [radius_to_score(r) for r in np.linspace(0, 12, 50)]
    assert all(a >= b for a, b in zip(scores, scores[1:], strict=False))


def test_decode_ordinal_peak_and_confidence() -> None:
    sharp = np.full(10, -20.0, dtype=np.float32)
    sharp[0] = 5.0
    radius, conf = decode_ordinal(sharp)
    assert radius < 0.1 and conf > 0.9

    flat = np.zeros(10, dtype=np.float32)
    radius, conf = decode_ordinal(flat)
    assert conf < 0.05


def test_index_to_radius_interpolates() -> None:
    assert float(index_to_radius(np.array([0.0]))[0]) == 0.0
    assert float(index_to_radius(np.array([1.5]))[0]) == 0.75
    assert float(index_to_radius(np.array([9.0]))[0]) == 12.0
