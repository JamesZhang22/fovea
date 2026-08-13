from pathlib import Path

from fovea.core.score.calibrate import Calibration


def test_seed_percentiles_are_monotonic() -> None:
    cal = Calibration()
    ps = [cal.percentile(r) for r in (0.2, 1.0, 3.0, 6.0, 11.0)]
    assert all(a > b for a, b in zip(ps, ps[1:], strict=False))
    assert ps[0] > 0.85 and ps[-1] < 0.1


def test_recording_shifts_percentiles() -> None:
    cal = Calibration()
    before = cal.percentile(2.0)
    for _ in range(500):
        cal.record(0.5)
    assert cal.percentile(2.0) < before


def test_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    cal = Calibration(path)
    for _ in range(100):
        cal.record(4.0)
    cal.save()

    fresh = Calibration(path)
    assert fresh.percentile(2.0) > Calibration().percentile(2.0)
