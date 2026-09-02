import json
from pathlib import Path

import pytest

from fovea.core.metadata.canon_af import _bit_indices, parse_af_frame

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "af_modes.json").read_text())


def test_manual_focus_has_no_points() -> None:
    frame = parse_af_frame(FIXTURES["manual"])
    assert frame is not None
    assert frame.points == []
    assert not frame.lattice


@pytest.mark.parametrize("mode", ["expansion_4pt", "spot", "expansion_8pt", "one_point"])
def test_single_point_modes(mode: str) -> None:
    frame = parse_af_frame(FIXTURES[mode])
    assert frame is not None
    assert len(frame.points) == 1
    assert not frame.lattice
    assert frame.display_points == frame.points


def test_whole_area_locked_coordinates() -> None:
    # IMG_0001: box (783, -381, 174, 174), AF grid 6960x4640 == image dims.
    frame = parse_af_frame(FIXTURES["whole_area_locked"])
    assert frame is not None
    (p,) = frame.points
    assert p.cx == pytest.approx(6960 / 2 + 783)
    assert p.cy == pytest.approx(4640 / 2 + 381)
    assert p.w == pytest.approx(174)
    assert p.in_focus


def test_whole_area_lattice_filters_to_in_focus() -> None:
    frame = parse_af_frame(FIXTURES["whole_area_lattice"])
    assert frame is not None
    assert frame.lattice
    assert len(frame.points) == 651
    shown = frame.display_points
    assert 0 < len(shown) < 651
    assert all(p.in_focus for p in shown)


def test_points_inside_image_bounds() -> None:
    for name, meta in FIXTURES.items():
        frame = parse_af_frame(meta)
        assert frame is not None, name
        for p in frame.points:
            assert 0 <= p.cx <= 6960 and 0 <= p.cy <= 4640, name


def test_bit_indices_multiword() -> None:
    assert _bit_indices([0b1000000000000101, 0b11]) == {0, 2, 15, 16, 17}
    assert _bit_indices([0]) == set()
