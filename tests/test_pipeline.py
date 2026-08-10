from fovea.core.pipeline import RATING_BEST, RATING_TOP, rating_for, score_patch_box


def test_rating_for() -> None:
    assert rating_for(1.0, burst_size=10, top_rank=0.8) == RATING_BEST
    assert rating_for(0.85, burst_size=10, top_rank=0.8) == RATING_TOP
    assert rating_for(0.5, burst_size=10, top_rank=0.8) is None
    assert rating_for(1.0, burst_size=1, top_rank=0.8) is None
    assert rating_for(None, burst_size=10, top_rank=0.8) is None


def af_entry(points: list[dict], img_w: int = 6960) -> dict:
    return {"meta": {"ImageWidth": img_w}, "af": {"display_points": points}}


def test_score_patch_box_uses_in_focus_af_region() -> None:
    entry = af_entry([{"cx": 3480, "cy": 2320, "w": 800, "h": 800, "in_focus": True}])
    x0, y0, x1, y1 = score_patch_box(entry, 3480, 2320)
    assert (x0, y0, x1, y1) == (1540, 960, 1940, 1360)


def test_score_patch_box_pads_small_af_boxes() -> None:
    entry = af_entry([{"cx": 3480, "cy": 2320, "w": 100, "h": 100, "in_focus": True}])
    x0, y0, x1, y1 = score_patch_box(entry, 3480, 2320)
    assert x1 - x0 >= 256 and y1 - y0 >= 256


def test_score_patch_box_falls_back_to_center() -> None:
    entry = {"meta": {"ImageWidth": 6960}, "af": None}
    assert score_patch_box(entry, 1000, 800) == (250, 200, 750, 600)


def test_score_patch_box_clamps_to_image() -> None:
    entry = af_entry([{"cx": 50, "cy": 50, "w": 400, "h": 400, "in_focus": True}])
    x0, y0, x1, y1 = score_patch_box(entry, 3480, 2320)
    assert x0 == 0 and y0 == 0 and x1 <= 3480 and y1 <= 2320


def test_birdbox_scaled() -> None:
    from fovea.core.detect.bird import BirdBox

    b = BirdBox(10.0, 20.0, 110.0, 220.0, 0.9).scaled(2.0)
    assert (b.x0, b.y0, b.x1, b.y1, b.confidence) == (20.0, 40.0, 220.0, 440.0, 0.9)
