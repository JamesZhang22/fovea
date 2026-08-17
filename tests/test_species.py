import numpy as np

from fovea.core.pipeline import best_species_frame
from fovea.core.species.classify import merge_groups


def frame(focus=None, eye_conf=None, box_conf=0.9, birds=True) -> dict:
    return {
        "birds": [{"confidence": box_conf}] if birds else [],
        "eye": {"confidence": eye_conf} if eye_conf is not None else None,
        "metrics": {"focus_score": focus} if focus is not None else None,
    }


def test_best_species_frame_prefers_focus_score() -> None:
    burst = [frame(focus=40), frame(focus=90), frame(focus=70)]
    assert best_species_frame(burst) is burst[1]


def test_best_species_frame_falls_back_to_eye_then_box() -> None:
    burst = [frame(eye_conf=0.4), frame(eye_conf=0.8)]
    assert best_species_frame(burst) is burst[1]
    burst = [frame(box_conf=0.5), frame(box_conf=0.95)]
    assert best_species_frame(burst) is burst[1]


def test_best_species_frame_needs_birds() -> None:
    assert best_species_frame([frame(birds=False), frame(birds=False)]) is None


def test_merge_groups_sums_synonym_mass() -> None:
    probs = np.array([0.5, 0.3, 0.2])
    merged = merge_groups(probs, np.array([0, 1, 0]), 2)
    assert np.allclose(merged, [0.7, 0.3])
