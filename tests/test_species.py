import hashlib

import numpy as np
import pytest

from fovea.core.pipeline import PipelineConfig, best_species_frame, species_keywords
from fovea.core.species import download
from fovea.core.species.classify import merge_groups, region_mask


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


def pred(common="Great Egret", scientific="Ardea alba", family="Ardeidae", conf=0.9) -> dict:
    return {"common": common, "scientific": scientific, "family": family, "confidence": conf}


def test_species_keywords_from_confident_prediction() -> None:
    entry = {"species": [pred()]}
    assert species_keywords(entry, PipelineConfig()) == [
        "Nature|Birds|Ardeidae|Great Egret",
        "Ardea alba",
    ]


def test_species_keywords_skips_low_confidence() -> None:
    entry = {"species": [pred(conf=0.3)]}
    assert species_keywords(entry, PipelineConfig()) == []


def test_species_keywords_confirmed_wins_and_ignores_floor() -> None:
    entry = {
        "species": [pred(conf=0.3), pred("Snowy Egret", "Egretta thula", "Ardeidae", 0.2)],
        "user": {"species": "Snowy Egret"},
    }
    assert species_keywords(entry, PipelineConfig()) == [
        "Nature|Birds|Ardeidae|Snowy Egret",
        "Egretta thula",
    ]


def test_species_keywords_without_family_or_common() -> None:
    entry = {"species": [pred(common=None, family=None)]}
    assert species_keywords(entry, PipelineConfig()) == ["Nature|Birds|Ardea alba"]


def test_species_keywords_typed_correction_resolves_labels(tmp_path) -> None:
    labels = tmp_path / "labels.npz"
    np.savez(
        labels,
        common=np.array(["Barred Owl"]),
        scientific=np.array(["Strix varia"]),
        family=np.array(["Strigidae"]),
    )
    entry = {"species": [pred()], "user": {"species": "Barred Owl"}}
    config = PipelineConfig(species_labels=str(labels))
    assert species_keywords(entry, config) == ["Nature|Birds|Strigidae|Barred Owl", "Strix varia"]


def test_download_species_model_verifies_and_installs(tmp_path, monkeypatch) -> None:
    src = tmp_path / "release"
    src.mkdir()
    payload = b"tiny model bytes"
    (src / "species.onnx").write_bytes(payload)
    (src / "species.onnx.data").write_bytes(payload)
    files = [("species.onnx", hashlib.sha256(payload).hexdigest(), len(payload))]
    monkeypatch.setattr(download, "MODEL_FILES", files)
    monkeypatch.setattr(download, "downloaded_dir", lambda: tmp_path / "dest")

    seen = []
    download.download_species_model(lambda d, t: seen.append(d), base_url=src.as_uri() + "/")
    assert (tmp_path / "dest" / "species.onnx").read_bytes() == payload
    assert not list((tmp_path / "dest").glob("*.partial"))
    assert seen[-1] == len(payload)


def test_download_species_model_rejects_bad_checksum(tmp_path, monkeypatch) -> None:
    src = tmp_path / "release"
    src.mkdir()
    (src / "species.onnx").write_bytes(b"tampered")
    monkeypatch.setattr(download, "MODEL_FILES", [("species.onnx", "0" * 64, 8)])
    monkeypatch.setattr(download, "downloaded_dir", lambda: tmp_path / "dest")

    with pytest.raises(ValueError, match="checksum"):
        download.download_species_model(base_url=src.as_uri() + "/")
    assert not (tmp_path / "dest" / "species.onnx").exists()


def test_region_mask_filters_by_code_and_fails_open() -> None:
    regions = np.array(["NA,MA", "AF", "", "SO", "EU,OR"])
    assert region_mask(regions, None).all()
    assert region_mask(regions, "north-america").tolist() == [True, False, True, True, False]
    assert region_mask(regions, "africa").tolist() == [False, True, True, True, False]
    assert region_mask(regions, "south-asia").tolist() == [False, False, True, True, True]
