import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fovea.core.export.xmp import Sidecar, is_fovea_sidecar, sidecar_path, write_sidecar
from fovea.core.group.burst import group_bursts
from fovea.core.ingest import cr3, decode
from fovea.core.ingest.cache import Cache
from fovea.core.scan import scan_folder
from fovea.core.score.blurtype import spectral_anisotropy
from fovea.core.score.classical import metrics, rank_burst, to_gray

RATING_BEST = 4
RATING_TOP = 3
MIN_PATCH_PX = 256  # focus metrics get unstable below this patch size
EYE_PATCH_FRACTION = 0.25  # eye patch side as a fraction of the bird box side
EYE_MIN_CONFIDENCE = 0.5  # from the val error-vs-confidence curve, below this predictions are junk
MIN_EYE_BOX_PX = 400  # bird box below this means the eye is too small to score honestly
SPECIES_KEYWORD_MIN_CONFIDENCE = 0.5  # unconfirmed predictions below this stay out of sidecars


@dataclass
class PipelineConfig:
    group: bool = True
    score: bool = True
    detect: bool = False
    eye: bool = False
    species: bool = False
    export: bool = True
    gap_seconds: float = 3.0
    decode_width: int = 3480
    detect_threshold: float = 0.4
    detect_model: str = "models/bird.onnx"
    eye_model: str = "models/eye.onnx"
    focus_model: str = "models/focus.onnx"
    species_model: str = "models/species.onnx"
    species_labels: str = "models/species_labels.npz"
    calibration_path: str | None = None  # None keeps percentiles seed-only, no persistence
    metric: str = "brenner"
    top_rank: float = 0.8
    workers: int = 10


Progress = Callable[[str, int, int], None]


def run_pipeline(
    folder: Path,
    config: PipelineConfig,
    cache: Cache | None = None,
    progress: Progress | None = None,
) -> list[dict]:
    """Scan then run the enabled stages, returns entries annotated in place."""
    tick = progress or (lambda stage, done, total: None)
    tick("scan", 0, 0)
    entries = scan_folder(folder, cache)
    tick("scan", len(entries), len(entries))

    bursts = group_bursts(entries, config.gap_seconds) if config.group else [[e] for e in entries]
    for i, burst in enumerate(bursts):
        for e in burst:
            e["burst"] = i
            e["burst_size"] = len(burst)

    if config.detect:
        _detect_stage(entries, config, cache, tick)

    if config.eye:
        _eye_stage(entries, config, cache, tick)

    if config.score:
        from fovea.core.score.calibrate import Calibration

        scorer = None
        focus_path = Path(config.focus_model)
        if focus_path.exists():
            from fovea.core.score.model import FocusScorer

            scorer = FocusScorer(focus_path)
        calibration = Calibration(
            Path(config.calibration_path) if config.calibration_path else None
        )
        scored = []
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            for i, m in enumerate(
                pool.map(lambda e: _score_entry(e, config, cache, scorer, calibration), entries)
            ):
                scored.append(m)
                tick("score", i + 1, len(entries))
        for e, m in zip(entries, scored, strict=True):
            if m and m.get("focus_radius_px") is not None:
                m["focus_percentile"] = calibration.percentile(m["focus_radius_px"])
            e["metrics"] = m
        calibration.save()
        for burst in bursts:
            ranks = rank_burst([b["metrics"][config.metric] for b in burst if b["metrics"]])
            for e, r in zip([b for b in burst if b["metrics"]], ranks, strict=True):
                e["rank"] = r

    if config.species:
        _species_stage(bursts, config, cache, tick)

    if config.export:
        written = skipped = 0
        for i, e in enumerate(entries):
            if _export_entry(e, config):
                written += 1
            else:
                skipped += 1
            tick("export", i + 1, len(entries))
        print(f"sidecars: {written} written, {skipped} skipped (existing)", file=sys.stderr)

    return entries


def rating_for(rank: float | None, burst_size: int, top_rank: float) -> int | None:
    """Burst best gets RATING_BEST, top fraction RATING_TOP, singletons stay unrated."""
    if rank is None or burst_size < 2:
        return None
    if rank == 1.0:
        return RATING_BEST
    if rank >= top_rank:
        return RATING_TOP
    return None


def score_patch_box(entry: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """Where to measure focus: detected eye first, else in-focus AF region, else center half."""
    img_w = entry["meta"].get("ImageWidth") or width
    scale = width / img_w
    eye = entry.get("eye")
    if eye and eye["confidence"] >= EYE_MIN_CONFIDENCE and entry.get("birds"):
        box = max(entry["birds"], key=lambda b: b["confidence"])
        side = max(box["x1"] - box["x0"], box["y1"] - box["y0"]) * EYE_PATCH_FRACTION * scale
        half = side / 2
        x0, y0 = eye["x"] * scale - half, eye["y"] * scale - half
        x1, y1 = eye["x"] * scale + half, eye["y"] * scale + half
        return _clamp_padded(x0, y0, x1, y1, width, height)
    af = entry.get("af")
    points = [p for p in (af["display_points"] if af else []) if p["in_focus"]]
    if points:
        x0 = min(p["cx"] - p["w"] / 2 for p in points) * scale
        y0 = min(p["cy"] - p["h"] / 2 for p in points) * scale
        x1 = max(p["cx"] + p["w"] / 2 for p in points) * scale
        y1 = max(p["cy"] + p["h"] / 2 for p in points) * scale
    else:
        x0, y0, x1, y1 = width / 4, height / 4, 3 * width / 4, 3 * height / 4
    return _clamp_padded(x0, y0, x1, y1, width, height)


def _clamp_padded(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Pad up to MIN_PATCH_PX and clamp to the image."""
    pad_x = max(0.0, MIN_PATCH_PX - (x1 - x0)) / 2
    pad_y = max(0.0, MIN_PATCH_PX - (y1 - y0)) / 2
    return (
        max(0, int(x0 - pad_x)),
        max(0, int(y0 - pad_y)),
        min(width, int(x1 + pad_x)),
        min(height, int(y1 + pad_y)),
    )


def _detect_stage(
    entries: list[dict], config: PipelineConfig, cache: Cache | None, tick: Progress
) -> None:
    """Annotate entries with bird boxes in full-resolution pixel coordinates."""
    from dataclasses import asdict

    from fovea.core.detect.bird import DETECT_WIDTH_PX, BirdDetector, OnnxBirdDetector

    detector = None
    kind = f"birds:{config.detect_threshold}"
    for n, e in enumerate(entries):
        tick("detect", n + 1, len(entries))
        path = Path(e["path"])
        if cache and (cached := cache.get_json(path, kind)) is not None:
            e["birds"] = cached["boxes"]
            continue
        if detector is None:
            onnx_model = Path(config.detect_model)
            if onnx_model.exists():
                detector = OnnxBirdDetector(onnx_model, config.detect_threshold)
            else:
                detector = BirdDetector(config.detect_threshold)
        previews = cr3.read_previews(path)
        src = previews.full or previews.prvw
        if src is None:
            e["birds"] = []
            continue
        im = decode.decode_scaled(cr3.read_range(path, src), DETECT_WIDTH_PX)
        factor = (e["meta"].get("ImageWidth") or im.width) / im.width
        e["birds"] = [asdict(b.scaled(factor)) for b in detector.detect(im)]
        if cache:
            cache.put_json(path, kind, {"boxes": e["birds"]})


def _eye_stage(
    entries: list[dict], config: PipelineConfig, cache: Cache | None, tick: Progress
) -> None:
    """Locate the eye of each entry's most confident bird box."""
    from fovea.core.detect.eye import EyeLocator

    model_path = Path(config.eye_model)
    locator = None
    kind = f"eye:{model_path.stat().st_mtime_ns}"
    for n, e in enumerate(entries):
        tick("eye", n + 1, len(entries))
        if not e.get("birds"):
            continue
        path = Path(e["path"])
        if cache and (cached := cache.get_json(path, kind)) is not None:
            e["eye"] = cached["eye"]
            continue
        if locator is None:
            locator = EyeLocator(model_path)
        previews = cr3.read_previews(path)
        if previews.full is None:
            continue
        jpeg = cr3.read_range(path, previews.full)
        best_box = max(e["birds"], key=lambda b: b["confidence"])
        e["eye"] = locator.locate(jpeg, best_box)
        if cache:
            cache.put_json(path, kind, {"eye": e["eye"]})


def best_species_frame(burst: list[dict]) -> dict | None:
    """Frame to classify: sharpest trusted eye, then eye confidence, then box confidence."""
    candidates = [e for e in burst if e.get("birds")]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda e: (
            (e.get("metrics") or {}).get("focus_score") or -1.0,
            (e.get("eye") or {}).get("confidence") or 0.0,
            max(b["confidence"] for b in e["birds"]),
        ),
    )


def _species_stage(
    bursts: list[list[dict]], config: PipelineConfig, cache: Cache | None, tick: Progress
) -> None:
    """Classify each burst's best frame once, annotate every frame of the burst."""
    from fovea.core.species.classify import CROP_PAD_FRACTION, SpeciesClassifier

    model_path = Path(config.species_model)
    labels_path = Path(config.species_labels)
    if not model_path.exists() or not labels_path.exists():
        return
    kind = f"species:{model_path.stat().st_mtime_ns}:{labels_path.stat().st_mtime_ns}"
    classifier = None
    for n, burst in enumerate(bursts):
        tick("species", n + 1, len(bursts))
        best = best_species_frame(burst)
        if best is None:
            continue
        path = Path(best["path"])
        preds = (cache.get_json(path, kind) or {}).get("predictions") if cache else None
        if preds is None:
            if classifier is None:
                classifier = SpeciesClassifier(model_path, labels_path)
            previews = cr3.read_previews(path)
            src = previews.full or previews.prvw
            if src is None:
                continue
            im = decode.decode_scaled(cr3.read_range(path, src), config.decode_width)
            scale = im.width / (best["meta"].get("ImageWidth") or im.width)
            box = max(best["birds"], key=lambda b: b["confidence"])
            w, h = box["x1"] - box["x0"], box["y1"] - box["y0"]
            crop = im.crop(
                (
                    max(0, int((box["x0"] - CROP_PAD_FRACTION * w) * scale)),
                    max(0, int((box["y0"] - CROP_PAD_FRACTION * h) * scale)),
                    min(im.width, int((box["x1"] + CROP_PAD_FRACTION * w) * scale)),
                    min(im.height, int((box["y1"] + CROP_PAD_FRACTION * h) * scale)),
                )
            )
            preds = classifier.classify(crop)
            if cache:
                cache.put_json(path, kind, {"predictions": preds})
        for e in burst:
            e["species"] = preds


def _score_entry(
    entry: dict, config: PipelineConfig, cache: Cache | None, scorer=None, calibration=None
) -> dict | None:
    path = Path(entry["path"])
    eye = entry.get("eye")
    eye_used = bool(eye and eye["confidence"] >= EYE_MIN_CONFIDENCE and entry.get("birds"))
    kind = f"score3:{config.decode_width}:{'eye' if eye_used else 'std'}"
    if cache and (cached := cache.get_json(path, kind)) is not None:
        cached.pop("focus_percentile", None)
        return cached
    previews = cr3.read_previews(path)
    src = previews.full or previews.prvw
    if src is None:
        return None
    jpeg = cr3.read_range(path, src)
    im = decode.decode_scaled(jpeg, config.decode_width)
    patch = im.crop(score_patch_box(entry, im.width, im.height))
    gray = to_gray(patch)
    m = metrics(gray)
    aniso, angle = spectral_anisotropy(gray)
    m["anisotropy"] = round(aniso, 3)
    m["motion_angle"] = round(angle, 1)
    if scorer is not None and eye_used:
        m.update(_focus_score(entry, jpeg, scorer))
        if calibration is not None and m.get("focus_radius_px") is not None:
            calibration.record(m["focus_radius_px"])  # newly computed only, cache hits skip
    if cache:
        cache.put_json(path, kind, m)
    return m


def _focus_score(entry: dict, jpeg: bytes, scorer) -> dict:
    """Ordinal model on the native-res eye patch, abstains on tiny eyes."""
    import numpy as np

    from fovea.core.score.model import PATCH_PX

    box = max(entry["birds"], key=lambda b: b["confidence"])
    box_px = max(box["x1"] - box["x0"], box["y1"] - box["y0"])
    if box_px < MIN_EYE_BOX_PX:
        return {"focus_score": None, "focus_confidence": 0.0}
    eye = entry["eye"]
    half = PATCH_PX // 2
    region = (
        int(eye["x"] - half),
        int(eye["y"] - half),
        int(eye["x"] + half),
        int(eye["y"] + half),
    )
    patch = decode.roi_native(jpeg, region).convert("L")
    if patch.size != (PATCH_PX, PATCH_PX):
        return {"focus_score": None, "focus_confidence": 0.0}
    result = scorer.score(np.asarray(patch, dtype=np.float32))
    return {
        "focus_radius_px": result["radius_px"],
        "focus_score": result["score"],
        "focus_confidence": result["confidence"],
    }


def _export_entry(entry: dict, config: PipelineConfig) -> bool:
    """Write one sidecar, never overwrites an existing one, returns True if written."""
    path = Path(entry["path"])
    if sidecar_path(path).exists():
        return False
    rating = rating_for(entry.get("rank"), entry.get("burst_size", 1), config.top_rank)
    sidecar = Sidecar(
        rating=rating,
        keywords=species_keywords(entry, config),
        fovea=_fovea_fields(entry, config),
    )
    write_sidecar(path, sidecar)
    return True


def species_keywords(entry: dict, config: PipelineConfig) -> list[str]:
    """Nature|Birds|Family|Common hierarchy plus the scientific name as a flat keyword.

    A user-confirmed species always exports, an unconfirmed prediction only above the
    confidence floor. Typed corrections outside the top-3 resolve via the labels file.
    """
    confirmed = (entry.get("user") or {}).get("species")
    preds = entry.get("species") or []
    if confirmed:
        match = next((p for p in preds if (p["common"] or p["scientific"]) == confirmed), None)
        if match:
            name, scientific, family = confirmed, match["scientific"], match["family"]
        else:
            labels = Path(config.species_labels)
            scientific, family = (None, None)
            if labels.exists():
                from fovea.core.species.classify import species_info

                scientific, family = species_info(labels, confirmed)
            name = confirmed
    elif preds and preds[0]["confidence"] >= SPECIES_KEYWORD_MIN_CONFIDENCE:
        p = preds[0]
        name, scientific, family = p["common"] or p["scientific"], p["scientific"], p["family"]
    else:
        return []
    keywords = ["|".join(["Nature", "Birds"] + ([family] if family else []) + [name])]
    if scientific and scientific != name:
        keywords.append(scientific)
    return keywords


def _fovea_fields(entry: dict, config: PipelineConfig) -> dict[str, object]:
    fovea: dict[str, object] = {"BurstId": entry.get("burst")}
    if entry.get("metrics"):
        fovea["FocusMetric"] = round(entry["metrics"][config.metric], 1)
        if entry["metrics"].get("focus_score") is not None:
            fovea["FocusScore"] = entry["metrics"]["focus_score"]
            fovea["BlurRadiusPx"] = entry["metrics"]["focus_radius_px"]
    if entry.get("rank") is not None:
        fovea["BurstRank"] = round(entry["rank"], 3)
    if entry.get("eye"):
        fovea["EyeConfidence"] = entry["eye"]["confidence"]
    return fovea


def export_verdicts(entries: list[dict], config: PipelineConfig) -> tuple[int, int]:
    """Write sidecars carrying the user's verdicts, falling back to suggested ratings."""
    written = skipped = 0
    for entry in entries:
        path = Path(entry["path"])
        target = sidecar_path(path)
        if target.exists() and not is_fovea_sidecar(target):
            skipped += 1
            continue
        user = entry.get("user") or {}
        rejected = bool(user.get("rejected"))
        if user.get("rating") is not None:
            rating = user["rating"]
        elif rejected:
            rating = None
        else:
            rating = rating_for(entry.get("rank"), entry.get("burst_size", 1), config.top_rank)
        sidecar = Sidecar(
            rating=rating,
            rejected=rejected,
            keywords=species_keywords(entry, config),
            fovea=_fovea_fields(entry, config),
        )
        write_sidecar(path, sidecar)
        written += 1
    return written, skipped
