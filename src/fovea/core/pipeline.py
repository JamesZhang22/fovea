import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fovea.core.export.xmp import Sidecar, sidecar_path, write_sidecar
from fovea.core.group.burst import group_bursts
from fovea.core.ingest import cr3, decode
from fovea.core.ingest.cache import Cache
from fovea.core.scan import scan_folder
from fovea.core.score.classical import metrics, rank_burst, to_gray

RATING_BEST = 4
RATING_TOP = 3
MIN_PATCH = 256


@dataclass
class PipelineConfig:
    group: bool = True
    score: bool = True
    export: bool = True
    gap_seconds: float = 2.0
    decode_width: int = 3480
    metric: str = "brenner"
    top_rank: float = 0.8
    workers: int = 10


def run_pipeline(folder: Path, config: PipelineConfig, cache: Cache | None = None) -> list[dict]:
    """Scan then run the enabled stages, returns entries annotated in place."""
    entries = scan_folder(folder, cache)

    bursts = group_bursts(entries, config.gap_seconds) if config.group else [[e] for e in entries]
    for i, burst in enumerate(bursts):
        for e in burst:
            e["burst"] = i
            e["burst_size"] = len(burst)

    if config.score:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            scored = list(pool.map(lambda e: _score_entry(e, config, cache), entries))
        for e, m in zip(entries, scored, strict=True):
            e["metrics"] = m
        for burst in bursts:
            ranks = rank_burst([b["metrics"][config.metric] for b in burst if b["metrics"]])
            for e, r in zip([b for b in burst if b["metrics"]], ranks, strict=True):
                e["rank"] = r

    if config.export:
        written = skipped = 0
        for e in entries:
            if _export_entry(e, config):
                written += 1
            else:
                skipped += 1
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
    """Where to measure focus, the in-focus AF region when present, else the center half."""
    img_w = entry["meta"].get("ImageWidth") or width
    scale = width / img_w
    af = entry.get("af")
    points = [p for p in (af["display_points"] if af else []) if p["in_focus"]]
    if points:
        x0 = min(p["cx"] - p["w"] / 2 for p in points) * scale
        y0 = min(p["cy"] - p["h"] / 2 for p in points) * scale
        x1 = max(p["cx"] + p["w"] / 2 for p in points) * scale
        y1 = max(p["cy"] + p["h"] / 2 for p in points) * scale
    else:
        x0, y0, x1, y1 = width / 4, height / 4, 3 * width / 4, 3 * height / 4

    pad_x = max(0.0, MIN_PATCH - (x1 - x0)) / 2
    pad_y = max(0.0, MIN_PATCH - (y1 - y0)) / 2
    return (
        max(0, int(x0 - pad_x)),
        max(0, int(y0 - pad_y)),
        min(width, int(x1 + pad_x)),
        min(height, int(y1 + pad_y)),
    )


def _score_entry(entry: dict, config: PipelineConfig, cache: Cache | None) -> dict | None:
    path = Path(entry["path"])
    kind = f"score:{config.decode_width}"
    if cache and (cached := cache.get_json(path, kind)) is not None:
        return cached
    previews = cr3.read_previews(path)
    src = previews.full or previews.prvw
    if src is None:
        return None
    im = decode.decode_scaled(cr3.read_range(path, src), config.decode_width)
    patch = im.crop(score_patch_box(entry, im.width, im.height))
    m = metrics(to_gray(patch))
    if cache:
        cache.put_json(path, kind, m)
    return m


def _export_entry(entry: dict, config: PipelineConfig) -> bool:
    """Write one sidecar, never overwrites an existing one, returns True if written."""
    path = Path(entry["path"])
    if sidecar_path(path).exists():
        return False
    fovea: dict[str, object] = {"BurstId": entry.get("burst")}
    if entry.get("metrics"):
        fovea["FocusMetric"] = round(entry["metrics"][config.metric], 1)
    if entry.get("rank") is not None:
        fovea["BurstRank"] = round(entry["rank"], 3)
    rating = rating_for(entry.get("rank"), entry.get("burst_size", 1), config.top_rank)
    write_sidecar(path, Sidecar(rating=rating, fovea=fovea))
    return True
