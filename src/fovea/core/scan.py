from dataclasses import asdict
from pathlib import Path

from fovea.core.ingest.cache import Cache
from fovea.core.metadata.canon_af import parse_af_frame
from fovea.core.metadata.exiftool import ExifTool

BATCH = 200


def find_cr3s(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".cr3")


def scan_folder(folder: Path, cache: Cache | None = None) -> list[dict]:
    paths = find_cr3s(folder)
    entries: list[dict] = []
    missing: list[Path] = []

    for p in paths:
        cached = cache.get_json(p, "meta") if cache else None
        if cached is not None:
            entries.append(cached)
        else:
            missing.append(p)

    if missing:
        with ExifTool() as et:
            for i in range(0, len(missing), BATCH):
                for meta in et.metadata(missing[i : i + BATCH]):
                    entry = build_entry(meta)
                    entries.append(entry)
                    if cache:
                        cache.put_json(Path(entry["path"]), "meta", entry)

    entries.sort(key=lambda e: e["path"])
    return entries


def build_entry(meta: dict) -> dict:
    frame = parse_af_frame(meta)
    return {
        "path": meta["SourceFile"],
        "meta": {k: v for k, v in meta.items() if k not in ("SourceFile",)},
        "af": None
        if frame is None
        else {
            "lattice": frame.lattice,
            "orientation": frame.orientation,
            "n_points": len(frame.points),
            "display_points": [asdict(p) for p in frame.display_points],
        },
    }
