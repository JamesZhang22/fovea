import argparse
import json
import sys
from pathlib import Path

from fovea import __version__


def _folder_and_cache(args: argparse.Namespace):
    from fovea.core.ingest.cache import Cache

    folder = Path(args.folder).expanduser().resolve()
    cache = None if args.no_cache else Cache(folder / ".fovea" / "cache.sqlite")
    return folder, cache


def cmd_scan(args: argparse.Namespace) -> None:
    from fovea.core.scan import scan_folder

    folder, cache = _folder_and_cache(args)
    entries = scan_folder(folder, cache)
    out = json.dumps(entries, indent=None if args.compact else 2)
    if args.out:
        Path(args.out).write_text(out)
        print(f"{len(entries)} files -> {args.out}", file=sys.stderr)
    else:
        print(out)


def add_scan(sub) -> None:
    p = sub.add_parser("scan", help="scan a folder of CR3s to a JSON manifest")
    p.add_argument("folder")
    p.add_argument("-o", "--out", help="write manifest to file instead of stdout")
    p.add_argument("--compact", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_scan)


def cmd_cull(args: argparse.Namespace) -> None:
    from fovea.core.pipeline import PipelineConfig, run_pipeline

    folder, cache = _folder_and_cache(args)
    from fovea.core.score.calibrate import default_calibration_path

    config = PipelineConfig(
        group=not args.no_group,
        score=not args.no_score,
        detect=args.detect or args.eye or args.species,
        eye=args.eye,
        species=args.species,
        species_region=args.region,
        export=not args.no_export,
        gap_seconds=args.gap,
        metric=args.metric,
        calibration_path=str(default_calibration_path()),
    )
    entries = run_pipeline(folder, config, cache)
    rated = sum(1 for e in entries if e.get("rank") == 1.0 and e.get("burst_size", 1) > 1)
    bursts = len({e.get("burst") for e in entries})
    print(f"{len(entries)} files, {bursts} bursts, {rated} burst-best frames", file=sys.stderr)


def add_cull(sub) -> None:
    p = sub.add_parser("cull", help="score and rank a folder, write XMP sidecars for LrC import")
    p.add_argument("folder")
    p.add_argument("--gap", type=float, default=3.0, help="burst gap in seconds")
    p.add_argument(
        "--metric", default="brenner", choices=["brenner", "tenengrad", "edge_sharpness"]
    )
    p.add_argument("--detect", action="store_true", help="run bird detection (needs ml deps)")
    p.add_argument(
        "--eye", action="store_true", help="locate eyes and score them (implies --detect)"
    )
    p.add_argument(
        "--species", action="store_true", help="identify species per burst (implies --detect)"
    )
    p.add_argument(
        "--region",
        # keys of core.species.classify.REGIONS, not imported to keep CLI startup light
        choices=["north-america", "south-america", "eurasia", "africa", "south-asia", "australasia"],
        help="restrict species candidates to a continent (default: all)",
    )
    p.add_argument("--no-group", action="store_true", help="skip burst grouping")
    p.add_argument("--no-score", action="store_true", help="skip focus scoring")
    p.add_argument("--no-export", action="store_true", help="skip writing sidecars")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_cull)


def cmd_contactsheet(args: argparse.Namespace) -> None:
    from fovea.core.contactsheet import write_contactsheet
    from fovea.core.scan import scan_folder

    folder, cache = _folder_and_cache(args)
    entries = scan_folder(folder, cache)
    out_dir = Path(args.out) if args.out else folder / ".fovea" / "contactsheet"
    page = write_contactsheet(entries, out_dir)
    print(f"{len(entries)} files -> {page}", file=sys.stderr)


def add_contactsheet(sub) -> None:
    p = sub.add_parser("contactsheet", help="render an HTML contact sheet with AF overlays")
    p.add_argument("folder")
    p.add_argument("-o", "--out", help="output dir (default <folder>/.fovea/contactsheet)")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_contactsheet)


def cmd_app(args: argparse.Namespace) -> None:
    from fovea.app.main import main as run_app

    run_app()


def add_app(sub) -> None:
    p = sub.add_parser("app", help="launch the desktop app window")
    p.set_defaults(func=cmd_app)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fovea", description="Bird photo culling for macOS")
    parser.add_argument("--version", action="version", version=f"fovea {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for add in (add_scan, add_cull, add_contactsheet, add_app):
        add(sub)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
