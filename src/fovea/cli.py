import argparse
import json
import sys
from pathlib import Path

from fovea import __version__


def cmd_scan(args: argparse.Namespace) -> None:
    from fovea.core.ingest.cache import Cache
    from fovea.core.scan import scan_folder

    folder = Path(args.folder).expanduser().resolve()
    cache = None if args.no_cache else Cache(folder / ".fovea" / "cache.sqlite")
    entries = scan_folder(folder, cache)
    out = json.dumps(entries, indent=None if args.compact else 2)
    if args.out:
        Path(args.out).write_text(out)
        print(f"{len(entries)} files -> {args.out}", file=sys.stderr)
    else:
        print(out)


def cmd_contactsheet(args: argparse.Namespace) -> None:
    from fovea.core.contactsheet import write_contactsheet
    from fovea.core.ingest.cache import Cache
    from fovea.core.scan import scan_folder

    folder = Path(args.folder).expanduser().resolve()
    cache = None if args.no_cache else Cache(folder / ".fovea" / "cache.sqlite")
    entries = scan_folder(folder, cache)
    out_dir = Path(args.out) if args.out else folder / ".fovea" / "contactsheet"
    page = write_contactsheet(entries, out_dir)
    print(f"{len(entries)} files -> {page}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fovea", description="Bird photo culling for macOS")
    parser.add_argument("--version", action="version", version=f"fovea {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan a folder of CR3s to a JSON manifest")
    scan.add_argument("folder")
    scan.add_argument("-o", "--out", help="write manifest to file instead of stdout")
    scan.add_argument("--compact", action="store_true")
    scan.add_argument("--no-cache", action="store_true")
    scan.set_defaults(func=cmd_scan)

    sheet = sub.add_parser("contactsheet", help="render an HTML contact sheet with AF overlays")
    sheet.add_argument("folder")
    sheet.add_argument("-o", "--out", help="output directory (default <folder>/.fovea/contactsheet)")
    sheet.add_argument("--no-cache", action="store_true")
    sheet.set_defaults(func=cmd_contactsheet)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
