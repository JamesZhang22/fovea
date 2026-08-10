import argparse

from fovea import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="fovea", description="Bird photo culling for macOS")
    parser.add_argument("--version", action="version", version=f"fovea {__version__}")
    parser.parse_args()


if __name__ == "__main__":
    main()
