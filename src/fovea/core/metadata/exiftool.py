import json
import subprocess
from pathlib import Path
from types import TracebackType

TAGS = [
    "SourceFile",
    "FileName",
    "DateTimeOriginal",
    "SubSecTimeOriginal",
    "ExposureTime",
    "FNumber",
    "ISO",
    "FocalLength",
    "Orientation",
    "ImageWidth",
    "ImageHeight",
    "FocusMode",
    "ContinuousDrive",
    "AFAreaMode",
    "NumAFPoints",
    "ValidAFPoints",
    "AFImageWidth",
    "AFImageHeight",
    "AFAreaXPositions",
    "AFAreaYPositions",
    "AFAreaWidths",
    "AFAreaHeights",
    "AFPointsInFocus",
    "AFPointsSelected",
    "SubjectToDetect",
    "EyeDetection",
]

SENTINEL = b"{ready}"  # exiftool prints this after each -execute completes

# Finder-launched apps get a minimal PATH, so check Homebrew locations explicitly
EXIFTOOL_FALLBACKS = ("/opt/homebrew/bin/exiftool", "/usr/local/bin/exiftool")


def find_exiftool() -> str:
    import shutil

    found = shutil.which("exiftool")
    if found:
        return found
    for candidate in EXIFTOOL_FALLBACKS:
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError("exiftool not found, install it with: brew install exiftool")


class ExifTool:
    """Resident `exiftool -stay_open` process."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [find_exiftool(), "-stay_open", "True", "-@", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def execute(self, *args: str) -> bytes:
        """Send one command to the resident process and read output up to the {ready} sentinel."""
        assert self.proc.stdin and self.proc.stdout
        payload = "\n".join([*args, "-execute\n"]).encode()
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        out = b""
        while not out.rstrip().endswith(SENTINEL):
            chunk = self.proc.stdout.read1(65536)
            if not chunk:
                raise RuntimeError("exiftool died")
            out += chunk
        return out.rstrip()[: -len(SENTINEL)]

    def metadata(self, paths: list[Path], tags: list[str] = TAGS) -> list[dict]:
        """Batch-read the given tags for many files in a single exiftool invocation."""
        if not paths:
            return []
        args = ["-j", "-n", *[f"-{t}" for t in tags], *[str(p) for p in paths]]
        out = self.execute(*args)
        return json.loads(out) if out.strip() else []

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.write(b"-stay_open\nFalse\n")
            self.proc.stdin.flush()
        self.proc.wait(timeout=5)

    def __enter__(self) -> ExifTool:
        return self

    def __exit__(
        self, t: type[BaseException] | None, v: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
