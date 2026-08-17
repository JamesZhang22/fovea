"""Download the species encoder on first enable, it is too large to bundle (1.2 GB).

Files come from a GitHub release and land in Application Support, verified by sha256
and renamed into place atomically so a killed download never leaves a broken model.
"""

import hashlib
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from fovea.core.resources import resource_path

RELEASE_URL_BASE = "https://github.com/JamesZhang22/fovea/releases/download/models-v1/"

# (filename, sha256, bytes) of the exported encoder, pinned at export time
MODEL_FILES = [
    ("species.onnx", "7192f5c4f561d56c658c8347f178ed394e8d54763d925bfc35353d761ca4f31f", 2307765),
    (
        "species.onnx.data",
        "1ebbdb09e9cf0b680aa0d9c05de1acf7cf3d7cfb2ab8e59565dc417bf8a5495b",
        1215954944,
    ),
]
TOTAL_BYTES = sum(f[2] for f in MODEL_FILES)
CHUNK_BYTES = 1 << 20

Progress = Callable[[int, int], None]


def downloaded_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "fovea" / "models"


def species_model_path() -> Path | None:
    """The encoder to run: dev/bundled copy first, else the downloaded one."""
    for candidate in (resource_path("models/species.onnx"), downloaded_dir() / "species.onnx"):
        if candidate.exists():
            return candidate
    return None


def download_species_model(progress: Progress | None = None, base_url: str = RELEASE_URL_BASE):
    """Fetch, verify, and install the encoder files, calling progress(done, total) bytes."""
    dest_dir = downloaded_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for name, sha256, _ in MODEL_FILES:
        partial = dest_dir / f"{name}.partial"
        digest = hashlib.sha256()
        with urllib.request.urlopen(base_url + name) as response, open(partial, "wb") as out:
            while chunk := response.read(CHUNK_BYTES):
                digest.update(chunk)
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, TOTAL_BYTES)
        if digest.hexdigest() != sha256:
            partial.unlink()
            raise ValueError(f"checksum mismatch for {name}, download corrupted")
        shutil.move(partial, dest_dir / name)
