import shutil
from pathlib import Path

import pytest
from PIL import Image

from fovea.core.metadata.exiftool import ExifTool

pytestmark = pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool not installed")


def test_stay_open_batch(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (10 + i, 20)).save(p)
        paths.append(p)

    with ExifTool() as et:
        results = et.metadata(paths, tags=["SourceFile", "ImageWidth", "ImageHeight"])
        assert len(results) == 3
        assert sorted(r["ImageWidth"] for r in results) == [10, 11, 12]
        # second batch on the same process
        again = et.metadata(paths[:1], tags=["ImageHeight"])
        assert again[0]["ImageHeight"] == 20
