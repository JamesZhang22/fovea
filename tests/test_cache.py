import os
import time
from pathlib import Path

from fovea.core.ingest.cache import Cache


def test_roundtrip_and_invalidation(tmp_path: Path) -> None:
    f = tmp_path / "a.cr3"
    f.write_bytes(b"hello")
    cache = Cache(tmp_path / "db.sqlite")

    assert cache.get(f, "meta") is None
    cache.put_json(f, "meta", {"x": 1})
    assert cache.get_json(f, "meta") == {"x": 1}

    f.write_bytes(b"hello world")
    os.utime(f, (time.time() + 10, time.time() + 10))
    assert cache.get(f, "meta") is None


def test_kinds_are_independent(tmp_path: Path) -> None:
    f = tmp_path / "a.cr3"
    f.write_bytes(b"data")
    cache = Cache(tmp_path / "db.sqlite")
    cache.put(f, "thumb", b"\x01\x02")
    assert cache.get(f, "meta") is None
    assert cache.get(f, "thumb") == b"\x01\x02"
