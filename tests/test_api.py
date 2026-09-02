import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fovea.api.app import create_app


def wait_ready(client: TestClient, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = client.get("/api/status").json()
        if s["status"] in ("ready", "error"):
            return s
        time.sleep(0.05)
    raise TimeoutError


def test_open_missing_folder_404() -> None:
    client = TestClient(create_app())
    assert client.post("/api/folder", json={"path": "/nope/nothing"}).status_code == 404


def test_empty_folder_pipeline(tmp_path: Path) -> None:
    client = TestClient(create_app())
    r = client.post("/api/folder", json={"path": str(tmp_path), "detect": False, "eye": False})
    assert r.status_code == 200
    s = wait_ready(client)
    assert s == {"status": "ready", "error": None, "count": 0}
    assert client.get("/api/entries").json() == []
    assert client.get("/api/image/0/thumb").status_code == 404


def test_entries_and_images_on_real_folder() -> None:
    folder = Path("data/labeling-set")
    if not folder.is_dir():
        pytest.skip("local dataset not present")
    client = TestClient(create_app())
    client.post(
        "/api/folder", json={"path": str(folder), "detect": False, "eye": False}
    ).raise_for_status()
    s = wait_ready(client, timeout=120)
    assert s["status"] == "ready" and s["count"] > 0

    entries = client.get("/api/entries").json()
    assert entries[0]["burst_size"] >= 1 and entries[0]["rank"] is not None

    thumb = client.get("/api/image/0/thumb?w=300")
    assert thumb.headers["content-type"] == "image/jpeg" and len(thumb.content) > 1000
    full = client.get("/api/image/0/full")
    assert full.content[:2] == b"\xff\xd8" and len(full.content) > 1_000_000


def test_rate_persists_across_sessions(tmp_path: Path) -> None:
    img = tmp_path / "IMG_0001.CR3"
    img.write_bytes(b"not a real cr3")
    client = TestClient(create_app())
    client.post("/api/folder", json={"path": str(tmp_path), "detect": False, "eye": False})
    wait_ready(client)

    r = client.post("/api/rate", json={"id": 0, "rating": 4, "rejected": False})
    assert r.status_code == 200
    assert r.json()["user_rating"] == 4 and r.json()["rejected"] is False

    r = client.post("/api/rate", json={"id": 0, "rating": 0, "rejected": True})
    assert r.json()["user_rating"] is None and r.json()["rejected"] is True

    fresh = TestClient(create_app())
    fresh.post("/api/folder", json={"path": str(tmp_path), "detect": False, "eye": False})
    wait_ready(fresh)
    e = fresh.get("/api/entries").json()[0]
    assert e["user_rating"] is None and e["rejected"] is True

    assert fresh.post("/api/rate", json={"id": 99, "rating": 3}).status_code == 404


def test_export_writes_verdicts(tmp_path: Path) -> None:
    for name in ("IMG_0001.CR3", "IMG_0002.CR3", "IMG_0003.CR3"):
        (tmp_path / name).write_bytes(b"fake")
    foreign = tmp_path / "IMG_0003.xmp"
    foreign.write_text("<x:xmpmeta>adobe edits</x:xmpmeta>")

    client = TestClient(create_app())
    client.post("/api/folder", json={"path": str(tmp_path), "detect": False, "eye": False})
    wait_ready(client)
    client.post("/api/rate", json={"id": 0, "rating": 5})
    client.post("/api/rate", json={"id": 1, "rejected": True})

    r = client.post("/api/export").json()
    assert r == {"written": 2, "skipped_foreign": 1}
    assert 'xmp:Rating="5"' in (tmp_path / "IMG_0001.xmp").read_text()
    assert 'xmpDM:good="False"' in (tmp_path / "IMG_0002.xmp").read_text()
    assert foreign.read_text() == "<x:xmpmeta>adobe edits</x:xmpmeta>"

    client.post("/api/rate", json={"id": 0, "rating": 2})
    assert client.post("/api/export").json() == {"written": 2, "skipped_foreign": 1}
    assert 'xmp:Rating="2"' in (tmp_path / "IMG_0001.xmp").read_text()


def test_confirm_species_applies_to_burst(tmp_path: Path) -> None:
    for name in ("IMG_0001.CR3", "IMG_0002.CR3"):
        (tmp_path / name).write_bytes(b"fake")
    client = TestClient(create_app())
    client.post("/api/folder", json={"path": str(tmp_path), "detect": False, "eye": False})
    wait_ready(client)
    burst = client.get("/api/entries").json()[0]["burst"]

    r = client.post("/api/species", json={"burst": burst, "common": "Great Egret"})
    assert r.status_code == 200
    assert all(e["species_user"] == "Great Egret" for e in r.json())

    fresh = TestClient(create_app())
    fresh.post("/api/folder", json={"path": str(tmp_path), "detect": False, "eye": False})
    wait_ready(fresh)
    e = fresh.get("/api/entries").json()[0]
    assert e["species_user"] == "Great Egret"

    r = client.post("/api/species", json={"burst": burst, "common": None})
    assert all(e["species_user"] is None for e in r.json())

    assert client.post("/api/species", json={"burst": 999, "common": "x"}).status_code == 404
