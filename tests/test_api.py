import time
from pathlib import Path

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
    import pytest

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
