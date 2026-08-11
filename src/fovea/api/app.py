import asyncio
import io
import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from fovea.core.ingest import cr3, decode
from fovea.core.ingest.cache import Cache
from fovea.core.pipeline import PipelineConfig, run_pipeline

PROGRESS_EVERY = 10  # progress events per stage are throttled to every Nth item
THUMB_QUALITY = 85


class OpenFolderRequest(BaseModel):
    path: str
    detect: bool = True
    eye: bool = True
    gap_seconds: float = 2.0
    metric: str = "brenner"


class Session:
    """One open folder: entries, cache, and a broadcast queue for SSE progress."""

    def __init__(self) -> None:
        self.folder: Path | None = None
        self.entries: list[dict] = []
        self.cache: Cache | None = None
        self.status = "idle"
        self.error: str | None = None
        self.events: queue.Queue[dict | None] = queue.Queue()
        self.lock = threading.Lock()

    def emit(self, event: dict) -> None:
        self.events.put(event)

    def run(self, request: OpenFolderRequest) -> None:
        folder = Path(request.path).expanduser().resolve()
        self.folder = folder
        self.status = "running"
        self.error = None
        self.cache = Cache(folder / ".fovea" / "cache.sqlite")
        config = PipelineConfig(
            detect=request.detect,
            eye=request.eye and Path("models/eye.onnx").exists(),
            export=False,
            gap_seconds=request.gap_seconds,
            metric=request.metric,
        )

        def progress(stage: str, done: int, total: int) -> None:
            if done % PROGRESS_EVERY == 0 or done == total:
                self.emit({"type": "progress", "stage": stage, "done": done, "total": total})

        try:
            entries = run_pipeline(folder, config, self.cache, progress)
            with self.lock:
                self.entries = entries
            self.status = "ready"
            self.emit({"type": "done", "count": len(entries)})
        except Exception as exc:  # surfaced to the UI instead of dying silently
            self.status = "error"
            self.error = str(exc)
            self.emit({"type": "error", "message": str(exc)})


def entry_payload(idx: int, e: dict) -> dict:
    """Manifest entry trimmed to what the UI renders."""
    meta = e["meta"]
    return {
        "id": idx,
        "name": Path(e["path"]).name,
        "width": meta.get("ImageWidth"),
        "height": meta.get("ImageHeight"),
        "orientation": meta.get("Orientation", 1),
        "burst": e.get("burst"),
        "burst_size": e.get("burst_size", 1),
        "rank": e.get("rank"),
        "metrics": e.get("metrics"),
        "eye": e.get("eye"),
        "birds": e.get("birds"),
        "af": e.get("af"),
        "shot_time": meta.get("DateTimeOriginal"),
    }


def create_app() -> FastAPI:
    app = FastAPI(title="fovea")
    session = Session()

    @app.post("/api/folder")
    def open_folder(request: OpenFolderRequest) -> dict:
        folder = Path(request.path).expanduser().resolve()
        if not folder.is_dir():
            raise HTTPException(404, f"not a directory: {folder}")
        if session.status == "running":
            raise HTTPException(409, "pipeline already running")
        threading.Thread(target=session.run, args=(request,), daemon=True).start()
        return {"status": "started", "folder": str(folder)}

    @app.get("/api/status")
    def status() -> dict:
        return {"status": session.status, "error": session.error, "count": len(session.entries)}

    @app.get("/api/entries")
    def entries() -> list[dict]:
        with session.lock:
            return [entry_payload(i, e) for i, e in enumerate(session.entries)]

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        async def stream():
            while True:
                event = await asyncio.to_thread(session.events.get)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("done", "error"):
                    break

        return StreamingResponse(stream(), media_type="text/event-stream")

    def _entry(idx: int) -> dict:
        with session.lock:
            if not 0 <= idx < len(session.entries):
                raise HTTPException(404, "no such entry")
            return session.entries[idx]

    @app.get("/api/image/{idx}/thumb")
    def thumb(idx: int, w: int = 400) -> Response:
        e = _entry(idx)
        path = Path(e["path"])
        kind = f"thumb:{w}"
        if session.cache and (cached := session.cache.get(path, kind)):
            return Response(cached, media_type="image/jpeg")
        previews = cr3.read_previews(path)
        src = previews.full or previews.prvw
        if src is None:
            raise HTTPException(404, "no preview")
        im = decode.thumbnail(cr3.read_range(path, src), w)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=THUMB_QUALITY)
        data = buf.getvalue()
        if session.cache:
            session.cache.put(path, kind, data)
        return Response(data, media_type="image/jpeg")

    @app.get("/api/image/{idx}/full")
    def full(idx: int) -> Response:
        """The embedded full-resolution JPEG, raw bytes, no re-encode."""
        e = _entry(idx)
        path = Path(e["path"])
        previews = cr3.read_previews(path)
        if previews.full is None:
            raise HTTPException(404, "no full-size preview")
        return Response(
            cr3.read_range(path, previews.full),
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=300"},
        )

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=7343, log_level="warning")


if __name__ == "__main__":
    main()
