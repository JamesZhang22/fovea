import queue
import threading
from pathlib import Path

from fovea.api.schemas import OpenFolderRequest
from fovea.core.ingest.cache import Cache
from fovea.core.pipeline import PipelineConfig, run_pipeline
from fovea.core.resources import resource_path

PROGRESS_EVERY = 10  # progress events per stage are throttled to every Nth item


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

    def rate(self, idx: int, rating: int | None, rejected: bool | None) -> dict:
        """Update one entry's user verdict, persisted in the cache across restarts."""
        with self.lock:
            entry = self.entries[idx]
            user = dict(entry.get("user") or {})
            if rating is not None:
                user["rating"] = None if rating == 0 else rating
            if rejected is not None:
                user["rejected"] = rejected
            entry["user"] = user
        if self.cache:
            self.cache.put_json(Path(entry["path"]), "user", user)
        return user

    def start(self, request: OpenFolderRequest) -> None:
        threading.Thread(target=self._run, args=(request,), daemon=True).start()

    def _run(self, request: OpenFolderRequest) -> None:
        folder = Path(request.path).expanduser().resolve()
        self.folder = folder
        self.status = "running"
        self.error = None
        self.cache = Cache(folder / ".fovea" / "cache.sqlite")
        eye_model = resource_path("models/eye.onnx")
        config = PipelineConfig(
            detect=request.detect,
            eye=request.eye and eye_model.exists(),
            detect_model=str(resource_path("models/bird.onnx")),
            eye_model=str(eye_model),
            export=False,
            gap_seconds=request.gap_seconds,
            metric=request.metric,
        )

        def progress(stage: str, done: int, total: int) -> None:
            if done % PROGRESS_EVERY == 0 or done == total:
                self.emit({"type": "progress", "stage": stage, "done": done, "total": total})

        try:
            entries = run_pipeline(folder, config, self.cache, progress)
            for e in entries:
                e["user"] = self.cache.get_json(Path(e["path"]), "user") or {}
            with self.lock:
                self.entries = entries
            self.status = "ready"
            self.emit({"type": "done", "count": len(entries)})
        except Exception as exc:  # surfaced to the UI instead of dying silently
            self.status = "error"
            self.error = str(exc)
            self.emit({"type": "error", "message": str(exc)})
