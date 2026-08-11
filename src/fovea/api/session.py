import queue
import threading
from pathlib import Path

from fovea.api.schemas import OpenFolderRequest
from fovea.core.ingest.cache import Cache
from fovea.core.pipeline import PipelineConfig, run_pipeline

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

    def start(self, request: OpenFolderRequest) -> None:
        threading.Thread(target=self._run, args=(request,), daemon=True).start()

    def _run(self, request: OpenFolderRequest) -> None:
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
