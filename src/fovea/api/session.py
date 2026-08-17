import queue
import threading
from pathlib import Path

from fovea.api.schemas import OpenFolderRequest
from fovea.core.ingest.cache import Cache
from fovea.core.pipeline import PipelineConfig, run_pipeline
from fovea.core.resources import resource_path
from fovea.core.score.calibrate import default_calibration_path
from fovea.core.species.download import species_model_path

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
        self.model_download = {"state": "idle", "done": 0, "error": None}

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

    def start_model_download(self) -> None:
        """Fetch the species encoder on a worker thread, progress polled via status."""
        from fovea.core.species.download import download_species_model

        with self.lock:
            if self.model_download["state"] == "downloading":
                return
            self.model_download = {"state": "downloading", "done": 0, "error": None}

        def run() -> None:
            def progress(done: int, total: int) -> None:
                self.model_download["done"] = done

            try:
                download_species_model(progress)
                self.model_download["state"] = "done"
            except Exception as exc:
                self.model_download = {"state": "error", "done": 0, "error": str(exc)}

        threading.Thread(target=run, daemon=True).start()

    def confirm_species(self, burst: int, common: str | None) -> list[int]:
        """Set or clear the confirmed species on every frame of a burst, returns their ids."""
        ids = []
        with self.lock:
            for i, entry in enumerate(self.entries):
                if entry.get("burst") != burst:
                    continue
                user = dict(entry.get("user") or {})
                if common is None:
                    user.pop("species", None)
                else:
                    user["species"] = common
                entry["user"] = user
                ids.append(i)
        if self.cache:
            for i in ids:
                self.cache.put_json(Path(self.entries[i]["path"]), "user", self.entries[i]["user"])
        return ids

    def start(self, request: OpenFolderRequest) -> None:
        threading.Thread(target=self._run, args=(request,), daemon=True).start()

    def _run(self, request: OpenFolderRequest) -> None:
        folder = Path(request.path).expanduser().resolve()
        self.folder = folder
        self.status = "running"
        self.error = None
        self.cache = Cache(folder / ".fovea" / "cache.sqlite")
        eye_model = resource_path("models/eye.onnx")
        species_model = species_model_path()
        config = PipelineConfig(
            detect=request.detect,
            eye=request.eye and eye_model.exists(),
            species=request.species and species_model is not None,
            species_region=request.species_region,
            detect_model=str(resource_path("models/bird.onnx")),
            eye_model=str(eye_model),
            focus_model=str(resource_path("models/focus.onnx")),
            species_model=str(species_model or ""),
            species_labels=str(resource_path("models/species_labels.npz")),
            calibration_path=str(default_calibration_path()),
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
