import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from fovea.api.schemas import (
    Entry,
    ExportResponse,
    OpenFolderRequest,
    OpenFolderResponse,
    RateRequest,
    SpeciesModelStatus,
    SpeciesRequest,
    StatusResponse,
    entry_from_pipeline,
)
from fovea.api.session import Session
from fovea.core.pipeline import PipelineConfig, export_verdicts
from fovea.core.resources import resource_path
from fovea.core.species.classify import REGIONS, label_names
from fovea.core.species.download import TOTAL_BYTES, species_model_path


def folder_router(session: Session) -> APIRouter:
    router = APIRouter()

    @router.post("/api/folder", response_model=OpenFolderResponse)
    def open_folder(request: OpenFolderRequest) -> OpenFolderResponse:
        folder = Path(request.path).expanduser().resolve()
        if not folder.is_dir():
            raise HTTPException(404, f"not a directory: {folder}")
        if request.species_region is not None and request.species_region not in REGIONS:
            raise HTTPException(422, f"region must be one of {sorted(REGIONS)}")
        if session.status == "running":
            raise HTTPException(409, "pipeline already running")
        session.start(request)
        return OpenFolderResponse(status="started", folder=str(folder))

    @router.get("/api/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        return StatusResponse(
            status=session.status, error=session.error, count=len(session.entries)
        )

    @router.get("/api/entries", response_model=list[Entry])
    def entries() -> list[Entry]:
        with session.lock:
            return [entry_from_pipeline(i, e) for i, e in enumerate(session.entries)]

    @router.post("/api/rate", response_model=Entry)
    def rate(request: RateRequest) -> Entry:
        if not 0 <= request.id < len(session.entries):
            raise HTTPException(404, "no such entry")
        if request.rating is not None and not 0 <= request.rating <= 5:
            raise HTTPException(422, "rating must be 0-5")
        session.rate(request.id, request.rating, request.rejected)
        with session.lock:
            return entry_from_pipeline(request.id, session.entries[request.id])

    @router.post("/api/species", response_model=list[Entry])
    def confirm_species(request: SpeciesRequest) -> list[Entry]:
        ids = session.confirm_species(request.burst, request.common)
        if not ids:
            raise HTTPException(404, "no such burst")
        with session.lock:
            return [entry_from_pipeline(i, session.entries[i]) for i in ids]

    @router.get("/api/species/model", response_model=SpeciesModelStatus)
    def species_model() -> SpeciesModelStatus:
        d = session.model_download
        return SpeciesModelStatus(
            present=species_model_path() is not None,
            downloading=d["state"] == "downloading",
            done_bytes=d["done"],
            total_bytes=TOTAL_BYTES,
            error=d["error"],
        )

    @router.post("/api/species/model", response_model=SpeciesModelStatus)
    def download_species_model() -> SpeciesModelStatus:
        if species_model_path() is None:
            session.start_model_download()
        return species_model()

    @router.get("/api/species/names", response_model=list[str])
    def species_names() -> list[str]:
        labels = resource_path("models/species_labels.npz")
        if not labels.exists():
            return []
        return label_names(labels)

    @router.post("/api/export", response_model=ExportResponse)
    def export() -> ExportResponse:
        if session.status != "ready":
            raise HTTPException(409, "no folder loaded")
        with session.lock:
            entries = list(session.entries)
        config = PipelineConfig(species_labels=str(resource_path("models/species_labels.npz")))
        written, skipped = export_verdicts(entries, config)
        return ExportResponse(written=written, skipped_foreign=skipped)

    @router.get("/api/events")
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

    return router
