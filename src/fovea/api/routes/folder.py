import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from fovea.api.schemas import (
    Entry,
    OpenFolderRequest,
    OpenFolderResponse,
    StatusResponse,
    entry_from_pipeline,
)
from fovea.api.session import Session


def folder_router(session: Session) -> APIRouter:
    router = APIRouter()

    @router.post("/api/folder", response_model=OpenFolderResponse)
    def open_folder(request: OpenFolderRequest) -> OpenFolderResponse:
        folder = Path(request.path).expanduser().resolve()
        if not folder.is_dir():
            raise HTTPException(404, f"not a directory: {folder}")
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
