import io
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from fovea.api.session import Session
from fovea.core.ingest import cr3, decode

THUMB_QUALITY = 85


def images_router(session: Session) -> APIRouter:
    router = APIRouter()

    def entry_path(idx: int) -> Path:
        with session.lock:
            if not 0 <= idx < len(session.entries):
                raise HTTPException(404, "no such entry")
            return Path(session.entries[idx]["path"])

    @router.get("/api/image/{idx}/thumb")
    def thumb(idx: int, w: int = 400) -> Response:
        path = entry_path(idx)
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

    @router.get("/api/image/{idx}/full")
    def full(idx: int) -> Response:
        """The embedded full-resolution JPEG, raw bytes, no re-encode."""
        path = entry_path(idx)
        previews = cr3.read_previews(path)
        if previews.full is None:
            raise HTTPException(404, "no full-size preview")
        return Response(
            cr3.read_range(path, previews.full),
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=300"},
        )

    return router
