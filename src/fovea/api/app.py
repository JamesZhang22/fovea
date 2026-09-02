import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from fovea.api.routes.folder import folder_router
from fovea.api.routes.images import images_router
from fovea.api.session import Session
from fovea.core.resources import resource_path

HOST = "127.0.0.1"  # localhost only, never exposed to the network.
PORT = 7343


def create_app() -> FastAPI:
    app = FastAPI(title="fovea")
    session = Session()
    app.include_router(folder_router(session))
    app.include_router(images_router(session))
    ui_dist = resource_path("ui/dist")
    if ui_dist.is_dir():
        app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")
    return app


def main() -> None:
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
