from fastapi import FastAPI

from fovea.api.routes.folder import folder_router
from fovea.api.routes.images import images_router
from fovea.api.session import Session

HOST = "127.0.0.1"  # localhost only, never exposed to the network
PORT = 7343


def create_app() -> FastAPI:
    app = FastAPI(title="fovea")
    session = Session()
    app.include_router(folder_router(session))
    app.include_router(images_router(session))
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
