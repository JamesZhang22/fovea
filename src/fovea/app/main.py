import socket
import threading
import time

import uvicorn


def wait_for_port(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"backend did not come up on {host}:{port}")


def main() -> None:
    import webview

    from fovea.api.app import HOST, PORT, create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host=HOST, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    wait_for_port(HOST, PORT)

    webview.create_window("fovea", f"http://{HOST}:{PORT}", width=1440, height=900)
    webview.start()


if __name__ == "__main__":
    main()
