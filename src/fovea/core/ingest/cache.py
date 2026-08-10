import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    path TEXT NOT NULL,
    mtime INTEGER NOT NULL,
    size INTEGER NOT NULL,
    kind TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (path, kind)
);
"""


class Cache:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute(SCHEMA)

    def _stat(self, path: Path) -> tuple[int, int]:
        st = path.stat()
        return int(st.st_mtime), st.st_size

    def get(self, path: Path, kind: str) -> bytes | None:
        mtime, size = self._stat(path)
        with self.lock:
            row = self.conn.execute(
                "SELECT data FROM entries WHERE path=? AND kind=? AND mtime=? AND size=?",
                (str(path), kind, mtime, size),
            ).fetchone()
        return row[0] if row else None

    def put(self, path: Path, kind: str, data: bytes) -> None:
        mtime, size = self._stat(path)
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO entries (path, mtime, size, kind, data) VALUES (?,?,?,?,?)",
                (str(path), mtime, size, kind, data),
            )
            self.conn.commit()

    def get_json(self, path: Path, kind: str) -> dict | None:
        data = self.get(path, kind)
        return json.loads(data) if data else None

    def put_json(self, path: Path, kind: str, obj: dict) -> None:
        self.put(path, kind, json.dumps(obj).encode())

    def close(self) -> None:
        self.conn.close()
