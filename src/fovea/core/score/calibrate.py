import json
import threading
from pathlib import Path

N_BINS = 120
MAX_RADIUS_PX = 12.0

# reference distribution of blur radii from a real mixed R7 library, keeps early-use
# percentiles sane until the user's own history dominates (~300 seed samples)
# fmt: off
SEED = [1, 3, 6, 8, 10, 10, 10, 9, 8, 8, 8, 7, 6, 5, 4, 4, 4, 3, 4, 4, 4, 4, 4, 3, 3, 3,
        3, 3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2,
        2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 3, 3, 4, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# fmt: on


class Calibration:
    """Running histogram of scored blur radii across everything the user has scored.

    percentile(r) answers "sharper than what fraction of the library", which makes a
    displayed 'top 12%' mean the same thing regardless of lens or subject mix.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.counts = list(SEED)
        self.lock = threading.Lock()
        if path and path.exists():
            stored = json.loads(path.read_text())
            if len(stored.get("counts", [])) == N_BINS:
                self.counts = [s + c for s, c in zip(SEED, stored["counts"], strict=True)]
                self._user_counts = stored["counts"]
                return
        self._user_counts = [0] * N_BINS

    def _bin(self, radius_px: float) -> int:
        return min(int(radius_px / MAX_RADIUS_PX * N_BINS), N_BINS - 1)

    def record(self, radius_px: float) -> None:
        with self.lock:
            b = self._bin(radius_px)
            self.counts[b] += 1
            self._user_counts[b] += 1

    def percentile(self, radius_px: float) -> float:
        """Fraction of the library this radius is sharper than, 1.0 = sharpest."""
        with self.lock:
            b = self._bin(radius_px)
            total = sum(self.counts)
            softer = sum(self.counts[b + 1 :]) + self.counts[b] / 2
        return round(softer / total, 3) if total else 0.5

    def save(self) -> None:
        if not self.path:
            return
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"counts": self._user_counts}))


def default_calibration_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "fovea" / "calibration.json"
