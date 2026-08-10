from contextlib import suppress
from datetime import datetime

DEFAULT_GAP_SECONDS = 2.0


def shot_time(meta: dict) -> float | None:
    dt = meta.get("DateTimeOriginal")
    if not dt:
        return None
    try:
        ts = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None
    subsec = str(meta.get("SubSecTimeOriginal", "") or "0")
    with suppress(ValueError):
        ts += int(subsec) / 10 ** len(subsec)
    return ts


def group_bursts(entries: list[dict], gap_seconds: float = DEFAULT_GAP_SECONDS) -> list[list[dict]]:
    """Cluster scan entries into bursts by shot-time gaps, untimestamped entries become singletons."""
    timed, untimed = [], []
    for e in entries:
        ts = shot_time(e["meta"])
        (timed if ts is not None else untimed).append((ts, e))

    timed.sort(key=lambda t: (t[0], t[1]["path"]))
    bursts: list[list[dict]] = []
    prev_ts = None
    for ts, e in timed:
        if prev_ts is None or ts - prev_ts > gap_seconds:
            bursts.append([])
        bursts[-1].append(e)
        prev_ts = ts

    bursts.extend([e] for _, e in untimed)
    return bursts
