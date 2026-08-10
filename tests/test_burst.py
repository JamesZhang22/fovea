from fovea.core.group.burst import group_bursts, shot_time


def entry(path: str, dt: str | None, subsec: str | None = None) -> dict:
    meta: dict = {"DateTimeOriginal": dt}
    if subsec is not None:
        meta["SubSecTimeOriginal"] = subsec
    return {"path": path, "meta": meta}


def test_shot_time_with_subsec() -> None:
    a = shot_time({"DateTimeOriginal": "2026:08:09 10:00:00", "SubSecTimeOriginal": "25"})
    b = shot_time({"DateTimeOriginal": "2026:08:09 10:00:00", "SubSecTimeOriginal": "75"})
    assert a is not None and b is not None
    assert b - a == 0.5


def test_groups_split_on_gap() -> None:
    entries = [
        entry("a", "2026:08:09 10:00:00"),
        entry("b", "2026:08:09 10:00:01"),
        entry("c", "2026:08:09 10:00:10"),
        entry("d", "2026:08:09 10:00:11"),
    ]
    bursts = group_bursts(entries, gap_seconds=2.0)
    assert [[e["path"] for e in b] for b in bursts] == [["a", "b"], ["c", "d"]]


def test_unsorted_input_and_singletons() -> None:
    entries = [
        entry("late", "2026:08:09 12:00:00"),
        entry("early", "2026:08:09 10:00:00"),
        entry("no_time", None),
    ]
    bursts = group_bursts(entries)
    assert [[e["path"] for e in b] for b in bursts] == [["early"], ["late"], ["no_time"]]


def test_subsecond_burst_stays_together() -> None:
    entries = [entry(f"f{i}", "2026:08:09 10:00:00", subsec=str(i * 6).zfill(2)) for i in range(15)]
    assert len(group_bursts(entries)) == 1
