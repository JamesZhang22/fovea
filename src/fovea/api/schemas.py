from pathlib import Path

from pydantic import BaseModel

from fovea.core.pipeline import EYE_MIN_CONFIDENCE


class OpenFolderRequest(BaseModel):
    path: str
    detect: bool = True
    eye: bool = True
    species: bool = True
    species_region: str | None = None
    gap_seconds: float = 3.0
    metric: str = "brenner"


class OpenFolderResponse(BaseModel):
    status: str
    folder: str


class StatusResponse(BaseModel):
    status: str
    error: str | None
    count: int


class Eye(BaseModel):
    x: float
    y: float
    confidence: float


class BirdBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float


class AFPoint(BaseModel):
    cx: float
    cy: float
    w: float
    h: float
    in_focus: bool
    selected: bool


class AFInfo(BaseModel):
    lattice: bool
    orientation: int
    n_points: int
    display_points: list[AFPoint]


class SpeciesPrediction(BaseModel):
    common: str | None
    scientific: str
    family: str | None
    confidence: float


class SpeciesRequest(BaseModel):
    burst: int
    common: str | None  # None clears the confirmation.


class SpeciesModelStatus(BaseModel):
    present: bool
    downloading: bool
    done_bytes: int
    total_bytes: int
    error: str | None


class ExportResponse(BaseModel):
    written: int
    skipped_foreign: int


class RateRequest(BaseModel):
    id: int
    rating: int | None = None  # 0 clears, 1-5 sets.
    rejected: bool | None = None


class Entry(BaseModel):
    id: int
    name: str
    width: int | None
    height: int | None
    orientation: int
    burst: int | None
    burst_size: int
    rank: float | None
    metrics: dict[str, float | None] | None
    eye: Eye | None
    eye_used: bool
    birds: list[BirdBox] | None
    af: AFInfo | None
    species: list[SpeciesPrediction] | None
    species_user: str | None
    shot_time: str | None
    user_rating: int | None
    rejected: bool


def entry_from_pipeline(idx: int, e: dict) -> Entry:
    """Pipeline entry dict trimmed to what the UI renders."""
    meta = e["meta"]
    eye = e.get("eye")
    return Entry(
        id=idx,
        name=Path(e["path"]).name,
        width=meta.get("ImageWidth"),
        height=meta.get("ImageHeight"),
        orientation=meta.get("Orientation", 1),
        burst=e.get("burst"),
        burst_size=e.get("burst_size", 1),
        rank=e.get("rank"),
        metrics=e.get("metrics"),
        eye=eye,
        eye_used=bool(eye and eye["confidence"] >= EYE_MIN_CONFIDENCE),
        birds=e.get("birds"),
        af=e.get("af"),
        species=e.get("species"),
        species_user=e.get("user", {}).get("species"),
        shot_time=meta.get("DateTimeOriginal"),
        user_rating=e.get("user", {}).get("rating"),
        rejected=e.get("user", {}).get("rejected", False),
    )
