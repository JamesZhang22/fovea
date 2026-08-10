from dataclasses import dataclass

LATTICE_THRESHOLD = 100  # ValidAFPoints above this means the R7 dumped its full AF lattice


@dataclass(frozen=True)
class AFPoint:
    """Pixel-space box in unrotated (landscape) sensor coordinates."""

    cx: float
    cy: float
    w: float
    h: float
    in_focus: bool
    selected: bool


@dataclass(frozen=True)
class AFFrame:
    points: list[AFPoint]
    lattice: bool
    orientation: int

    @property
    def display_points(self) -> list[AFPoint]:
        # full-lattice reports (Whole Area, no subject lock): only in-focus cells carry signal
        if self.lattice:
            return [p for p in self.points if p.in_focus]
        return self.points


def _ints(v: object) -> list[int]:
    """Exiftool array tags arrive as space-separated strings, single values as numbers."""
    if isinstance(v, str):
        return [int(x) for x in v.split()]
    if isinstance(v, (int, float)):
        return [int(v)]
    return []


def _bit_indices(words: list[int]) -> set[int]:
    """Set bit positions across a list of 16-bit words, bit j of word k = index 16k+j."""
    out = set()
    for k, w in enumerate(words):
        w &= 0xFFFF
        for j in range(16):
            if w >> j & 1:
                out.add(16 * k + j)
    return out


def parse_af_frame(meta: dict) -> AFFrame | None:
    """Canon AFInfo2 metadata to pixel-space AF boxes, None if required tags are missing."""
    n = int(meta.get("ValidAFPoints") or 0)
    orientation = int(meta.get("Orientation") or 1)
    if n == 0:
        return AFFrame(points=[], lattice=False, orientation=orientation)

    af_w, af_h = meta.get("AFImageWidth"), meta.get("AFImageHeight")
    img_w, img_h = meta.get("ImageWidth"), meta.get("ImageHeight")
    if not (af_w and af_h and img_w and img_h):
        return None
    # AF math happens in landscape sensor space
    if img_w < img_h:
        img_w, img_h = img_h, img_w
    sx, sy = img_w / af_w, img_h / af_h

    xs = _ints(meta.get("AFAreaXPositions"))[:n]
    ys = _ints(meta.get("AFAreaYPositions"))[:n]
    ws = _ints(meta.get("AFAreaWidths"))[:n]
    hs = _ints(meta.get("AFAreaHeights"))[:n]
    if len(xs) < n or len(ys) < n or len(ws) < n or len(hs) < n:
        return None
    in_focus = _bit_indices(_ints(meta.get("AFPointsInFocus")))
    selected = _bit_indices(_ints(meta.get("AFPointsSelected")))

    points = [
        AFPoint(
            cx=(af_w / 2 + xs[i]) * sx,
            cy=(af_h / 2 - ys[i]) * sy,  # EOS: positive Y is up
            w=ws[i] * sx,
            h=hs[i] * sy,
            in_focus=i in in_focus,
            selected=i in selected,
        )
        for i in range(n)
    ]
    return AFFrame(points=points, lattice=n > LATTICE_THRESHOLD, orientation=orientation)
