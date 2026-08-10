import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

HEADER_READ = 200_000
JPEG_SOI = b"\xff\xd8"


@dataclass(frozen=True)
class ByteRange:
    offset: int
    size: int


@dataclass(frozen=True)
class Cr3Previews:
    full: ByteRange | None
    prvw: ByteRange | None
    thmb: ByteRange | None


def iter_boxes(buf: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    off = start
    while off + 8 <= end:
        size, typ = struct.unpack_from(">I4s", buf, off)
        hdr = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, off + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - off
        if size < hdr or off + size > end:
            return
        if typ == b"uuid":
            hdr += 16
        yield typ, off + hdr, off + size
        off += size


def _find_jpeg_in_payload(buf: bytes, start: int, end: int) -> ByteRange | None:
    soi = buf.find(JPEG_SOI, start, end)
    if soi == -1:
        return None
    return ByteRange(soi, end - soi)


def _trak_first_sample(buf: bytes, start: int, end: int) -> ByteRange | None:
    stbl = None
    node = (start, end)
    for name in (b"mdia", b"minf", b"stbl"):
        found = next(((s, e) for t, s, e in iter_boxes(buf, *node) if t == name), None)
        if found is None:
            return None
        node = found
    stbl = node
    size = offset = None
    for typ, s, _ in iter_boxes(buf, *stbl):
        if typ == b"stsz":
            fixed, count = struct.unpack_from(">II", buf, s + 4)
            if count:
                size = fixed if fixed else struct.unpack_from(">I", buf, s + 12)[0]
        elif typ == b"co64":
            offset = struct.unpack_from(">Q", buf, s + 8)[0]
        elif typ == b"stco":
            offset = struct.unpack_from(">I", buf, s + 8)[0]
    if size is None or offset is None:
        return None
    return ByteRange(offset, size)


def parse_previews(f: BinaryIO) -> Cr3Previews:
    head = f.read(HEADER_READ)
    full = prvw = thmb = None
    for typ, s, e in iter_boxes(head, 0, len(head)):
        if typ == b"moov":
            for t, ts, te in iter_boxes(head, s, e):
                if t == b"trak":
                    sample = _trak_first_sample(head, ts, te)
                    if sample is None:
                        continue
                    f.seek(sample.offset)
                    if f.read(2) == JPEG_SOI and (full is None or sample.size > full.size):
                        full = sample
                elif t == b"uuid":
                    for t2, s2, e2 in iter_boxes(head, ts, te):
                        if t2 == b"THMB":
                            thmb = _find_jpeg_in_payload(head, s2, e2)
        elif typ == b"uuid":
            for t2, s2, e2 in iter_boxes(head, s, e):
                if t2 == b"PRVW":
                    prvw = _find_jpeg_in_payload(head, s2, e2)
    return Cr3Previews(full=full, prvw=prvw, thmb=thmb)


def read_range(path: Path, r: ByteRange) -> bytes:
    with open(path, "rb") as f:
        f.seek(r.offset)
        return f.read(r.size)


def read_previews(path: Path) -> Cr3Previews:
    with open(path, "rb") as f:
        return parse_previews(f)
