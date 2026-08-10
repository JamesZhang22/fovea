import io
import struct

from fovea.core.ingest.cr3 import iter_boxes, parse_previews


def box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), typ) + payload


def uuid_box(usertype: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 24 + len(payload), b"uuid") + usertype + payload


JPEG = b"\xff\xd8fakejpegdata\xff\xd9"


def stbl(sample_size: int, offset: int) -> bytes:
    stsz = box(b"stsz", struct.pack(">III", 0, sample_size, 1))
    stco = box(b"stco", struct.pack(">II", 0, 1) + struct.pack(">I", offset))
    return box(b"stbl", stsz + stco)


def test_iter_boxes_walks_siblings() -> None:
    buf = box(b"aaaa", b"x" * 4) + box(b"bbbb", b"y" * 10)
    found = [(t, e - s) for t, s, e in iter_boxes(buf, 0, len(buf))]
    assert found == [(b"aaaa", 4), (b"bbbb", 10)]


def test_parse_previews_finds_trak_jpeg_and_prvw() -> None:
    mdat = box(b"mdat", JPEG)  # JPEG payload lands at file offset 8
    trak = box(b"trak", box(b"mdia", box(b"minf", stbl(len(JPEG), 8))))
    moov = box(b"moov", trak)
    prvw_payload = b"\x00" * 8 + JPEG
    prvw = uuid_box(b"\xea\xf4\x2b\x5e\x1c\x98\x4b\x88\xb9\xfb\xb7\xdc\x40\x6e\x4d\x16",
                    box(b"PRVW", prvw_payload))
    full = mdat + moov + prvw

    previews = parse_previews(io.BytesIO(full))
    assert previews.prvw is not None
    assert full[previews.prvw.offset : previews.prvw.offset + 2] == b"\xff\xd8"
    assert previews.full is not None
    assert previews.full.offset == 8
    assert previews.full.size == len(JPEG)


def test_parse_previews_ignores_non_jpeg_trak() -> None:
    trak = box(b"trak", box(b"mdia", box(b"minf", stbl(4, 0))))
    moov = box(b"moov", trak)
    buf = b"CRAW" + moov
    previews = parse_previews(io.BytesIO(buf))
    assert previews.full is None
