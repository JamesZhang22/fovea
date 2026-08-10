import io

from PIL import Image


def decode_scaled(jpeg: bytes, target_width: int) -> Image.Image:
    im = Image.open(io.BytesIO(jpeg))
    # draft picks the largest DCT scale (1/1..1/8) that still covers target_width
    im.draft("RGB", (target_width, 1))
    im.load()
    return im


def decode_full(jpeg: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(jpeg))
    im.load()
    return im


def roi_native(jpeg: bytes, box: tuple[int, int, int, int]) -> Image.Image:
    # full decode then crop; swap to TurboJPEG lossless crop if this shows up in profiles
    return decode_full(jpeg).crop(box)


def thumbnail(jpeg: bytes, width: int) -> Image.Image:
    im = decode_scaled(jpeg, width)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    return im
