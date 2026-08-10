import io

from PIL import Image


def decode_scaled(jpeg: bytes, target_width: int) -> Image.Image:
    """Decode at the largest DCT scale (1/1..1/8) that still covers target_width."""
    im = Image.open(io.BytesIO(jpeg))
    im.draft("RGB", (target_width, 1))
    im.load()
    return im


def decode_full(jpeg: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(jpeg))
    im.load()
    return im


def roi_native(jpeg: bytes, box: tuple[int, int, int, int]) -> Image.Image:
    """Native-resolution crop, currently full decode then crop, TurboJPEG if profiles demand."""
    return decode_full(jpeg).crop(box)


def thumbnail(jpeg: bytes, width: int) -> Image.Image:
    """Exact-width thumbnail, DCT scale down then Lanczos for the final step."""
    im = decode_scaled(jpeg, width)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    return im
