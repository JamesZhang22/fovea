import numpy as np
import pytest
from conftest import structured_image
from PIL import ImageFilter

from fovea.core.score.blurtype import classify, spectral_anisotropy
from fovea.core.score.classical import to_gray


def motion_blur(g: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    """Average of copies shifted along a line, a simple linear motion kernel."""
    out = np.zeros_like(g)
    rad = np.deg2rad(angle_deg)
    for t in np.linspace(-length / 2, length / 2, length):
        out += np.roll(np.roll(g, round(t * np.sin(rad)), 0), round(t * np.cos(rad)), 1)
    return out / length


def test_defocus_is_isotropic() -> None:
    blurred = structured_image(256).filter(ImageFilter.GaussianBlur(4))
    assert classify(to_gray(blurred)) == "defocus"


@pytest.mark.parametrize("angle", [0, 45, 90, 135])
def test_motion_is_anisotropic_and_angle_recovered(angle: int) -> None:
    g = motion_blur(to_gray(structured_image(256)), length=15, angle_deg=angle)
    aniso, est = spectral_anisotropy(g)
    assert classify(g) == "motion"
    err = min(abs(est - angle), 180 - abs(est - angle))
    assert err <= 15, f"angle {angle} estimated as {est}"


def test_motion_beats_defocus_anisotropy() -> None:
    sharp = to_gray(structured_image(256))
    defocus, _ = spectral_anisotropy(
        to_gray(structured_image(256).filter(ImageFilter.GaussianBlur(4)))
    )
    motion, _ = spectral_anisotropy(motion_blur(sharp, 15, 30))
    assert motion > defocus
