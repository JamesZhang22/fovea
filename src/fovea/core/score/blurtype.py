import numpy as np

N_ANGLE_BINS = 18  # angular sectors over 180 degrees, 10 degrees each.
MOTION_THRESHOLD = 0.3  # anisotropy above this reads as motion blur, below as defocus.


def spectral_anisotropy(gray: np.ndarray) -> tuple[float, float]:
    """Anisotropy of the log spectrum in [0, 1) and the implied motion angle in degrees.

    Linear motion blur suppresses energy along the motion direction, defocus is isotropic.
    Log magnitude keeps scene-content edges from dominating the measure.
    """
    g = gray - gray.mean()
    wy = np.hanning(g.shape[0])[:, None]
    wx = np.hanning(g.shape[1])[None, :]
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g * wy * wx))))

    h, w = spectrum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    fy, fx = (yy - h // 2) / h, (xx - w // 2) / w
    radius = np.sqrt(fy**2 + fx**2)
    annulus = (radius > 0.05) & (radius < 0.35)

    angle = np.arctan2(fy, fx) % np.pi
    bins = np.minimum((angle / np.pi * N_ANGLE_BINS).astype(int), N_ANGLE_BINS - 1)
    energy = np.array(
        [
            spectrum[annulus & (bins == i)].mean() if (annulus & (bins == i)).any() else 0.0
            for i in range(N_ANGLE_BINS)
        ]
    )
    if energy.max() <= 0:
        return 0.0, 0.0

    aniso = float(1.0 - energy.min() / energy.max())
    ridge_deg = energy.argmax() * (180.0 / N_ANGLE_BINS)
    motion_deg = (ridge_deg + 90.0) % 180.0
    return aniso, motion_deg


def classify(gray: np.ndarray, threshold: float = MOTION_THRESHOLD) -> str:
    """Label a blurry patch as 'motion' or 'defocus', only meaningful when the patch is soft."""
    aniso, _ = spectral_anisotropy(gray)
    return "motion" if aniso > threshold else "defocus"
