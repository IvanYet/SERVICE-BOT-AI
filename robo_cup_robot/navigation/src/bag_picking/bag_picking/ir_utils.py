"""Shared image conversion, so training images and live detection look exactly the same."""
import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def to_bgr8(img: np.ndarray) -> np.ndarray:
    """Convert a camera image (8/16-bit mono IR, or colour) into an 8-bit, 3-channel image for YOLO.

    Dabai IR is dark, uneven, and has a few very bright speckles from the projector, so a plain
    min/max stretch leaves almost everything black. Instead:
      1. stretch between the 2nd and 99.5th percentile of ALL pixels (speckles get clipped),
      2. a gamma of 0.6 lifts the dark midtones,
      3. a blur smooths out the projector's dot pattern,
      4. CLAHE evens out brightness across the image.
    The same conversion is used when recording training images and when detecting, which matters
    more than the exact recipe.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        return img if img.dtype == np.uint8 else cv2.convertScaleAbs(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    f = img.astype(np.float32)
    lo, hi = np.percentile(f, (2, 99.5))
    if hi - lo < 1e-6:
        hi = float(f.max())
        lo = float(f.min())
    if hi - lo < 1e-6:
        return cv2.cvtColor(np.zeros(img.shape, np.uint8), cv2.COLOR_GRAY2BGR)
    g = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    g = np.power(g, 0.6)
    g8 = (g * 255.0).astype(np.uint8)
    # The depth projector paints a fine dot pattern over the IR image (on the Dabai the dots ARE
    # most of the light). A Gaussian blur (sigma 2.5 px) smears the dots into even surfaces while
    # keeping object outlines, which is what YOLO needs to recognise the bag's shape.
    g8 = cv2.GaussianBlur(g8, (0, 0), 2.5)
    g8 = _clahe.apply(g8)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
