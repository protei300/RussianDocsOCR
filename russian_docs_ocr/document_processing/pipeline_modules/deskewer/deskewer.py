"""
Projection-profile deskew for document images.

After perspective correction, pages can still have a residual tilt (≤ ±5°).
This module finds and corrects that tilt by maximising the variance of
horizontal projection profiles — text lines create sharp peaks when horizontal
and smeared peaks when tilted.

Works entirely with OpenCV / NumPy — no templates required.
"""

import cv2
import numpy as np


class DocDeskewer:
    """Correct residual tilt in perspective-corrected document images.

    Args:
        angle_range: Search range in degrees, symmetric around 0.  Default ±5°.
        angle_steps: Number of candidate angles to test.  101 → 0.1° resolution.
        min_angle: Skip rotation if detected angle is below this threshold (deg).
            Avoids unnecessary resampling on already-flat documents.  Default 0.3°.
        scale: Fraction of original resolution used for angle search.  Default 0.3.
    """

    def __init__(
        self,
        angle_range: float = 2.0,
        angle_steps: int = 81,
        min_angle: float = 0.5,
        scale: float = 0.3,
    ):
        self.angles = np.linspace(-angle_range, angle_range, angle_steps)
        self.min_angle = min_angle
        self.scale = scale

    def deskew(self, img: np.ndarray, n_segments: int = 1) -> np.ndarray:
        """Apply deskew correction.

        Args:
            img: RGB document crop (H × W × 3, uint8).
            n_segments: Number of page segments from DocDetector.
                When 2, each vertical half is deskewed independently
                to handle spine-bend in open passports.

        Returns:
            Deskewed image (same shape), or original if no correction needed.
        """
        if n_segments == 2:
            return self._deskew_two_page(img)
        return self._deskew_single(img)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deskew_single(self, img: np.ndarray) -> np.ndarray:
        angle = self._find_angle(img)
        if abs(angle) < self.min_angle:
            return img
        return self._rotate(img, angle)

    def _deskew_two_page(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        mid = h // 2
        upper = img[:mid]
        lower = img[mid:]

        upper_out = self._deskew_single(upper)
        lower_out = self._deskew_single(lower)

        if upper_out is upper and lower_out is lower:
            return img
        return np.vstack([upper_out, lower_out])

    def _find_angle(self, img: np.ndarray) -> float:
        """Return the skew angle (degrees) that maximises projection variance."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()

        # Downscale for speed
        h, w = gray.shape[:2]
        sh = max(1, int(h * self.scale))
        sw = max(1, int(w * self.scale))
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)

        # Binarize: dark text on light background → invert so text pixels = 255
        _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        scores = np.empty(len(self.angles))
        cx, cy = sw / 2.0, sh / 2.0
        for i, angle in enumerate(self.angles):
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotated = cv2.warpAffine(binary, M, (sw, sh),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            scores[i] = rotated.sum(axis=1).astype(np.float64).var()

        best_idx = int(np.argmax(scores))
        # If the maximum is at the boundary of the search range, the projection
        # profile has no clear peak — most likely a false detection (stamp,
        # hologram, textured background).  Return 0 to skip correction.
        if best_idx == 0 or best_idx == len(self.angles) - 1:
            return 0.0

        return float(self.angles[best_idx])

    def _rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
