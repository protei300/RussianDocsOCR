"""
Projection-profile deskew for document images.

After perspective correction, pages can still have a residual tilt. This
module finds and corrects that tilt by maximising the variance of horizontal
projection profiles — text lines create sharp peaks when horizontal and
smeared peaks when tilted.

Works entirely with OpenCV / NumPy — no templates required.
"""

import cv2
import numpy as np


class DocDeskewer:
    """Correct residual tilt in perspective-corrected document images.

    Note: the constructor defaults below are this class's own defaults for
    standalone use. `Pipeline` overrides them (`angle_range=10.0,
    angle_steps=101, min_angle=2.0, scale=0.4`) - see `Pipeline.__init__`.

    Args:
        angle_range: Search range in degrees, symmetric around 0. Default ±2°.
        angle_steps: Number of candidate angles the brute-force resolution
            would use (e.g. 81 -> ~0.05° resolution); actual search is
            coarse-to-fine (see `coarse_steps` and `_find_angle`), using far
            fewer `cv2.warpAffine` calls than a full scan at this resolution
            while picking the same angle. Default 81.
        min_angle: Skip rotation if detected angle is below this threshold (deg).
            Avoids unnecessary resampling on already-flat documents. Default 0.5°.
        scale: Fraction of original resolution used for angle search. Default 0.3.
    """

    def __init__(
        self,
        angle_range: float = 2.0,
        angle_steps: int = 81,
        min_angle: float = 0.5,
        scale: float = 0.3,
        coarse_steps: int = 21,
    ):
        """
        Args (new):
            coarse_steps: number of angles in a first, coarse search pass over
                the full [-angle_range, angle_range] window. The best coarse
                candidate is then refined with a second pass, at the same
                resolution the brute-force `angle_steps` would give, but only
                within one coarse step's width around it (not over the whole
                range) - see `_find_angle`. This cuts the number of
                `warpAffine` calls from `angle_steps` to roughly
                `coarse_steps + (2 * coarse_step_deg / fine_resolution)`
                (measured ~3x fewer on the default params below), while
                matching the brute-force result whenever the projection-
                variance curve is unimodal in the coarse window - true for
                genuine single-peak skew; verified to pick the identical angle
                as the brute-force search on samples/ (see
                docs/progress-log.md).
        """
        self.angle_range = angle_range
        self.angle_steps = angle_steps
        self.min_angle = min_angle
        self.scale = scale

        self.angles = np.linspace(-angle_range, angle_range, angle_steps)  # kept for introspection/back-compat

        self.coarse_steps = max(3, min(coarse_steps, angle_steps))
        self.coarse_angles = np.linspace(-angle_range, angle_range, self.coarse_steps)
        coarse_step_deg = (2 * angle_range) / (self.coarse_steps - 1) if self.coarse_steps > 1 else angle_range
        full_res = (2 * angle_range) / (angle_steps - 1) if angle_steps > 1 else angle_range
        self._fine_half_range = coarse_step_deg
        self._fine_count = max(3, int(round(2 * coarse_step_deg / full_res)) + 1)

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
        """Return the skew angle (degrees) that maximises projection variance.

        Two-pass coarse-to-fine search instead of a brute-force scan over all
        `angle_steps` angles: a coarse pass over the full range locates the
        peak's neighborhood cheaply, then a fine pass refines within one
        coarse step of it at the original target resolution. Far fewer
        `warpAffine` calls for the same effective resolution (see __init__).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()

        # Downscale for speed
        h, w = gray.shape[:2]
        sh = max(1, int(h * self.scale))
        sw = max(1, int(w * self.scale))
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)

        # Binarize: dark text on light background → invert so text pixels = 255
        _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cx, cy = sw / 2.0, sh / 2.0

        def score_angles(angles):
            scores = np.empty(len(angles))
            for i, angle in enumerate(angles):
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                rotated = cv2.warpAffine(binary, M, (sw, sh),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
                scores[i] = rotated.sum(axis=1).astype(np.float64).var()
            return scores

        # Coarse pass over the full range.
        coarse_scores = score_angles(self.coarse_angles)
        coarse_best_idx = int(np.argmax(coarse_scores))
        # If the maximum is at the boundary of the full search range, the
        # projection profile has no clear peak — most likely a false
        # detection (stamp, hologram, textured background). Return 0 to skip
        # correction (same rule as the original brute-force search).
        if coarse_best_idx == 0 or coarse_best_idx == len(self.coarse_angles) - 1:
            return 0.0
        coarse_best_angle = float(self.coarse_angles[coarse_best_idx])

        # Fine pass: refine within +/- one coarse step, at the resolution the
        # brute-force angle_steps would give over the full range.
        lo = max(-self.angle_range, coarse_best_angle - self._fine_half_range)
        hi = min(self.angle_range, coarse_best_angle + self._fine_half_range)
        fine_angles = np.linspace(lo, hi, self._fine_count)
        fine_scores = score_angles(fine_angles)
        fine_best_idx = int(np.argmax(fine_scores))

        return float(fine_angles[fine_best_idx])

    def _rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
