"""Where a point of a pipeline canvas lies on the input image.

A caller that shows a document to a person needs the field on the PHOTO, not on the
canvas the pipeline built: the canvas is resized, turned upright, warped page by page,
stitched and deskewed, and none of those transforms were kept, so the way back could
only be reconstructed by repeating the pipeline's own logic outside it.

Every stage that changes the geometry of the image it passes on describes that
change as a map from its OUTPUT back to its INPUT:

  * Pipeline._prepare_image              - resize to img_size             (Scale)
  * DocTypeAngles.predict_transform      - quarter turns to upright       (QuarterTurns)
  * fix_perspective                      - four-point warp of each page,
                                           resize and stitch of a spread  (Homography, Pieces)
  * DocDeskewer.deskew_with_geometry     - residual tilt                  (Homography)
  * AddressLinesDetector.crop_geometry   - rotated address-line crop      (Chain)

The pipeline chains the maps of the stages that actually ran, in the order they ran
(``PipelineResults.geometry``), so any point of the final canvas maps back to the
image passed to ``Pipeline.process_img``.

COORDINATES ARE CONTINUOUS PIXELS. An image spans [0, width] x [0, height], and the
pixel with index (i, j) is the unit square centred at (i + 0.5, j + 0.5); box edges
from the detectors are read the same way. OpenCV's warps put pixel centres on
integer coordinates instead, so ``Homography`` shifts by half a pixel on the way in
and out and the matrices OpenCV computed are used unchanged. ``cv2.resize`` and
``cv2.rotate`` need no shift in these coordinates.

A MAP RECEIVES THE POINTS OF ONE SHAPE. A stitched canvas picks the piece a shape
lies on by the shape's centroid, so the corners of one box are never sent to
different pages.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ['Geometry', 'Scale', 'Offset', 'QuarterTurns', 'Homography', 'Chain', 'Pieces',
           'Unknown', 'corners']


def corners(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Corners of an axis-aligned box, clockwise from the top-left, as a (4, 2) array."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _points(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


class Geometry:
    """A map from the output of a stage back to its input.

    ``to_input`` answers None when the way back is not known. NOT KNOWN IS NOT THE
    SAME AS UNCHANGED: a stage that passes its image on untouched contributes no map
    at all (``Chain.then(None)``), while a stage whose effect no point map can express
    contributes ``Unknown()`` and makes every answer downstream None. A box that
    quietly lands somewhere else is worse than no box: the caller draws it over a
    photo and trusts it.
    """

    def to_input(self, points) -> Optional[np.ndarray]:
        """Points of the output, (N, 2), on the input; None - the way back is not known."""
        raise NotImplementedError


@dataclass(frozen=True)
class Scale(Geometry):
    """The input resized: output = input * (sx, sy)."""

    sx: float
    sy: float

    def to_input(self, points) -> Optional[np.ndarray]:
        return _points(points) / (self.sx, self.sy)


@dataclass(frozen=True)
class Offset(Geometry):
    """The input shifted: output = input + (dx, dy); a crop is a negative shift."""

    dx: float
    dy: float

    def to_input(self, points) -> Optional[np.ndarray]:
        return _points(points) - (self.dx, self.dy)


@dataclass(frozen=True)
class QuarterTurns(Geometry):
    """``cv2.ROTATE_90_COUNTERCLOCKWISE`` applied ``turns`` times to a width x height input."""

    width: int
    height: int
    turns: int

    def to_input(self, points) -> Optional[np.ndarray]:
        points = _points(points)
        widths = []
        w, h = self.width, self.height
        for _ in range(self.turns % 4):
            widths.append(w)
            w, h = h, w
        # One turn sends (x, y) of a W-wide image to (y, W - x); the last turn is undone first.
        for w in reversed(widths):
            points = np.stack([w - points[:, 1], points[:, 0]], axis=1)
        return points


@dataclass(frozen=True, eq=False)
class Homography(Geometry):
    """The output of ``cv2.warpPerspective`` or ``cv2.warpAffine`` with ``matrix`` (input -> output).

    A 2x3 affine matrix is completed to 3x3.
    """

    matrix: np.ndarray

    def to_input(self, points) -> Optional[np.ndarray]:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape == (2, 3):
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
        shifted = _points(points) - 0.5
        mapped = np.hstack([shifted, np.ones((len(shifted), 1))]) @ np.linalg.inv(matrix).T
        return mapped[:, :2] / mapped[:, 2:3] + 0.5


@dataclass(frozen=True)
class Unknown(Geometry):
    """A stage that changed the image in a way no point map expresses.

    A bend map straightens a curved page pixel by pixel; there is no matrix for it and
    no inverse to compose. The canvas is still correct and recognition is unaffected -
    only the way back is gone, and it stays gone for every stage after this one.
    """

    def to_input(self, points) -> Optional[np.ndarray]:
        return None


@dataclass(frozen=True)
class Chain(Geometry):
    """Maps of stages in the order the stages ran: the first one reads the input."""

    maps: Tuple[Geometry, ...] = ()

    def then(self, later: Optional[Geometry]) -> 'Chain':
        """This chain followed by one more stage; None - the stage passed its input on unchanged."""
        return self if later is None else Chain(self.maps + (later,))

    def to_input(self, points) -> Optional[np.ndarray]:
        points = _points(points)
        for geometry in reversed(self.maps):
            points = geometry.to_input(points)
            # One stage that cannot answer ends the walk: the stages before it are
            # fine, but their input is no longer known.
            if points is None:
                return None
        return points


@dataclass(frozen=True)
class Pieces(Geometry):
    """A canvas made of pieces: a rectangle (x0, y0, x1, y1) of the output and the map of what fills it."""

    pieces: Tuple[Tuple[Tuple[float, float, float, float], Geometry], ...]

    def to_input(self, points) -> Optional[np.ndarray]:
        points = _points(points)
        cx, cy = points.mean(axis=0)

        def distance(piece) -> float:
            (x0, y0, x1, y1), _ = piece
            dx = max(x0 - cx, 0.0, cx - x1)
            dy = max(y0 - cy, 0.0, cy - y1)
            return dx * dx + dy * dy

        _, geometry = min(self.pieces, key=distance)
        return geometry.to_input(points)  # None if that piece cannot answer
