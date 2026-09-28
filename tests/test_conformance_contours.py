"""Tests for the R-01 contour comparison (conformance/runner/compare.py).

The rule has to hold two ways at once, and both are asserted here:

1. **One mask pixel is not a difference.** The outline is traced from a mask
   thresholded at 0.5, so a pixel sitting at the threshold may land on either side
   depending on the machine's arithmetic. That flip failed CI on 2026-09-28 through
   the centroid (1.353e-3 against the old 1e-3), and the vertex-set Hausdorff
   distance was one flip away from failing as well.
2. **A real change is still a difference.** A page moved by one pixel, or a
   different outline, must fail.

Uses the committed golden of INTPASSPORT_2011-12 (the case that flipped), so no
models are needed.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from conformance.runner.compare import (CPU, _hausdorff, _polygon_centroid,
                                        compare_contours)

GOLDEN = (Path(__file__).resolve().parents[1] / "conformance" / "cases"
          / "INTPASSPORT_2011-12_CR_INTPASSPORT_2011" / "stages" / "borders.segments.json")


def _trace(mask: np.ndarray) -> list:
    """The outline the library would emit for this mask (largest external contour)."""
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    c = contours[int(np.argmax([len(x) for x in contours]))]
    return c.reshape(-1, 2).astype(float).tolist()


def _vertex_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


@pytest.fixture(scope="module")
def page():
    """Page 1 of the golden, plus the mask it was traced from."""
    if not GOLDEN.exists():
        # The public tree withdrew this case (conformance/deviations.json, W-01): the
        # sample was a photograph of a real document, and only case.json remains.
        pytest.skip("golden of INTPASSPORT_2011-12 is absent here (case withdrawn, W-01)")
    outline = json.loads(GOLDEN.read_text(encoding="utf-8"))[1]
    pts = np.asarray(outline, dtype=np.int32)
    mask = np.zeros((int(pts[:, 1].max()) + 3, int(pts[:, 0].max()) + 3), np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    assert _trace(mask) == outline, "rasterising the golden must reproduce it exactly"
    return outline, mask


def test_identical_outline_passes(page):
    outline, _ = page
    assert compare_contours([outline], [outline], CPU) == []


def test_one_flipped_mask_pixel_passes(page):
    outline, mask = page
    golden = np.asarray(outline, float)
    gc = np.array(_polygon_centroid(golden))
    boundary = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    # Find a boundary pixel whose flip trips BOTH old rules, so the test exercises
    # exactly what the new ones relax.
    for x, y in boundary.reshape(-1, 2):
        flipped = mask.copy()
        flipped[y, x] = 0
        traced = _trace(flipped)
        moved = np.asarray(traced, float)
        if (np.abs(np.array(_polygon_centroid(moved)) - gc).max() > 1e-3
                and _vertex_hausdorff(golden, moved) > 1.0):
            break
    else:
        pytest.fail("no single-pixel flip trips the old rules; the fixture changed")
    assert compare_contours([outline], [traced], CPU) == []


def test_page_moved_by_one_pixel_fails(page):
    outline, _ = page
    moved = (np.asarray(outline) + [1.0, 0.0]).tolist()
    diffs = compare_contours([outline], [moved], CPU)
    assert [d.path for d in diffs] == ["borders.segments[0].centroid"]


def test_different_outline_fails(page):
    outline, _ = page
    pts = np.asarray(outline, float)
    c = np.array(_polygon_centroid(pts))
    # Grown by 2 % about its own centroid: the centroid stays, the shape does not.
    grown = (c + (pts - c) * 1.02).tolist()
    paths = {d.path for d in compare_contours([outline], [grown], CPU)}
    assert paths == {"borders.segments[0].area", "borders.segments[0].hausdorff"}


def test_hausdorff_is_measured_to_the_outline_not_the_vertices():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    # The same square with an extra vertex in the middle of one edge.
    resampled = np.array([[0, 0], [5, 0], [10, 0], [10, 10], [0, 10]], float)
    assert _vertex_hausdorff(square, resampled) == 5.0
    assert _hausdorff(square, resampled) == 0.0
    assert compare_contours([square.tolist()], [resampled.tolist()], CPU) == []


def test_hausdorff_of_exactly_the_allowance_passes():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    # One corner pulled out by exactly 1 px: the outline distance is exactly 1.0.
    notched = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 5], [-1, 5], [0, 5]]
    assert _hausdorff(np.asarray(square, float), np.asarray(notched, float)) == 1.0
    assert not [d for d in compare_contours([square], [notched], CPU)
                if d.path.endswith(".hausdorff")]
