"""Every stage that changes the image knows the way back to its input.

Checked with the library's own functions and OpenCV on drawings with a mark: the mark
found on the output of a stage must land where it was drawn after the map is applied.
A synthetic mark is the only ground truth that does not come from the maps themselves.

COORDINATES ARE CONTINUOUS: the pixel with index (i, j) is the unit square centred at
(i + 0.5, j + 0.5). Tests state the expected point with that half added, which is also
what keeps them honest about the half-pixel shift OpenCV's warps need.
"""

import cv2
import numpy as np
import pytest

from document_processing.geometry import Chain, Offset, Pieces, QuarterTurns, Scale, Unknown
from document_processing.pipeline_modules import AddressLinesDetector, DocDeskewer
from document_processing.pipeline_modules.doc_detector.image_transformation import fix_perspective

BACKGROUND = (40, 70, 110)
PAPER = (235, 235, 235)
MARK = (230, 20, 20)


def canvas(width, height, color=BACKGROUND):
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def marks(image, color=MARK, tolerance=70):
    """Centres of the marks of `color`, one per connected blob, in continuous coordinates."""
    close = np.all(np.abs(image.astype(int) - color) <= tolerance, axis=2).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(close)
    found = [centroids[i] + 0.5 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 5]
    assert found, 'no mark on the image'
    return found


def page(image, quad):
    """A page on the photo and its contour, in the shape the segmentation returns."""
    points = np.asarray(quad, dtype=np.int32)
    cv2.fillPoly(image, [points], PAPER)
    return points.reshape(-1, 1, 2)


def assert_lands(found, expected, atol):
    for point in expected:
        nearest = min(found, key=lambda item: np.linalg.norm(item - point))
        np.testing.assert_allclose(nearest, point, atol=atol)


def lines(image, *, top, bottom, degrees):
    for row in range(top, bottom, 28):
        rise = int(round(640 * np.tan(np.radians(degrees))))
        cv2.line(image, (80, row), (720, row + rise), (20, 20, 20), 5)


@pytest.mark.parametrize('turns', [0, 1, 2, 3])
def test_quarter_turns_put_the_pixel_back(turns):
    """Against cv2.rotate itself: a turn count of its own is the easiest thing to get wrong by one."""
    image = np.zeros((4, 7), dtype=np.uint8)
    image[1, 5] = 255
    for _ in range(turns):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ys, xs = np.nonzero(image)
    np.testing.assert_allclose(QuarterTurns(7, 4, turns).to_input([[xs[0] + 0.5, ys[0] + 0.5]]), [[5.5, 1.5]])


def test_chain_undoes_the_last_stage_first():
    chain = Chain((Scale(0.5, 0.25), Offset(10, 20)))
    np.testing.assert_allclose(chain.to_input([[15, 45]]), [[10, 100]])
    # A stage that passed its image on unchanged adds nothing to the chain.
    assert chain.then(None) is chain


def test_a_stage_with_no_way_back_makes_the_whole_chain_answer_none():
    """The contract a canvas built with a bend map depends on.

    A map that cannot be inverted must not be silently dropped: the stages around it
    are still correct, so the chain would keep answering - with a point that quietly
    belongs to another place on the photo. The caller draws it over a document.
    """
    chain = Chain((Scale(0.5, 0.5), Unknown(), Offset(3, 4)))

    assert chain.to_input([[10, 10]]) is None
    # Not identity: the stage did change the image, it is the way back that is gone.
    assert chain.then(Unknown()).to_input([[10, 10]]) is None
    assert Pieces((((0, 0, 10, 10), Unknown()),)).to_input([[1, 1]]) is None
    # And a chain without it still answers.
    np.testing.assert_allclose(Chain((Scale(0.5, 0.5),)).to_input([[10, 10]]), [[20, 20]])


def test_a_straightened_page_puts_the_mark_back_on_the_photo():
    image = canvas(900, 700)
    contour = page(image, [(120, 90), (760, 140), (700, 610), (90, 560)])
    cv2.circle(image, (400, 300), 5, MARK, -1)

    warped, _, geometry = fix_perspective(image, [contour], return_geometry=True)
    plain = fix_perspective(image, [contour])

    # Without the flag the function answers exactly as it did before.
    assert len(plain) == 2 and np.array_equal(plain[0], warped)
    assert_lands([geometry.to_input(point)[0] for point in marks(warped)], [(400.5, 300.5)], atol=1.0)


@pytest.mark.parametrize(
    ('size', 'quads', 'points'),
    [
        ((1024, 760), [[(40, 60), (430, 80), (420, 640), (50, 620)], [(480, 90), (960, 70), (980, 700), (470, 660)]],
         [(200, 300), (700, 400)]),
        ((800, 1040), [[(60, 40), (700, 60), (690, 420), (70, 400)], [(80, 470), (720, 460), (740, 1000), (60, 980)]],
         [(300, 200), (400, 700)]),
    ],
    ids=['side by side', 'one above the other'],
)
def test_a_stitched_spread_puts_the_mark_of_each_page_back(size, quads, points):
    """Both pages, because a stitched canvas picks the piece by the centroid of the shape.

    Two pages of different size are resized to a common side before the stitch, so a
    box that is mapped through the wrong piece lands plausibly - on the other page.
    """
    image = canvas(*size)
    contours = [page(image, quad) for quad in quads]
    for x, y in points:
        cv2.circle(image, (x, y), 5, MARK, -1)

    warped, _, geometry = fix_perspective(image, contours, return_geometry=True)

    assert isinstance(geometry, Pieces)
    found = [geometry.to_input(point)[0] for point in marks(warped)]
    assert len(found) == 2
    assert_lands(found, [(x + 0.5, y + 0.5) for x, y in points], atol=1.5)


def test_deskew_puts_the_mark_back():
    image = np.full((600, 800, 3), 245, dtype=np.uint8)
    lines(image, top=40, bottom=540, degrees=5.0)
    cv2.circle(image, (433, 287), 5, MARK, -1)
    deskewer = DocDeskewer(angle_range=10.0, angle_steps=101, min_angle=2.0, scale=0.4)

    straight, geometry = deskewer.deskew_with_geometry(image)

    assert geometry is not None, '5 degrees is above min_angle, so the image is turned'
    # deskew() keeps answering with the image alone, pixel for pixel.
    assert np.array_equal(straight, deskewer.deskew(image))
    assert_lands([geometry.to_input(point)[0] for point in marks(straight)], [(433.5, 287.5)], atol=1.0)

    flat = np.full((600, 800, 3), 245, dtype=np.uint8)
    same, none = deskewer.deskew_with_geometry(flat)
    # Nothing to turn: the image is passed on as it is, and the stage adds no map.
    assert same is flat and none is None


def test_deskew_of_two_halves_puts_the_mark_of_each_half_back():
    image = np.full((1000, 800, 3), 245, dtype=np.uint8)
    lines(image, top=40, bottom=420, degrees=5.0)
    lines(image, top=560, bottom=940, degrees=-4.0)
    cv2.circle(image, (400, 250), 5, MARK, -1)
    cv2.circle(image, (380, 760), 5, MARK, -1)
    deskewer = DocDeskewer(angle_range=10.0, angle_steps=101, min_angle=2.0, scale=0.4)

    straight, geometry = deskewer.deskew_with_geometry(image, n_segments=2)

    assert isinstance(geometry, Pieces)
    assert_lands([geometry.to_input(point)[0] for point in marks(straight)], [(400.5, 250.5), (380.5, 760.5)], atol=1.0)


@pytest.mark.parametrize('degrees', [0.0, 7.0, -12.0])
def test_a_rotated_address_line_crop_puts_the_mark_back_on_the_page(degrees):
    image = canvas(640, 480, color=PAPER)
    cv2.circle(image, (350, 250), 3, MARK, -1)
    obbox = (330.0, 240.0, 220.0, 60.0, np.radians(degrees))

    patch = AddressLinesDetector.crop_rotated(image, obbox)
    geometry = AddressLinesDetector.crop_geometry(obbox)

    assert_lands([geometry.to_input(point)[0] for point in marks(patch)], [(350.5, 250.5)], atol=1.0)
