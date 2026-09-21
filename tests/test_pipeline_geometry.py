"""The whole pipeline: where a field read on the canvas lies on the photo.

Two kinds of check, and both are needed:

* with the models replaced by colour detectors - the geometry stages are the real ones
  (resize, quarter turns, perspective fix with the spread stitch, deskew, field and
  address-line crops), and the drawing says where the answer must land. Pages are drawn
  in perspective with tilted text, the way a photo taken by hand looks;
* on a real sample from samples/ - a synthetic mark of one pixel cannot show a
  half-pixel convention error, and that is the mistake this code invites. Here the
  patch the pipeline cut from the canvas is compared with the same field cut from the
  photo through the map: the residual shift between them is measured with sub-pixel
  precision, and half a pixel would be plain to see.
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from document_processing.geometry import Unknown
from document_processing.pipeline.pipeline import Pipeline
from document_processing.pipeline_modules import (
    AddressLinesDetector,
    DocDeskewer,
    DocDetector,
    DocTypeAngles,
    TextFieldsDetector,
    WordsDetector,
)

BACKGROUND, PAPER, INK = (40, 70, 110), (235, 235, 235), (25, 25, 25)
RED, BLUE, GREEN, BAND = (220, 30, 30), (30, 30, 220), (30, 200, 30), (170, 20, 20)
#: How far a corner may be from its place, in pixels of the photo: the photo is
#: downscaled more than twice on the way in and the edges of the colours blur.
ATOL = 15

SAMPLE = Path('../samples/INTPASSPORT_2011/6_CR_INTPASSPORT_2011.jpg')


def mask(image, color, tolerance=60):
    return np.all(np.abs(image.astype(int) - color) <= tolerance, axis=2).astype(np.uint8)


def module(cls, name, predict):
    instance = object.__new__(cls)
    instance.model_name, instance.model = name, SimpleNamespace(predict=predict)
    return instance


def pages_of(image):
    """"Segmentation": outer contours of the paper; text and fields inside are holes."""
    paper = (image.min(axis=2) > 150).astype(np.uint8)
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 0.02 * paper.size]
    return [cv2.boundingRect(c) for c in contours], [None] * len(contours), contours


def box_of(image, color, label):
    ys, xs = np.nonzero(mask(image, color))
    return [[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, 0.9, 0, label]] if len(xs) else []


def fields_of(image):
    return box_of(image, RED, 'Last_name_ru') + box_of(image, BLUE, 'Licence_number')


def words_of(patch):
    ys, xs = np.nonzero(mask(patch, GREEN))
    return [[float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1, 0.9]] if len(xs) else []


def lines_of(image):
    """An address line: a tilted band, turned so that the crop lands on it."""
    points = cv2.findNonZero(mask(image, BAND, tolerance=45))
    if points is None:
        return []
    (cx, cy), (w0, h0), degrees = cv2.minAreaRect(points)
    candidates = [
        (w, h, np.radians(angle))
        for angle in (degrees, -degrees, degrees + 90, -degrees - 90, degrees - 90, 90 - degrees)
        for w, h in ((w0, h0), (h0, w0)) if w >= h
    ]
    w, h, angle = max(
        candidates,
        key=lambda c: mask(AddressLinesDetector.crop_rotated(image, (cx, cy, c[0], c[1], c[2])), BAND, 45).mean(),
    )
    return [[cx, cy, w, h, angle, 0.9, 0, 'Address_line']]


def doctype(kind, angle, *, fallback=False):
    """Document type and turn.

    `fallback`: the type is not recognised on the photo with a background and is
    recognised on the straightened pages - the path where the borders run first. The
    turn is the one that puts the image it is given upright, so pages straightened out
    of an already turned photo need no turn of their own.
    """
    def predict(image):
        if not fallback:
            return (kind, 0.1, 1.0), (angle, 0.99)
        if mask(image, BACKGROUND, tolerance=20).mean() > 0.10:
            return ('NONE', 0.0, 1.0), (angle, 0.9)
        return (kind, 0.1, 1.0), (0, 0.99)
    return predict


def pipeline(kind, angle, *, fallback=False):
    stub = object.__new__(Pipeline)
    stub.probe, stub.ocr_device, stub.ocr_options = None, 'cpu', None
    stub.doctype_angles = module(DocTypeAngles, 'DocTypeAngles', doctype(kind, angle, fallback=fallback))
    stub.doc_detector = module(DocDetector, 'DocDetector', pages_of)
    stub.deskewer = DocDeskewer(angle_range=10.0, angle_steps=101, min_angle=2.0, scale=0.4)
    stub.text_fields = module(TextFieldsDetector, 'TextFieldsDetector', fields_of)
    stub.words_detector = module(WordsDetector, 'WordsDetector', words_of)
    stub.address_lines = module(AddressLinesDetector, 'AddressLinesDetector', lines_of)
    stub.address_textkind = SimpleNamespace(model_name='Kind', predict=lambda patch: {'Kind': ('printed', 0.1)})
    stub.ocr_cyr = SimpleNamespace(model_name='OCRCyrillic', predict=lambda word: {'OCRCyrillic': {'ocr_output': 'Г. МОСКВА'}})
    stub.ocr_lat = SimpleNamespace(model_name='OCRLatin', predict=lambda word: {'OCRLatin': {'ocr_output': 'G. MOSKVA'}})
    return stub


class Page:
    """A 1000 x 700 page in the perspective of the photo, with text tilted by `tilt` degrees."""

    def __init__(self, image, quad, tilt):
        source = np.float32([[0, 0], [1000, 0], [1000, 700], [0, 700]])
        self.matrix = cv2.getPerspectiveTransform(source, np.float32(quad))
        self.turn = cv2.getRotationMatrix2D((500, 350), -tilt, 1.0)
        self.image = image
        cv2.fillPoly(image, [np.int32(quad)], PAPER)

    def points(self, points):
        turned = cv2.transform(np.float32(points).reshape(-1, 1, 2), self.turn)
        return cv2.perspectiveTransform(turned, self.matrix).reshape(-1, 2)

    def box(self, x0, y0, x1, y1, color):
        corners = self.points([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        cv2.fillPoly(self.image, [np.int32(np.round(corners))], color)
        return corners

    def text(self, top, bottom):
        for row in range(top, bottom, 32):
            start, end = self.points([(90, row), (910, row)])
            cv2.line(self.image, tuple(np.int32(start)), tuple(np.int32(end)), INK, 7)


def assert_corners(found, expected):
    for corner in expected:
        assert min(np.linalg.norm(found - corner, axis=1)) <= ATOL, (found, expected)


def clockwise(points, height):
    """Points of an upright photo of `height` on the same photo turned clockwise: (x, y) -> (height - y, x)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.stack([height - points[:, 1], points[:, 0]], axis=1)


def spread_photo():
    """A passport spread photographed sideways: two pages in perspective, text tilted."""
    upright = np.empty((3400, 2400, 3), dtype=np.uint8)
    upright[:] = BACKGROUND
    upper = Page(upright, [(260, 180), (2150, 260), (2080, 1560), (330, 1500)], tilt=4)
    lower = Page(upright, [(300, 1760), (2120, 1700), (2200, 3150), (220, 3060)], tilt=4)
    upper.text(160, 660)
    lower.text(120, 660)
    name = upper.box(150, 50, 600, 110, RED)
    lower.box(880, 60, 940, 600, BLUE)
    word = lower.box(890, 90, 930, 220, GREEN)
    far_end = lower.points([(910, 570)])
    return cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE), name, word, far_end


@pytest.mark.parametrize('fallback', [False, True], ids=['type at once', 'type after borders'])
def test_a_spread_field_and_the_number_word_land_on_the_photo(fallback):
    """The photo is sideways: the type model asks for a quarter turn and the text stands up."""
    image, name, word, far_end = spread_photo()

    results = pipeline('INTPASSPORT_2011', 90, fallback=fallback).process_img(image, ocr=False, check_quality=False)

    (found,) = results.field_quads['Last_name_ru']
    assert_corners(found, clockwise(name, 3400))

    # The series/number patch is turned once before the words are found on it, so its
    # word must land on its own end of the field - not on the other end.
    (licence_word,) = results.word_quads['Licence_number']
    polygon = np.float32(licence_word).reshape(-1, 1, 2)
    word_centre = clockwise(word.mean(axis=0), 3400)[0]
    assert cv2.pointPolygonTest(polygon, tuple(map(float, word_centre)), False) >= 0
    assert cv2.pointPolygonTest(polygon, tuple(map(float, clockwise(far_end, 3400)[0])), False) < 0


def test_a_registration_address_line_lands_on_the_photo():
    image = np.empty((3000, 2200, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    page = Page(image, [(200, 300), (2000, 380), (1950, 2700), (260, 2600)], tilt=3)
    page.text(300, 660)
    band = page.box(120, 120, 760, 190, BAND)

    results = pipeline('INTPASSPORTADDR', 0).process_img(image, ocr=True, check_quality=False)

    assert [line['kind'] for line in results.meta_results['Address_lines']] == ['printed']
    (found,) = results.address_line_quads
    assert_corners(found, band)


def test_a_stage_with_no_way_back_leaves_no_quadrilaterals():
    """A canvas built with a bend map: recognition still works, the quadrilaterals do not.

    The stage here stands for one: it hands its image on and says the way back cannot be
    expressed. Nothing downstream may pretend otherwise.
    """
    image, _, _, _ = spread_photo()
    stub = pipeline('INTPASSPORT_2011', 90)
    straighten = stub.deskewer.deskew_with_geometry
    stub.deskewer.deskew_with_geometry = lambda img, **kw: (straighten(img, **kw)[0], Unknown())

    results = stub.process_img(image, ocr=False, check_quality=False)

    assert results.meta_results['TextFieldsDetector']['bbox'], 'the fields are still found'
    assert results.to_input([[10, 10]]) is None
    assert results.field_quads is None
    assert results.word_quads is None


@pytest.mark.skipif(not SAMPLE.exists(), reason='sample not in the tree')
def test_a_field_of_a_real_sample_is_cut_from_the_photo_by_its_quadrilateral():
    """The check a synthetic mark cannot make: the same pixels, to a fraction of a pixel.

    The pipeline cut the field out of its canvas; cutting the same field out of the
    photo through the quadrilateral must give the same picture. The residual shift
    between the two is found by correlation and refined on the parabola through the
    peak, so the half-pixel shift OpenCV's warps need - the usual mistake here - shows
    up as a shift of about 0.5 and not as a blur of the numbers.
    """
    image = cv2.imread(str(SAMPLE))
    assert image is not None

    results = Pipeline(model_format='ONNX', device='cpu', verbose=False).process_img(
        image, ocr=False, check_quality=False,
    )
    bboxes = results.meta_results['TextFieldsDetector']['bbox']
    patches = results.meta_results['TextFieldsDetector']['warped_img']
    quads = results.field_quads
    assert quads, 'the way back is known for this sample'

    def peak(response):
        """Position of the best match, refined to a fraction of a pixel."""
        _, best, _, (x, y) = cv2.minMaxLoc(response)

        def vertex(left, middle, right):
            denominator = left - 2 * middle + right
            return 0.0 if denominator == 0 else 0.5 * (left - right) / denominator

        dx = vertex(response[y, x - 1], response[y, x], response[y, x + 1]) if 0 < x < response.shape[1] - 1 else 0.0
        dy = vertex(response[y - 1, x], response[y, x], response[y + 1, x]) if 0 < y < response.shape[0] - 1 else 0.0
        return best, x + dx, y + dy

    checked = 0
    for label in ('Last_name_ru', 'Birth_date', 'Sex_ru'):
        found = [i for i, bbox in enumerate(bboxes) if bbox[-1] == label]
        if len(found) != 1 or label not in quads:
            continue
        patch = patches[found[0]]
        height, width = patch.shape[:2]
        margin = 4
        target = np.float32([
            [margin, margin], [width + margin, margin],
            [width + margin, height + margin], [margin, height + margin],
        ])
        matrix = cv2.getPerspectiveTransform(np.float32(quads[label][0]), target)
        cut = cv2.warpPerspective(image, matrix, (width + 2 * margin, height + 2 * margin))

        score, x, y = peak(cv2.matchTemplate(cut, patch, cv2.TM_CCOEFF_NORMED))
        assert score > 0.9, f'{label}: the quadrilateral does not cut out the same field ({score:.2f})'
        assert abs(x - margin) < 0.5 and abs(y - margin) < 0.5, \
            f'{label}: shift ({x - margin:+.2f}, {y - margin:+.2f}) px'
        checked += 1

    assert checked, 'none of the expected fields was read on the sample'
