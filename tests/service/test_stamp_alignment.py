"""Stamp tilt, expanded-canvas geometry and the live field-reader boundary."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from document_processing.pipeline_modules.registration_stamps.alignment import (
    align_stamp, estimate_stamp_angle,
)
from service.ml.registration.config import RegistrationConfig
from service.ml.registration.ocr import CTCAlphabet
from service.ml.registration.stamp import StampReader


def stamp(angle=0., ruling=True):
    image = np.full((300, 600, 3), 245, np.uint8)
    for y in (40, 95, 150, 205, 260):
        if ruling:
            cv2.line(image, (25, y), (575, y), (30, 30, 30), 2)
        else:
            cv2.putText(image, 'ADDRESS 123', (35, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .8, (25, 25, 25), 2)
    matrix = cv2.getRotationMatrix2D((300, 150), -angle, 1.)
    return cv2.warpAffine(image, matrix, (600, 300), borderValue=(245, 245, 245))


@pytest.mark.parametrize('angle', [-5., -3., 3., 5.])
@pytest.mark.parametrize('ruling', [False, True])
def test_corrects_both_signs_and_keeps_all_original_corners(angle, ruling):
    image = stamp(angle, ruling)
    before = image.copy()
    aligned, backward, correction = align_stamp(image)
    assert abs(correction-angle) < .35
    assert abs(estimate_stamp_angle(aligned)) < .35
    h, w = aligned.shape[:2]
    corners = np.array([[0, 0, 1], [599, 0, 1], [599, 299, 1], [0, 299, 1.]])
    forward = np.linalg.inv(backward)
    projected = corners @ forward.T
    assert np.all(projected[:, :2] >= 0)
    assert np.all(projected[:, 0] < w) and np.all(projected[:, 1] < h)
    np.testing.assert_allclose(projected @ backward.T, corners, atol=1e-10)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('image', [np.zeros((100, 200, 3), np.uint8),
    np.full((100, 200, 3), 255, np.uint8), stamp(), stamp(.2),
    np.zeros((4, 10, 3), np.uint8)])
def test_no_op_keeps_exact_pixels(image):
    aligned, backward, correction = align_stamp(image)
    assert aligned is image and correction == 0
    np.testing.assert_array_equal(backward, np.eye(3))


@pytest.mark.parametrize('fail_at', [None, 'detector', 'ocr'])
def test_reader_detects_on_aligned_stamp_maps_boxes_back_and_wipes(fail_at):
    image = stamp(3)
    reader = object.__new__(StampReader)
    reader.cfg = RegistrationConfig()
    captured = {}; q = np.array([[80, 70], [160, 70], [160, 95], [80, 95.]])
    expected, backward, _ = align_stamp(image)

    def detect(img):
        captured['image'] = img
        np.testing.assert_array_equal(img, expected)
        if fail_at == 'detector': raise RuntimeError('detector')
        return [dict(field='date', conf=.9, quad=q)]

    def probs(patch):
        if fail_at == 'ocr': raise RuntimeError('ocr')
        return np.array([[0, 1]], np.float32)

    reader.det = SimpleNamespace(predict=detect)
    reader.ocr = SimpleNamespace(abc=CTCAlphabet('A'), probs=probs)
    if fail_at:
        with pytest.raises(RuntimeError, match=fail_at): reader.read(image)
    else:
        result = reader.read(image)
        wanted = np.c_[q, np.ones(4)] @ backward.T
        np.testing.assert_allclose(result['boxes'][0]['quad'], wanted[:, :2], atol=.051)
        assert result['fields'] == {'date': 'A'}
        # The existing page crop->photo transform composes after this map.
        np.testing.assert_allclose(np.asarray(result['boxes'][0]['quad'])+[170, 250],
                                   wanted[:, :2]+[170, 250], atol=.051)
    assert not captured['image'].any()
    assert image.any()


def test_switch_preserves_original_detector_input(monkeypatch):
    monkeypatch.setenv('RDOCS_STAMP_DESKEW', '0')
    reader = object.__new__(StampReader); reader.cfg = RegistrationConfig()
    assert reader.cfg.stamp_deskew is False
    image = stamp(3)
    seen = []
    reader.det = SimpleNamespace(predict=lambda img: seen.append(img) or [])
    assert reader.read(image) == {'fields': {}, 'boxes': []}
    assert seen[0] is image and image.any()


def test_page_maps_each_independently_aligned_stamp_through_photo_homography():
    from service.ml.registration.page import RegistrationReader
    from document_processing.pipeline_modules import AddressLinesDetector
    page = RegistrationReader()
    reader = object.__new__(StampReader); reader.cfg = RegistrationConfig()
    page._stamps = reader
    q = np.array([[90, 70], [190, 70], [190, 95], [90, 95.]])
    seen = []
    reader.det = SimpleNamespace(predict=lambda image: seen.append(image) or [
        dict(field='date', conf=.9, quad=q)])
    reader.ocr = SimpleNamespace(abc=CTCAlphabet('A'), probs=lambda _: np.array([[0, 1]], np.float32))
    canvas = np.hstack([stamp(3), stamp(-3)])
    photo_matrix = np.array([[1.2, .04, 60], [.02, .9, 130], [.0001, .0002, 1.]])

    def to_photo(points):
        pts = np.c_[points, np.ones(len(points))] @ photo_matrix.T
        return pts[:, :2]/pts[:, 2:]

    obbs = [[299.5, 149.5, 600, 300, 0, .99, 0, 'registered'],
            [899.5, 149.5, 600, 300, 0, .99, 0, 'registered']]
    crops = page._stamp_crops(canvas, to_photo, [dict(x0=0, x1=1200, decision='registration')],
                              obbs, SimpleNamespace(KINDS={'registered': 'registered'}))
    expected = []
    for item, ob in zip(crops, obbs):
        _, backward, _ = align_stamp(item['crop_bgr'])
        native = np.c_[q, np.ones(4)] @ backward.T
        # The stamp reader rounds original-crop boxes to 0.1 px.
        native = native[:, :2].round(1)
        on_canvas = AddressLinesDetector.crop_geometry(ob[:5]).to_input(native)
        expected.append(to_photo(on_canvas).round(1))
    answer = page.read_stamps(dict(stamps=crops, quality={}, pages=[], decision='registration',
        timings={}, doc_type='INTPASSPORTADDR_ALL', types_in_frame=['INTPASSPORTADDR_ALL']))
    by_index = {s['index']: s for s in answer['stamps']}
    for index, quad in enumerate(expected, 1):
        np.testing.assert_allclose(by_index[index]['boxes'][0]['quad'], quad, atol=1e-8)
    assert len(seen) == 2 and all(not image.any() for image in seen)
