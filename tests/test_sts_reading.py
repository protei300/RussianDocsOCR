"""Two reading rules for the vehicle registration certificate (STS).

- The special-marks lines are labelled tight to their letters, so the crop that
  is READ gets a vertical margin (Pipeline._read_margins) - never past halfway to
  the next line, and the box itself unchanged.
- The vehicle make is printed in Latin on the new form and in Cyrillic on the
  old one, so its engine follows the form year (OCROptionsClass.engine_by_year).

No models here.
"""
from types import SimpleNamespace

import numpy as np

from document_processing.geometry import Offset
from document_processing.pipeline.pipeline import OCROptionsClass, OCROptionsSTS, Pipeline


def canvas():
    img = np.zeros((200, 300, 3), np.uint8)
    img[:, :, 0] = np.arange(200)[:, None]        # row number in the red channel
    return img


def cut(img, boxes):
    return {'bbox': [list(b) for b in boxes],
            'warped_img': [img[b[1]:b[3], b[0]:b[2]] for b in boxes]}


def run(boxes, margin):
    img = canvas()
    fields = cut(img, boxes)
    frames = [Offset(-float(b[0]), -float(b[1])) for b in boxes]
    me = SimpleNamespace(ocr_options=SimpleNamespace(read_margin=margin))
    Pipeline._read_margins(me, fields, frames, img)
    return fields, frames


def test_the_read_crop_grows_and_the_box_does_not():
    box = [20, 100, 280, 120, 0.9, 0, 'Special_marks']
    fields, frames = run([box], {'Special_marks': 0.25})
    patch = fields['warped_img'][0]
    assert fields['bbox'][0] == box
    assert patch.shape[0] == 30                                # 20 px + 5 above + 5 below
    assert patch[0, 0, 0] == 95 and patch[-1, 0, 0] == 124
    assert frames[0].to_input(np.float32([[0, 0]]))[0, 1] == 95  # the frame moved with the crop


def test_the_margin_stops_halfway_to_the_next_line():
    upper = [20, 80, 280, 96, 0.9, 0, 'Special_marks']
    lower = [20, 100, 280, 116, 0.9, 0, 'Special_marks']
    fields, _ = run([upper, lower], {'Special_marks': 0.5})
    rows_upper = fields['warped_img'][0][:, 0, 0]
    rows_lower = fields['warped_img'][1][:, 0, 0]
    assert rows_upper.max() < rows_lower.min(), 'the two crops overlap'
    assert rows_upper.max() == 97 and rows_lower.min() == 98


def test_a_box_beside_it_does_not_limit_the_margin():
    marks = [20, 100, 140, 120, 0.9, 0, 'Special_marks']
    beside = [160, 80, 280, 98, 0.9, 0, 'Reg_number']         # no shared width
    fields, _ = run([marks, beside], {'Special_marks': 0.25})
    assert fields['warped_img'][0].shape[0] == 30


def test_other_fields_and_other_types_are_untouched():
    box = [20, 100, 280, 120, 0.9, 0, 'Vehicle_color']
    fields, frames = run([box], {'Special_marks': 0.25})
    assert fields['warped_img'][0].shape[0] == 20
    assert OCROptionsClass.read_margin == {} and OCROptionsClass.engine_by_year == {}


def test_the_make_engine_follows_the_sts_form():
    assert 'Vehicle_make_ru' in OCROptionsSTS.ru_fields          # the old form's route
    assert set(OCROptionsSTS.read_margin) == {'Special_marks'}
    engine = lambda year, field='Vehicle_make_ru': Pipeline._engine_by_year(
        SimpleNamespace(ocr_options=OCROptionsSTS, _doc_year=year), field)
    assert engine('2019') == 'lat', 'the new form prints the make in Latin'
    assert engine('1996') is None, 'the old form keeps the Cyrillic route'
    assert engine('2019', 'Vehicle_color') is None

