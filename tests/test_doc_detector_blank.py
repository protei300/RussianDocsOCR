"""Blank-segment rule of the document detector: a segment without ink (the
white lid of a flatbed scanner, a blank sheet) is dropped from the page
selection when an inked segment exists, and kept when it is all there is.
Synthetic, no model."""
import cv2
import numpy as np

from document_processing.pipeline_modules.doc_detector import doc_detector as dd


def _scene():
    """1500x1000 photo: a blank white 'lid' on the left, a 'page' with text
    lines and rules on the right, both on a dark table."""
    img = np.full((1000, 1500, 3), 40, np.uint8)
    cv2.rectangle(img, (20, 20), (700, 980), (245, 245, 245), -1)            # lid
    cv2.rectangle(img, (780, 100), (1450, 900), (230, 228, 225), -1)         # page
    for y in range(160, 880, 48):
        cv2.line(img, (820, y), (1400, y), (60, 60, 60), 2)
        for x in range(830, 1380, 26):
            cv2.rectangle(img, (x, y - 20), (x + 16, y - 6), (50, 50, 50), -1)
    lid = np.float32([[20, 20], [700, 20], [700, 980], [20, 980]])
    page = np.float32([[780, 100], [1450, 100], [1450, 900], [780, 900]])
    return img, lid, page


def test_ink_separates_blank_from_page():
    img, lid, page = _scene()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    assert dd.segment_ink(gray, lid) < dd.BLANK_INK
    assert dd.segment_ink(gray, page) > 3 * dd.BLANK_INK


def test_blank_segment_dropped_only_next_to_an_inked_one():
    img, lid, page = _scene()
    keep, ink = dd.drop_blank_segments(img, [lid, page])
    assert keep == [1] and ink[0] < dd.BLANK_INK < ink[1]
    keep, _ = dd.drop_blank_segments(img, [lid])
    assert keep == [0]                     # a lone blank sheet still goes through
    keep, _ = dd.drop_blank_segments(img, [page, page.copy()])
    assert keep == [0, 1]
