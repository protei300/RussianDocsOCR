"""The passport series/number is read by the CYRILLIC engine, and that is load-bearing.

Issue #12 (public repo): the Latin engine reads the red '3' of a passport series
as '8' -- confidently, at p=0.94..1.00 with '3' as the runner-up at 0.004, so no
threshold, alphabet mask or upscaling recovers it. The Cyrillic engine reads the
SAME crops correctly. Measured over samples/: Latin 103/113 exact on
Licence_number, Cyrillic 111/113, and Cyrillic is never worse on any doc type
except BIRTHCERT, whose series is a Roman numeral.

The conformance suite does NOT cover this: both of its passport cases are
documents the Latin engine happened to read correctly, so the two engines agree
there and a regression would pass 44/44. Hence these tests, on the documents that
actually failed.
"""
from pathlib import Path

import pytest

from document_processing import Pipeline
from document_processing.pipeline.pipeline import (OCROptionsEXTPassport,
                                                   OCROptionsINTPassport)

SAMPLES = Path(__file__).resolve().parent.parent / 'samples'
IMG_EXTS = ('.jpg', '.jpeg', '.png')

# Documents whose number the Latin engine got wrong, all of them 3 -> 8.
REGRESSIONS = [
    ('INTPASSPORT_1997/25_BG_INTPASSPORT_2001', '45 09 400234'),
    ('INTPASSPORT_2011/5_BG_INTPASSPORT_2011', '53 13 314243'),
]


class TestRouting:
    """Cheap structural guard: the routing itself, without running a model."""

    @pytest.mark.parametrize('options', [OCROptionsINTPassport, OCROptionsEXTPassport])
    def test_licence_number_goes_to_the_cyrillic_engine(self, options):
        assert 'Licence_number' in options.ru_fields
        assert 'Licence_number' not in options.en_fields


@pytest.fixture(scope='module')
def pipeline():
    return Pipeline(model_format='ONNX', device='cpu', ocr='accurate', verbose=False)


@pytest.mark.parametrize('rel,expected', REGRESSIONS)
def test_series_number_is_read_correctly(pipeline, rel, expected):
    """End-to-end, on the documents the defect was found on."""
    # Extension-explicit on purpose: globbing '<name>.*' also matches the
    # ground-truth <name>.json sitting next to the image, and which of the two
    # comes first is filesystem order - this passed on Windows and failed on CI.
    stem = SAMPLES / rel
    image = next(p for ext in IMG_EXTS for p in [stem.with_suffix(ext)] if p.exists())
    results = pipeline.process_img(str(image), ocr=True, check_quality=False)
    assert results.ocr.get('Licence_number') == expected
