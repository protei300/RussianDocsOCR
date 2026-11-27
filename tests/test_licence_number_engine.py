"""The passport series/number is read by the CYRILLIC engine, and that is load-bearing.

Issue #12 (public repo): the Latin engine reads the red '3' of a passport series
as '8' -- confidently, at p=0.94..1.00 with '3' as the runner-up at 0.004, so no
threshold, alphabet mask or upscaling recovers it. The Cyrillic engine reads the
SAME crops correctly.
"""
from pathlib import Path

import pytest

from document_processing import Pipeline

SAMPLES = Path(__file__).resolve().parent.parent / 'samples'
IMG_EXTS = ('.jpg', '.jpeg', '.png')

REGRESSIONS = [
    ('INTPASSPORT_1997/25_BG_INTPASSPORT_2001', '45 09 400234'),
    ('INTPASSPORT_2011/5_BG_INTPASSPORT_2011', '53 13 314243'),
]


@pytest.fixture(scope='module')
def pipeline():
    return Pipeline(model_format='ONNX', device='cpu', ocr='accurate', verbose=False)


@pytest.mark.parametrize('rel,expected', REGRESSIONS)
def test_series_number_is_read_correctly(pipeline, rel, expected):
    """End-to-end, on the documents the defect was found on."""
    stem = SAMPLES / rel
    image = next((p for ext in IMG_EXTS for p in [stem.with_suffix(ext)] if p.exists()), None)
    if image is None:
        pytest.skip(f'{rel}: sample withdrawn from the public tree on 2026-08-25 '
                    f'(photograph of a real document). This guard runs in the closed '
                    f'repository; it returns here with a synthetic replacement.')
    results = pipeline.process_img(str(image), ocr=True, check_quality=False)
    assert results.ocr.get('Licence_number') == expected
