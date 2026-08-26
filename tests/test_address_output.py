"""Regression tests for the INTPASSPORTADDR output contract.

Two bugs this guards against (both found in the 2026-07 orchestrator audit):

1. ``_ocr`` used to ASSIGN ``meta_results['OCR']`` while ``_address_lines`` MERGES
   into it. ``_ocr`` runs later, so the recognized address was silently dropped as
   soon as ``OCROptionsINTPASSPORTADDR`` gained any field (it has none today,
   which is the only reason the bug was latent).

2. The handwriting placeholder used U+27E8/U+27E9, which cannot be encoded in
   cp1251/cp866 - printing the report on a Russian Windows console raised
   UnicodeEncodeError. Every OCR value must survive the legacy codepages.

Paths are relative to tests/ (see conftest.py, which chdirs there).
"""
from pathlib import Path

import pytest

from document_processing import Pipeline
from document_processing.pipeline.pipeline import PipelineResults

CANVASES = sorted(Path('images/AddressLines').glob('canvas_*.png'))


@pytest.fixture(scope='module')
def pipeline():
    return Pipeline(model_format='ONNX', device='cpu')


@pytest.mark.parametrize('ocr_path', ['_ocr_serial', '_ocr_batched'])
def test_ocr_does_not_clobber_address(pipeline, ocr_path):
    """Whatever _address_lines already wrote must survive the OCR stage."""
    pipeline.results = PipelineResults()
    # Through the private attribute deliberately: this line stands in for what
    # ``_address_lines`` wrote from inside the pipeline, and ``meta_results``
    # now hands out a copy - writing there would land in a discarded dict and
    # this test would fail for a reason unrelated to what it checks.
    pipeline.results._meta_results['OCR'] = {'Address': 'Г. МОСКВА УЛ. ЛЕНИНА Д. 1'}

    getattr(pipeline, ocr_path)({}, 'INTPASSPORTADDR')

    assert pipeline.results.meta_results['OCR'].get('Address') == 'Г. МОСКВА УЛ. ЛЕНИНА Д. 1'


def test_address_output_is_console_encodable(pipeline):
    """OCR output must print on a cp1251/cp866 console, not just UTF-8."""
    if not CANVASES:
        pytest.skip('No AddressLines canvas fixtures found')

    result = pipeline.process_img(CANVASES[0])
    assert result.ocr, 'Expected address OCR output'

    for key, value in result.ocr.items():
        if not isinstance(value, str):
            continue
        for encoding in ('cp1251', 'cp866'):
            try:
                value.encode(encoding)
            except UnicodeEncodeError as e:
                pytest.fail(f'OCR[{key!r}] is not {encoding}-encodable: {e}')
