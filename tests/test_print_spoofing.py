"""Print-spoofing detector: a photograph of a printout must not pass as a document.

OUT OF SERVICE IN THE PUBLIC REPOSITORY since 2026-08-25. Both classes were whole
documents - printed and re-photographed for ``fake``, shot directly for ``real`` -
and left with the rest of the real-document material. Re-photographing a printout
cannot be simulated from a clean render, so there was nothing to swap them for; the
suite runs in the closed repository.

The explicit guard below matters more than the skip. This test used to be a bare
``for`` over a glob: with the directory gone it iterated nothing, asserted nothing
and reported success. A suite that goes green when its material disappears is worse
than a red one - red is a question, green is a wrong answer.
"""
import glob
from pathlib import Path

import pytest

from document_processing.pipeline_modules import *

FIXTURES = Path(__file__).resolve().parent / 'images' / 'PrintSpoofing'


def test_print_spoofing():
    files = sorted(FIXTURES.glob('*'))
    if not files:
        pytest.skip('material withdrawn from the public tree on 2026-08-25 (whole '
                    'documents, printed and re-photographed); runs in the closed '
                    'repository, awaits synthetic replacements here')
    print_spoofing = PrintSpoofing('ONNX')
    for image_file_path in files:
        ground_truth = image_file_path.name.split('.')[0].split('_')[0].upper()
        result = print_spoofing.predict(image_file_path)['PrintSpoofing'][0]
        assert ground_truth == result, f'{image_file_path.name}: expected {ground_truth}, got {result}'
