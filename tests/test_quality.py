"""End-to-end quality regression test: full pipeline over samples/ ground truth.

Runs every image in samples/ (115 images, ~2 min on CPU) and asserts that doctype
accuracy, exact-match rate and per-doctype mean CER do not regress below the
baseline measured on 2026-07-22 (see docs/progress-log.md), with a small margin.

Slow by design, so it is skipped unless explicitly requested:

    RUN_QUALITY=1 python -m pytest tests/test_quality.py -v          (bash)
    $env:RUN_QUALITY='1'; python -m pytest tests/test_quality.py -v  (powershell)

Metric definitions live in scripts/eval_quality.py (single source of truth).
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_QUALITY') != '1',
    reason='quality eval is slow (~2 min); set RUN_QUALITY=1 to run',
)


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        'eval_quality', REPO_ROOT / 'scripts' / 'eval_quality.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Thresholds = measured baseline plus a 2-3x safety margin so environment noise
# doesn't flake the test while real regressions still trip it.
# Baseline 2026-07-22, ONNX/cpu/ocr='accurate', image-verified GT:
#   doctype 115/115 (100%), overall exact 1419/1556 (91.2%), mean CER 0.0144
#   per-doctype mean CER: DL_2011 0.0137, DL_2020 0.0107, EXTPASSPORTBIO_2007 0.0105,
#   EXTPASSPORT_2003 0.0086, INTPASSPORT_1997 0.0556, INTPASSPORT_2011 0.0242,
#   SNILS_1996 0.0021
MIN_DOCTYPE_ACCURACY = 0.97
MIN_EXACT_RATE = 0.85
MAX_MEAN_CER = {
    'DL_2011': 0.035,
    'DL_2020': 0.035,
    'EXTPASSPORTBIO_2007': 0.035,
    'EXTPASSPORT_2003': 0.035,
    'INTPASSPORT_1997': 0.09,
    'INTPASSPORT_2011': 0.06,
    'SNILS_1996': 0.02,
}


@pytest.fixture(scope='module')
def eval_results():
    eval_quality = _load_eval_module()
    from document_processing import Pipeline
    pipeline = Pipeline(model_format='ONNX', device='cpu', ocr='accurate',
                        verbose=False)
    overall, by_doctype = eval_quality.eval_samples(
        pipeline, REPO_ROOT / 'samples')
    return overall, by_doctype


def test_no_crashes(eval_results):
    overall, _ = eval_results
    crashes = [f for f in overall.failures if not f[1].startswith('doctype=')]
    assert not crashes, f'pipeline crashed on: {crashes}'


def test_doctype_accuracy(eval_results):
    overall, _ = eval_results
    assert overall.doc_total > 100, 'samples/ ground truth went missing'
    assert overall.doctype_accuracy >= MIN_DOCTYPE_ACCURACY, (
        f'doctype accuracy {overall.doctype_accuracy:.3f} < {MIN_DOCTYPE_ACCURACY}; '
        f'misclassified: {[f for f in overall.failures if f[1].startswith("doctype=")]}')


def test_overall_exact_match(eval_results):
    overall, _ = eval_results
    assert overall.exact_rate >= MIN_EXACT_RATE, (
        f'exact-match rate {overall.exact_rate:.3f} < {MIN_EXACT_RATE} '
        f'(mean CER={overall.mean_cer:.4f})')


def test_per_doctype_cer(eval_results):
    _, by_doctype = eval_results
    assert set(by_doctype) == set(MAX_MEAN_CER), (
        f'doctype folders changed: {sorted(by_doctype)}')
    breaches = {
        doctype: agg.mean_cer
        for doctype, agg in by_doctype.items()
        if agg.mean_cer > MAX_MEAN_CER[doctype]
    }
    assert not breaches, f'mean CER above threshold: {breaches}'
