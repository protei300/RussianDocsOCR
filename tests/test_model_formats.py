"""Regression tests: every shipped model format must actually load.

Guards against config/artifact drift between formats - e.g. a model.json
pointing at a centers file the postprocessing cannot read (the OpenVINO
DocTypeAngles referenced a pickled centers.pkl while MetricPostprocessing only
accepts a pickle-free .npz), or a runtime that breaks once the pipeline runs
its stages concurrently ("Infer Request is busy" on OpenVINO).

Paths are relative to tests/ (see conftest.py, which chdirs there).
"""
import importlib.util
import json
from pathlib import Path

import pytest

from document_processing import Pipeline

MODELS_DIR = Path('../document_processing/models')
SAMPLES_DIR = Path('../samples')


def _formats():
    """ONNX always; OpenVINO only where the runtime is installed.

    `openvino` is an optional dependency - commented out in requirements2026.txt,
    since the ONNX runtime is the default and OpenVINO only pays off on Intel
    CPUs. Parametrising over it unconditionally made these two cases fail with
    ModuleNotFoundError on any machine that took the file at its word, CI
    included. A skip states the requirement; a failure just looks broken.
    """
    formats = ['ONNX']
    if importlib.util.find_spec('openvino') is not None:
        formats.append('OpenVINO')
    else:
        formats.append(pytest.param(
            'OpenVINO',
            marks=pytest.mark.skip(reason='openvino not installed (optional; '
                                          'see requirements2026.txt)')))
    return formats


@pytest.mark.parametrize('model_format', _formats())
def test_pipeline_constructs(model_format):
    """Pipeline must initialize for every supported model_format."""
    Pipeline(model_format=model_format, device='cpu')


@pytest.mark.parametrize('model_format', _formats())
def test_pipeline_runs_with_parallel_stages(model_format):
    """A document must process end-to-end on every format.

    Uses the default low_quality=True, which dispatches the quality checks and
    border detection concurrently - OpenVINO raised "Infer Request is busy"
    there when every thread shared one infer request.
    """
    images = sorted(SAMPLES_DIR.joinpath('DL_2011').glob('*.jpg'))
    if not images:
        pytest.skip('No DL_2011 sample available')

    pipeline = Pipeline(model_format=model_format, device='cpu')
    result = pipeline.process_img(images[0], ocr=True, check_quality=True,
                                  low_quality=True, img_size=1500)

    assert result.doctype == 'DL_2011'
    assert result.ocr, 'Expected OCR fields'


def test_metric_centers_are_pickle_free():
    """Every Metric model.json must point at a loadable pickle-free .npz."""
    import numpy as np

    checked = 0
    for cfg_path in MODELS_DIR.rglob('model.json'):
        cfg = json.loads(cfg_path.read_text(encoding='utf8'))
        for output in cfg.get('Outputs', []):
            if output.get('Type') != 'Metric':
                continue
            centers = cfg_path.parent / output['Centers'].replace('\\', '/')
            assert centers.exists(), f'{cfg_path}: missing centers file {centers}'
            assert centers.suffix == '.npz', (
                f'{cfg_path}: centers must be a pickle-free .npz, got {centers.name}'
            )
            data = np.load(centers, allow_pickle=False)
            for key in ('labels', 'centers', 'max_distance'):
                assert key in data, f'{centers}: missing array {key!r}'
            checked += 1

    assert checked, 'No Metric outputs found - test would silently pass'
