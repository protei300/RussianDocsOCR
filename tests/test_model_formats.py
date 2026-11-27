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

    `openvino` is an optional dependency - commented out in requirements.txt,
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
                                          'see requirements.txt)')))
    return formats


@pytest.mark.parametrize('model_format', _formats())
def test_pipeline_runs_with_parallel_stages(model_format):
    """Pipeline must initialize and process a document end-to-end on every format.

    Uses the default low_quality=True, which dispatches the quality checks and
    border detection concurrently - OpenVINO raised "Infer Request is busy"
    there when every thread shared one infer request.
    """
    pipeline = Pipeline(model_format=model_format, device='cpu')

    images = sorted(SAMPLES_DIR.joinpath('DL_2011').glob('*.jpg'))
    if not images:
        pytest.skip('No DL_2011 sample available (construction already verified)')

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


def test_model_inference_not_implemented_and_openvino_contract():
    import threading
    from types import SimpleNamespace
    import numpy as np
    from document_processing.processing.inference import ModelInference

    inf = object.__new__(ModelInference)
    with pytest.raises(NotImplementedError):
        inf.predict([])

    buf = np.array([[1.0, 2.0]], dtype=np.float32)

    class FakeTensor:
        @property
        def data(self):
            return buf

    class FakeReq:
        def infer(self, t):
            pass

        def get_output_tensor(self, i):
            return FakeTensor()

    inf.model = SimpleNamespace(outputs=[0], create_infer_request=lambda: FakeReq())
    inf._ov_local = threading.local()
    out = inf._ModelInference__predict_openvino(np.zeros((1, 2), np.float32))
    assert isinstance(out, list) and len(out) == 1
    buf[0, 0] = 99.0
    assert out[0][0, 0] == 1.0  # copied, not aliased to InferRequest buffer


def test_model_loader_rejects_unknown_type(tmp_path):
    from document_processing.processing.models import ModelLoader

    cfg = tmp_path / 'model.json'
    cfg.write_text(json.dumps({
        'File': 'model.onnx',
        'ModelType': 'UnifiedModel',
        'Inputs': [{'Type': 'NoSuchPreprocess'}],
        'Outputs': [],
    }), encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown input preprocessing type'):
        ModelLoader()(cfg)
