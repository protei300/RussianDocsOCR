"""A requested CUDA session must not silently become a CPU session."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from document_processing.processing import inference


class Session:
    def __init__(self, providers, fallback_at=None):
        self.providers = list(providers)
        self.calls, self.fallback_at = 0, fallback_at

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return [SimpleNamespace(name='x', type='tensor(float)', shape=[1, 2])]

    def run(self, _, inputs):
        self.calls += 1
        if self.calls == self.fallback_at:
            self.providers = ['CPUExecutionProvider']
        return [inputs['x']]


def build(monkeypatch, session, device='gpu'):
    monkeypatch.setattr(inference, 'import_module', lambda _: SimpleNamespace(
        InferenceSession=lambda *a, **kw: session))
    return inference.ModelInference(Path('synthetic.onnx'), device=device)


def test_cuda_missing_at_construction_is_reported(monkeypatch):
    with pytest.raises(RuntimeError, match='CUDAExecutionProvider'):
        build(monkeypatch, Session(['CPUExecutionProvider']))


def test_cuda_disappearing_during_warmup_is_reported(monkeypatch):
    with pytest.raises(RuntimeError, match='CUDAExecutionProvider'):
        build(monkeypatch, Session(['CUDAExecutionProvider', 'CPUExecutionProvider'], fallback_at=1))


def test_cuda_disappearing_during_prediction_is_reported(monkeypatch):
    model = build(monkeypatch, Session(['CUDAExecutionProvider', 'CPUExecutionProvider'], fallback_at=2))
    with pytest.raises(RuntimeError, match='CUDAExecutionProvider'):
        model.predict([np.ones((1, 2), np.float32)])


@pytest.mark.parametrize('device,providers', [('cpu', ['CPUExecutionProvider']),
    ('gpu', ['CUDAExecutionProvider', 'CPUExecutionProvider'])])
def test_valid_cpu_or_cuda_keeps_its_result_and_lock(monkeypatch, device, providers):
    model = build(monkeypatch, Session(providers), device)
    x = np.array([[1., 2.]], np.float32)
    np.testing.assert_array_equal(model.predict([x])[0], x)
    assert (model._run_lock is not None) == (device == 'gpu')
