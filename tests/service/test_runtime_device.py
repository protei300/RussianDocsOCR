"""Report CPU fallback even when GPU visibility rejects the CUDA attempt."""
import queue
from types import SimpleNamespace

import pytest

import document_processing
from service.ml import runtime


@pytest.mark.parametrize('requested,visible,fail_cuda,expected,fell_back', [
    ('gpu', False, False, 'cpu', True),
    ('gpu', True, True, 'cpu', True),
    ('auto', False, False, 'cpu', False),
    ('cpu', True, False, 'cpu', False),
    ('gpu', True, False, 'gpu', False),
])
def test_requested_and_actual_devices(monkeypatch, requested, visible, fail_cuda, expected, fell_back):
    attempts = []
    def construct(**kw):
        attempts.append(kw['device'])
        if kw['device'] == 'gpu' and fail_cuda:
            raise RuntimeError('CUDAExecutionProvider is not active')
        return SimpleNamespace(device=kw['device'], ocr_device='cpu')
    monkeypatch.setattr(document_processing, 'Pipeline', construct)
    monkeypatch.setattr(runtime, '_POOL', queue.Queue())
    monkeypatch.setattr(runtime, '_INFO', runtime.DeviceInfo())
    monkeypatch.setattr(runtime, 'log_environment', lambda: None)
    monkeypatch.setattr(runtime, 'detect_providers', lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    monkeypatch.setattr(runtime, 'gpu_visible', lambda: visible)
    monkeypatch.setattr(runtime, '_find_warmup_sample', lambda _: None)
    info = runtime.init_runtime(compute_device=requested)
    assert info.state == 'ready'
    assert info.requested_device == requested
    assert info.device == expected
    assert info.ocr_device == 'cpu'  # intentional CPU OCR is not a failure
    assert info.fell_back is fell_back
    assert attempts == (['gpu', 'cpu'] if fail_cuda else [expected])
