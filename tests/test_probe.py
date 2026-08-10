"""Tests for the optional per-stage probe (document_processing/pipeline/probe.py).

Two properties matter and both are asserted here:

1. **Off by default, and free when off.** The probe exists for the
   cross-language conformance harness; it must not change what the library
   computes, and `Pipeline.probe is None` must stay the default. A regression
   here would silently slow every production call.
2. **The sinks behave as documented**, in particular that payloads are passed by
   REFERENCE unless a sink is explicitly asked to copy. Callers that store a
   payload and read it after the next `process_img` would otherwise get whatever
   the pipeline rebound that array to — the same class of bug the service hit
   with `PipelineResults`.

The sink tests need no models, so they are fast and always run. The one test that
constructs a real Pipeline is gated behind ``RUN_PROBE_E2E=1``, matching the
``RUN_QUALITY`` convention in ``test_quality.py``:

    RUN_PROBE_E2E=1 python -m pytest tests/test_probe.py -v          (bash)
    $env:RUN_PROBE_E2E='1'; python -m pytest tests/test_probe.py -v  (powershell)
"""
import json
import os

import numpy as np
import pytest

from document_processing.pipeline.probe import (DirectoryStageSink, NullStageSink,
                                                RecordingStageSink, StageSink)


class _FakeHost:
    """Minimal stand-in for Pipeline, exercising _emit's guard without models."""

    probe = None

    def _emit(self, name, payload):
        if self.probe is not None:
            self.probe.emit(name, payload)


def test_emit_is_a_noop_without_a_probe():
    host = _FakeHost()
    # Must not raise and must not touch the payload.
    host._emit('prepare', object())


def test_emit_reaches_an_attached_sink():
    host = _FakeHost()
    sink = RecordingStageSink()
    host.probe = sink
    host._emit('prepare', 1)
    host._emit('rotate', 2)
    assert sink.names() == ['prepare', 'rotate']
    assert sink.get('rotate') == 2


def test_null_sink_accepts_anything():
    sink = NullStageSink()
    sink.emit('anything', np.zeros((2, 2)))
    assert isinstance(sink, StageSink)


def test_recording_sink_stores_by_reference_by_default():
    sink = RecordingStageSink()
    arr = np.zeros((2, 2), dtype=np.uint8)
    sink.emit('img', arr)
    arr[0, 0] = 7
    # By reference: the mutation IS visible. This is the documented contract, and
    # the reason a sink that outlives a call must copy.
    assert sink.get('img')[0, 0] == 7


def test_recording_sink_copies_when_asked():
    sink = RecordingStageSink(copy=True)
    arr = np.zeros((2, 2), dtype=np.uint8)
    sink.emit('img', arr)
    arr[0, 0] = 7
    assert sink.get('img')[0, 0] == 0


def test_directory_sink_writes_npy_and_json(tmp_path):
    with DirectoryStageSink(tmp_path) as sink:
        sink.emit('prepare', np.arange(6, dtype=np.uint8).reshape(2, 3))
        sink.emit('doctype.label', {'doc_type': 'INTPASSPORT_2011',
                                    'doc_type_confidence': np.float32(0.98)})

    loaded = np.load(tmp_path / 'prepare.npy', allow_pickle=False)
    assert loaded.tolist() == [[0, 1, 2], [3, 4, 5]]

    label = json.loads((tmp_path / 'doctype.label.json').read_text(encoding='utf-8'))
    # numpy scalars must be coerced, or a golden file could not be written at all.
    assert label == {'doc_type': 'INTPASSPORT_2011', 'doc_type_confidence': pytest.approx(0.98, abs=1e-6)}

    index = json.loads((tmp_path / 'stages.json').read_text(encoding='utf-8'))
    assert [e['stage'] for e in index['stages']] == ['prepare', 'doctype.label']
    assert index['stopped_early'] is False


def test_directory_sink_upto_stops_after_the_named_stage(tmp_path):
    with DirectoryStageSink(tmp_path, upto='rotate') as sink:
        sink.emit('prepare', np.zeros((1, 1), dtype=np.uint8))
        sink.emit('rotate', np.zeros((1, 1), dtype=np.uint8))
        sink.emit('fields.bbox', [[1, 2, 3, 4]])

    index = json.loads((tmp_path / 'stages.json').read_text(encoding='utf-8'))
    # 'rotate' is INCLUDED (--upto means "up to and including"), later stages are not.
    assert [e['stage'] for e in index['stages']] == ['prepare', 'rotate']
    assert index['stopped_early'] is True
    assert not (tmp_path / 'fields.bbox.json').exists()


@pytest.mark.skipif(
    os.environ.get('RUN_PROBE_E2E') != '1',
    reason='constructs a real Pipeline and processes a document twice (~25s); '
           'set RUN_PROBE_E2E=1 to run — same convention as RUN_QUALITY in test_quality.py',
)
def test_pipeline_probe_is_off_by_default_and_emits_when_attached():
    """The real thing: a document through a real Pipeline.

    Guards the property that actually matters in production — `probe is None`
    after construction — and confirms the emission sites fire in pipeline order.
    """
    from pathlib import Path

    from document_processing import Pipeline

    sample = Path(__file__).resolve().parents[1] / 'samples' / 'DL_2011' / '1_CR_DL_2010.jpg'
    if not sample.is_file():
        pytest.skip(f'sample missing: {sample}')

    pipeline = Pipeline(model_format='ONNX', device='cpu', verbose=False)
    assert pipeline.probe is None, 'the probe must be off unless a caller attaches one'

    baseline = pipeline.process_img(sample, img_size=1500)
    baseline_ocr = dict(baseline.ocr)

    sink = RecordingStageSink()
    pipeline.probe = sink
    try:
        probed = pipeline.process_img(sample, img_size=1500)
    finally:
        pipeline.probe = None

    # Attaching a probe must not change the result.
    assert dict(probed.ocr) == baseline_ocr

    names = sink.names()
    for expected in ('prepare', 'doctype.label', 'rotate', 'quality',
                     'borders.canvas', 'deskew.canvas', 'fields.bbox', 'join'):
        assert expected in names, f'{expected} was never emitted; got {names}'
    # Order is part of the contract: the stage vocabulary is ordered.
    assert names.index('prepare') < names.index('rotate') < names.index('fields.bbox')
    assert any(n.startswith('ocr.') and n.endswith('.words') for n in names)
