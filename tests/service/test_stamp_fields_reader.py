"""Five-class field decoding, raw OCR and numeric crop orientation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from service.ml.registration.fields_net import CLASSES, FieldsDetector
from service.ml.registration.ocr import CTCAlphabet, crop
from service.ml.registration.stamp import StampReader


def test_raw_obb_output_maps_all_classes_and_keeps_overlapping_different_fields(tmp_path, monkeypatch):
    import onnxruntime
    raw = np.zeros((1, 10, 6), np.float32)
    for i in range(6):
        raw[0, :4, i] = [512, 512, 100, 20]
        raw[0, 4 + (i % 5), i] = .9 if i < 5 else .8
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='images')],
                              run=lambda *_: [raw])
    monkeypatch.setattr(onnxruntime, 'InferenceSession', lambda *a, **kw: session)
    det = FieldsDetector(tmp_path / 'model.onnx')
    out = det.predict(np.zeros((1024, 1024, 3), np.uint8))
    assert sorted(b['field'] for b in out) == sorted(CLASSES)
    assert len(out) == 5  # same-class duplicate suppressed, different classes kept
    np.testing.assert_allclose(out[0]['quad'].mean(axis=0), [512, 512])
    session.run = lambda *_: [np.zeros((1, 300, 7), np.float32)]
    with pytest.raises(ValueError, match='five-class'):
        det.predict(np.zeros((20, 100, 3), np.uint8))


def test_wrong_model_class_order_is_rejected_before_loading(tmp_path):
    (tmp_path / 'model.json').write_text(json.dumps({'Outputs': [{'Labels': ['flat', 'date']}]}))
    with pytest.raises(ValueError, match='classes'):
        FieldsDetector(tmp_path / 'model.onnx')


def test_numeric_crop_does_not_turn_a_narrow_upright_digit_sideways():
    img = np.full((80, 80, 3), 200, np.uint8)
    q = np.array([[30, 20], [40, 20], [40, 50], [30, 50]])
    for shift in range(4):
        got = crop(img, np.roll(q, shift, axis=0), 0, 0, horizontal=True)
        assert got.shape[:2] == (30, 10)


def test_raw_address_lines_are_not_relabelled_as_streets():
    reader = object.__new__(StampReader)
    q = np.array([[5, 5], [45, 5], [45, 15], [5, 15]])
    reader.det = SimpleNamespace(predict=lambda _: [
        {'field': 'addr_line', 'conf': .9, 'quad': q},
        {'field': 'addr_line', 'conf': .9, 'quad': q + [0, 20]}])
    abc = CTCAlphabet('АБ')
    readings = iter([np.array([[0, 1, 0]], np.float32), np.array([[0, 0, 1]], np.float32)])
    reader.ocr = SimpleNamespace(abc=abc, probs=lambda _: next(readings))
    img = np.full((60, 60, 3), 200, np.uint8)
    result = reader.read(img)
    assert result['fields'] == {'addr_line': 'А\nБ'}
    assert [b['text'] for b in result['boxes']] == ['А', 'Б']
    assert img.min() == 200


def test_models_root_and_overrides_are_respected(tmp_path, monkeypatch):
    from service.ml.registration.config import RegistrationConfig
    monkeypatch.setenv('RDOCS_MODELS_ROOT', str(tmp_path))
    monkeypatch.delenv('RDOCS_STAMP_FIELDS_MODEL', raising=False)
    monkeypatch.delenv('RDOCS_STAMP_OCR', raising=False)
    cfg = RegistrationConfig()
    assert cfg.fields_model == tmp_path / 'document_processing/models/StampFields/ONNX/model.onnx'
    assert cfg.ocr_dir == tmp_path / 'document_processing/models/OCR/stamp_v3/ONNX'
    assert set(cfg.missing()) == {'RDOCS_STAMP_FIELDS_MODEL', 'RDOCS_STAMP_OCR'}
    monkeypatch.setenv('RDOCS_STAMP_FIELDS_MODEL', str(tmp_path / 'custom.onnx'))
    assert RegistrationConfig().fields_model == tmp_path / 'custom.onnx'
