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
    raw = np.zeros((1, 12, 8), np.float32)
    for i in range(8):
        raw[0, :4, i] = [320, 320, 100, 20]
        raw[0, 4 + (i % 7), i] = .9 if i < 7 else .8
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='images')],
                              run=lambda *_: [raw])
    monkeypatch.setattr(onnxruntime, 'InferenceSession', lambda *a, **kw: session)
    det = FieldsDetector(tmp_path / 'model.onnx')
    out = det.predict(np.zeros((640, 640, 3), np.uint8))
    assert sorted(b['field'] for b in out) == sorted(CLASSES)
    assert len(out) == 7  # same-class duplicate suppressed, different classes kept
    np.testing.assert_allclose(out[0]['quad'].mean(axis=0), [320, 320])
    session.run = lambda *_: [np.zeros((1, 300, 7), np.float32)]
    with pytest.raises(ValueError, match='7-class'):
        det.predict(np.zeros((20, 100, 3), np.uint8))


def test_native_small_input_keeps_pixels_and_restores_coordinates(tmp_path, monkeypatch):
    import onnxruntime
    raw = np.zeros((1,12,1),np.float32)
    raw[0,:4,0]=[285,300,30,10]  # native centre(50,30) + integer padding(235,270)
    raw[0,4,0]=.9
    seen=[]
    session=SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],
        run=lambda _,v:seen.append(v['images']) or [raw])
    monkeypatch.setattr(onnxruntime,'InferenceSession',lambda *a,**k:session)
    det=FieldsDetector(tmp_path/'model.onnx')
    image=np.random.default_rng(7).integers(0,255,(100,170,3),dtype=np.uint8)
    out=det.predict(image)
    np.testing.assert_array_equal(np.rint(seen[0][0,:,270:370,235:405].transpose(1,2,0)*255).astype(np.uint8),image[:,:,::-1])
    np.testing.assert_allclose(out[0]['quad'].mean(0),[50,30])


def test_legacy_five_class_config_still_decodes(tmp_path, monkeypatch):
    import onnxruntime
    (tmp_path/'model.json').write_text(json.dumps({'Inputs':[{'Shape':[640,640,3]}],
        'Outputs':[{'Labels':list(CLASSES[:5])}]}))
    raw=np.zeros((1,10,1),np.float32);raw[0,:4,0]=[320,320,30,10];raw[0,8,0]=.9
    session=SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *_:[raw])
    monkeypatch.setattr(onnxruntime,'InferenceSession',lambda *a,**k:session)
    assert FieldsDetector(tmp_path/'model.onnx').predict(np.zeros((640,640,3),np.uint8))[0]['field']=='flat'


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


def test_detector_reads_confidence_and_overlap_from_model_json(tmp_path, monkeypatch):
    import onnxruntime
    from service.ml.registration.config import RegistrationConfig

    # Two overlapping same-class boxes (probabilistic IoU ~0.59) plus one borderline box at 0.45.
    raw = np.zeros((1, 12, 3), np.float32)
    raw[0, :4, 0] = [300, 300, 100, 20]
    raw[0, 4, 0] = 0.90
    raw[0, :4, 1] = [335, 300, 100, 20]
    raw[0, 4, 1] = 0.80
    raw[0, :4, 2] = [300, 380, 100, 20]
    raw[0, 4, 2] = 0.45
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='images')], run=lambda *_: [raw])
    monkeypatch.setattr(onnxruntime, 'InferenceSession', lambda *a, **kw: session)
    img = np.zeros((640, 640, 3), np.uint8)

    # Without model.json: defaults conf=0.4, overlap=0.7 keep all three boxes.
    det_default = FieldsDetector(tmp_path / 'model.onnx')
    assert det_default.conf == pytest.approx(0.4)
    assert det_default.overlap == pytest.approx(0.7)
    assert len(det_default.predict(img)) == 3

    # With Confidence=0.50 and Overlap=0.40: drops the 0.45 box and suppresses the overlapping duplicate.
    (tmp_path / 'model.json').write_text(json.dumps({
        'Inputs': [{'Shape': [640, 640, 3]}],
        'Outputs': [{'Labels': list(CLASSES), 'Confidence': 0.50, 'Overlap': 0.40}],
    }), encoding='utf-8')
    det_cfg = FieldsDetector(tmp_path / 'model.onnx')
    assert det_cfg.conf == pytest.approx(0.50)
    assert det_cfg.overlap == pytest.approx(0.40)
    assert len(det_cfg.predict(img)) == 1

    # RegistrationConfig leaves fields_conf=None so StampReader preserves model.json thresholds.
    assert RegistrationConfig().fields_conf is None
    det_from_cfg = FieldsDetector(tmp_path / 'model.onnx', conf=RegistrationConfig().fields_conf)
    assert det_from_cfg.conf == pytest.approx(0.50)
    assert det_from_cfg.overlap == pytest.approx(0.40)

    # Explicit overrides take precedence over model.json.
    det_override = FieldsDetector(tmp_path / 'model.onnx', conf=0.40, overlap=0.70)
    assert len(det_override.predict(img)) == 3

    # Legacy CLS / IOU keys are also honoured when Confidence / Overlap are absent.
    (tmp_path / 'model.json').write_text(json.dumps({
        'Inputs': [{'Shape': [640, 640, 3]}],
        'Outputs': [{'Labels': list(CLASSES), 'CLS': 0.48, 'IOU': 0.42}],
    }), encoding='utf-8')
    det_legacy_keys = FieldsDetector(tmp_path / 'model.onnx')
    assert det_legacy_keys.conf == pytest.approx(0.48)
    assert det_legacy_keys.overlap == pytest.approx(0.42)


def test_config_prefers_ocr_v4_when_present_and_falls_back_to_v3(tmp_path, monkeypatch):
    from service.ml.registration.config import RegistrationConfig

    monkeypatch.setenv('RDOCS_MODELS_ROOT', str(tmp_path))
    monkeypatch.delenv('RDOCS_STAMP_OCR', raising=False)
    # Without stamp_v4 files -> falls back to stamp_v3
    assert RegistrationConfig().ocr_dir == tmp_path / 'document_processing/models/OCR/stamp_v3/ONNX'

    # Once stamp_v4 model.onnx + model.json exist -> prefers stamp_v4 automatically
    v4_dir = tmp_path / 'document_processing/models/OCR/stamp_v4/ONNX'
    v4_dir.mkdir(parents=True)
    (v4_dir / 'model.onnx').write_bytes(b'fake')
    (v4_dir / 'model.json').write_text('{}', encoding='utf-8')
    assert RegistrationConfig().ocr_dir == v4_dir


def test_stamp_reader_applies_postprocessor_and_zeroes_probs():
    reader = object.__new__(StampReader)
    q = np.array([[5, 5], [45, 5], [45, 15], [5, 15]])
    reader.det = SimpleNamespace(predict=lambda _: [
        {'field': 'date', 'conf': .9, 'quad': q},
        {'field': 'addr_line', 'conf': .9, 'quad': q + [0, 20]},
    ])
    abc = CTCAlphabet('АБ')
    mats = [np.array([[0, 1, 0]], np.float32), np.array([[0, 0, 1]], np.float32)]
    it = iter(mats)
    reader.ocr = SimpleNamespace(abc=abc, probs=lambda _: next(it))

    seen = {}

    class _StubPP:
        def process_crops(self, fields):
            seen['labels'] = [f['label'] for f in fields]
            return {
                'fields': [
                    {'idx': 0, 'text': '01.02.2020', 'raw': 'А', 'changed': True},
                    {'idx': 1, 'text': 'УЛ. ЛЕНИНА', 'raw': 'Б', 'changed': True},
                ],
                'stamp': {'date': '01.02.2020', 'street': 'ЛЕНИНА'},
            }

    reader.postprocessor = _StubPP()
    img = np.full((60, 60, 3), 200, np.uint8)
    result = reader.read(img)

    assert seen['labels'] == ['date', 'addr_line']
    assert result['fields'] == {'date': '01.02.2020', 'addr_line': 'УЛ. ЛЕНИНА'}
    assert [b['text'] for b in result['boxes']] == ['01.02.2020', 'УЛ. ЛЕНИНА']
    assert result['parsed'] == {'date': '01.02.2020', 'street': 'ЛЕНИНА'}
    # Zero-retention: CTC probability matrices must be wiped after reading
    assert all(m.max() == 0.0 and m.min() == 0.0 for m in mats)


def test_stamp_reader_strips_trailing_g_from_date_fields():
    reader = object.__new__(StampReader)
    q = np.array([[5, 5], [45, 5], [45, 15], [5, 15]])
    reader.det = SimpleNamespace(predict=lambda _: [
        {'field': 'date', 'conf': .9, 'quad': q},
        {'field': 'addr_line', 'conf': .9, 'quad': q + [0, 20]},
    ])
    abc = CTCAlphabet('0123456789 .Г')
    def _one_hot(s):
        rows = [np.eye(len(abc.alphabet) + 1)[0]]
        for ch in s:
            r = np.zeros(len(abc.alphabet) + 1, np.float32)
            r[abc.index[ch]] = 1.0
            rows += [r, np.eye(len(abc.alphabet) + 1)[0]]
        return np.array(rows, np.float32)

    mats = iter([_one_hot('18.09.2018 Г.'), _one_hot('Г.')])
    reader.ocr = SimpleNamespace(abc=abc, probs=lambda _: next(mats))
    reader.postprocessor = None
    img = np.full((60, 60, 3), 200, np.uint8)
    result = reader.read(img)
    assert result['fields']['date'] == '18.09.2018'
    assert result['boxes'][0]['text'] == '18.09.2018'
    assert result['fields']['addr_line'] == 'Г.'  # non-date fields untouched
