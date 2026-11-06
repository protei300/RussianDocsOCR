"""Local transcription export must keep suggestions separate and preserve edits."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.mark.parametrize('with_ocr', [False, True])
def test_export_is_local_and_blind_by_default(tmp_path, monkeypatch, with_ocr):
    path = Path(__file__).resolve().parents[2] / 'scripts/crop_stamp_fields.py'
    spec = importlib.util.spec_from_file_location('crop_stamp_fields', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'source'
    source.mkdir()
    image = np.full((60, 100, 3), 190, np.uint8)
    (source / 'stamp.png').write_bytes(cv2.imencode('.png', image)[1].tobytes())
    weight = tmp_path / 'model.onnx'
    weight.write_bytes(b'synthetic checkpoint')
    box = {'field': 'addr_line', 'conf': .9,
           'quad': np.array([[5, 5], [90, 5], [90, 20], [5, 20]], np.float64)}
    monkeypatch.setattr(module, 'FieldsDetector', lambda *a, **kw: SimpleNamespace(predict=lambda _: [box]))
    calls = []
    def make_ocr(*a, **kw):
        calls.append(1)
        return SimpleNamespace(probs=lambda _: np.array([[0, 1]], np.float32),
                               abc=SimpleNamespace(alphabet='А'))
    monkeypatch.setattr(module, 'StampOCR', make_ocr)
    cfg = SimpleNamespace(fields_model=weight, ocr_dir=tmp_path, fields_conf=.4)
    destination = tmp_path / 'export'
    assert module.collect(source, destination, cfg, with_ocr=with_ocr) == 1
    result = json.loads((destination / 'annotations.json').read_text('utf-8'))
    row = result['fields'][0]
    assert row['stamp_id'] == result['stamps'][0]['stamp_id']
    assert row['source_sha256'] == result['stamps'][0]['source_sha256']
    assert row['id'].startswith(row['stamp_id'] + '_')
    assert row['text'] is None and not row['reviewed'] and not row['unreadable']
    assert ('ocr_suggestion' in row) == with_ocr
    assert bool(calls) == with_ocr
    assert cv2.imdecode(np.frombuffer((destination / row['crop']).read_bytes(), np.uint8), 1) is not None
    assert not (destination / 'stamp.png').exists()
    (destination / 'annotations.json').write_text('user edits', encoding='utf-8')
    with pytest.raises(FileExistsError):
        module.collect(source, destination, cfg)
    assert (destination / 'annotations.json').read_text('utf-8') == 'user edits'
