"""Regression tests for Wave 1 library reliability, packaging, and edge-case fixes."""
from __future__ import annotations

import hashlib
import http.client
import json
import threading
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import document_processing
from document_processing import config as dp_config
from document_processing import fetch_models
from document_processing.geometry import Homography, Pieces
from document_processing.pipeline.documents import DocumentsStage
from document_processing.pipeline.options import OCROptionsClass, OCROptionsMigrationCard
from document_processing.pipeline.pipeline import Pipeline
from document_processing.pipeline.results import PipelineResults
from document_processing.pipeline_modules.deskewer.deskewer import DocDeskewer
from document_processing.pipeline_modules.glare_detector.glare import Glare
from document_processing.pipeline_modules.ocr_batch import _WidthToTime
from document_processing.pipeline_modules.page_registration import line_refine
from document_processing.pipeline_modules.page_registration.page_registration import (
    PageRegistrar,
    PageTemplate,
)
from document_processing.pipeline_modules.page_registration.quad_fit import _fit_line
from document_processing.pipeline_modules.textfields_detector.textfields_detector import (
    TextFieldsDetector,
)
from document_processing.processing.inference import ModelInference
from document_processing.processing.models import ModelLoader
from document_processing.processing.preprocessing import BasePreprocessing, YoloPreprocessing


def test_package_dunder_all_exports_valid_attributes():
    assert set(document_processing.__all__) == {'Pipeline', '__version__'}
    for name in document_processing.__all__:
        assert hasattr(document_processing, name)


def test_resolve_model_dir_honors_rdocs_models_root(tmp_path, monkeypatch):
    monkeypatch.setenv('RDOCS_MODELS_ROOT', str(tmp_path))
    resolved = dp_config.resolve_model_dir('Borders' if 'Borders' in dp_config._RAW_CFG else 'DocDetector')
    assert resolved == tmp_path / 'document_processing' / 'models' / 'Borders'

    monkeypatch.setenv('RDOCS_MODELS_ROOT', '')
    default_resolved = dp_config.resolve_model_dir('DocDetector')
    assert default_resolved == Path(dp_config.DEFAULT_CFG['DocDetector'])


def test_lsd_is_thread_local():
    ids = []

    def worker():
        ids.append(id(line_refine._lsd()))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(ids) == 2 and ids[0] != ids[1]


def test_geometry_handles_singular_homography_and_empty_pieces():
    singular = np.zeros((3, 3), dtype=np.float64)
    assert Homography(singular).to_input([[10, 20]]) is None

    # Perspective matrix whose inverse has w == 0 at x_shifted = -1 -> (-0.5, 0.5)
    vanishing = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    assert Homography(vanishing).to_input([[-0.5, 0.5]]) is None

    assert Pieces(()).to_input([[1.0, 2.0]]) is None
    empty_pts = Pieces((((0, 0, 10, 10), Homography(np.eye(3))),)).to_input(np.zeros((0, 2)))
    assert empty_pts is not None and empty_pts.shape == (0, 2)


def test_page_registration_handles_zero_sift_keypoints_and_singular_H():
    reg = object.__new__(PageRegistrar)
    quad = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
    kp, desc = reg._features_in_quad([], None, quad, (200, 200))
    assert kp == [] and desc is None

    tpl = object.__new__(PageTemplate)
    tpl.corners = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert tpl.quad_in_image(np.zeros((3, 3), dtype=np.float64)) is None


def test_fit_line_returns_none_when_second_refit_loses_inliers(monkeypatch):
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    calls = {'n': 0}

    def fake_fit_line(*a, **kw):
        calls['n'] += 1
        if calls['n'] == 1:
            return np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
        return np.array([[1.0], [0.0], [0.0], [100.0]], dtype=np.float32)

    monkeypatch.setattr('cv2.fitLine', fake_fit_line)
    assert _fit_line(pts, thr=1.0, rng=np.random.default_rng(0)) is None


def test_glare_predict_transform_matches_predict_direction(monkeypatch):
    class FakeModel:
        def __init__(self, label, conf):
            self._res = [(label, conf)]

        def predict(self, img):
            return self._res

    g_clean = object.__new__(Glare)
    g_clean.model_name = 'Glare'
    g_clean.model = FakeModel('NO', 0.99)
    res_clean = g_clean.predict_transform(np.full((64, 64, 3), 200, np.uint8))
    assert res_clean['Glare'][0] == 'good' and res_clean['Glare'][1] == 0.0
    assert 'warped_img' not in res_clean

    g_bad = object.__new__(Glare)
    g_bad.model_name = 'Glare'
    g_bad.model = FakeModel('GLARE', 0.95)
    res_bad = g_bad.predict_transform(np.full((64, 64, 3), 200, np.uint8))
    assert res_bad['Glare'][0] == 'bad' and res_bad['Glare'][1] == 1.0
    assert 'warped_img' in res_bad


def test_yolo_letterbox_handles_extreme_aspect_and_rejects_empty():
    pre = YoloPreprocessing(image_size=[640, 640, 3], padding_size=[0, 0], padding_color=[114, 114, 114])
    strip = np.zeros((1, 2000, 3), dtype=np.uint8)
    tensor, *_ = pre(strip)
    assert tensor.shape == (1, 640, 640, 3)

    with pytest.raises(ValueError, match='Empty image'):
        pre(np.zeros((0, 10, 3), dtype=np.uint8))


def test_textfields_predict_transform_clamps_boxes():
    det = object.__new__(TextFieldsDetector)
    det.model_name = 'TextFieldsDetector'
    det.model = SimpleNamespace(predict=lambda img: [[-5, -2, 120, 60, 0.9, 'Last_name_ru']])
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    out = det.predict_transform(img)['TextFieldsDetector']
    assert out['warped_img'][0].shape == (40, 80, 3)


def test_pipeline_results_incomplete_accessors_do_not_raise_keyerror():
    r = PipelineResults()
    assert r.rotated_image is None
    assert r.angle is None
    assert r.img_with_fixed_perspective is None


def test_migration_card_make_options_accepts_full_doctype():
    assert isinstance(OCROptionsClass.make_options('MIGRATIONCARD'), OCROptionsMigrationCard)
    assert isinstance(OCROptionsClass.make_options('MIGRATIONCARD_ALL'), OCROptionsMigrationCard)


def test_documents_stage_resets_scratch_and_handles_degenerate_crop():
    pipe = object.__new__(Pipeline)
    pipe._date_lines = {'Birth_date': [np.zeros((5, 5, 3), np.uint8)]}
    pipe._field_lines = {'Last_name_ru': [np.zeros((5, 5, 3), np.uint8)]}
    pipe._address_line_obboxes = [[0, 0, 1, 1, 0]]
    pipe._doc_year = 2011
    pipe._start_document()
    assert pipe._date_lines == {}
    assert pipe._field_lines == {}
    assert pipe._address_line_obboxes == []
    assert pipe._doc_year is None

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert DocumentsStage._document_crop(pipe, frame, {'box': [150, 150, 160, 160]}) is None
    assert DocumentsStage._document_crop(pipe, frame, {'box': [50, 50, 40, 40]}) is None


def test_prepare_image_clamps_extreme_aspect_and_rejects_bad_img_size():
    pipe = object.__new__(Pipeline)
    pipe._start_document()
    strip = np.zeros((1, 5000, 3), dtype=np.uint8)
    out = pipe._prepare_image(strip, img_size=1500)
    assert out.shape[0] == 1 and out.shape[1] in (1499, 1500) and out.shape[2] == 3

    with pytest.raises(ValueError, match='img_size'):
        pipe._prepare_image(strip, img_size=0)


def test_model_inference_not_implemented_and_openvino_contract():
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
    cfg = tmp_path / 'model.json'
    cfg.write_text(json.dumps({
        'File': 'model.onnx',
        'ModelType': 'UnifiedModel',
        'Inputs': [{'Type': 'NoSuchPreprocess'}],
        'Outputs': [],
    }), encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown input preprocessing type'):
        ModelLoader()(cfg)


def test_width_to_time_handles_non_power_of_two_stride_and_threads():
    class FakeOCR:
        class inference_model:
            @staticmethod
            def predict(xs):
                w = xs[0].shape[2]
                return [np.zeros((1, (w + 2) // 3, 10), dtype=np.float32)]

    calib = _WidthToTime(FakeOCR())
    results = []
    errors = []

    def worker():
        try:
            for w in range(10, 80):
                results.append((w, calib.t_of(w)))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for w, t in results:
        assert t == (w + 2) // 3


def test_deskewer_zero_range_and_single_row_image():
    d = DocDeskewer(angle_range=0.0, angle_steps=1)
    assert d._fine_count == 1
    row = np.zeros((1, 50, 3), dtype=np.uint8)
    out, geom = d.deskew_with_geometry(row, n_segments=2)
    assert out is row and geom is None


def test_base_preprocessing_raises_on_corrupt_image_file(tmp_path):
    bad = tmp_path / 'corrupt.jpg'
    bad.write_bytes(b'not a jpeg')
    pre = BasePreprocessing(image_size=[32, 32, 3], padding_size=[0, 0], padding_color=[0, 0, 0])
    with pytest.raises(ValueError, match='Could not decode image'):
        pre(bad)


def test_fetch_models_safe_dest_and_http_exception_retry(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='escapes'):
        fetch_models._safe_dest(tmp_path, '../outside.onnx')
    with pytest.raises(ValueError, match='unsafe'):
        fetch_models._safe_dest(tmp_path, str(tmp_path / 'abs.onnx'))
    with pytest.raises(ValueError, match='unsupported URL scheme'):
        fetch_models.download('ftp://example.com/a.onnx', tmp_path / 'a.onnx', '0' * 64)

    payload = b'hello-model'
    expected_sha = hashlib.sha256(payload).hexdigest()
    attempts = {'n': 0}

    class FakeResp:
        def __init__(self, fail_mid: bool):
            self._fail = fail_mid
            self._done = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n=-1):
            if self._fail:
                raise http.client.IncompleteRead(b'hel', len(payload) - 3)
            if self._done:
                return b''
            self._done = True
            return payload

    def fake_urlopen(req, timeout=60):
        attempts['n'] += 1
        return FakeResp(fail_mid=(attempts['n'] == 1))

    monkeypatch.setattr(fetch_models.urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(fetch_models.time, 'sleep', lambda _: None)
    dest = tmp_path / 'ok.onnx'
    fetch_models.download('https://example.com/ok.onnx', dest, expected_sha, expected_size=len(payload))
    assert attempts['n'] == 2
    assert dest.read_bytes() == payload

    # Permanent 404 must not retry 3 times.
    calls_404 = {'n': 0}

    def fake_404(req, timeout=60):
        calls_404['n'] += 1
        raise urllib.error.HTTPError(req.full_url, 404, 'Not Found', {}, None)

    monkeypatch.setattr(fetch_models.urllib.request, 'urlopen', fake_404)
    with pytest.raises(RuntimeError):
        fetch_models.download('https://example.com/missing.onnx', tmp_path / 'missing.onnx', expected_sha)
    assert calls_404['n'] == 1


def test_wave2_blur_glare_batches_28_tiles_into_single_inference_call():
    from document_processing.pipeline_modules.blur_detector.quality import (
        QualityChecker as BlurChecker,
    )
    from document_processing.pipeline_modules.glare_detector.quality import (
        QualityChecker as GlareChecker,
    )

    calls = []

    class BatchedModel:
        preprocessings = [SimpleNamespace(image_size=(128, 128, 3), padding_size=(0, 0))]

        class inference_model:
            @staticmethod
            def predict(xs):
                calls.append(xs[0].shape)
                return [np.tile(np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], np.float32), (xs[0].shape[0], 1))]

        postprocessings = [lambda row: ('NonBlur', 1.0)]

    img = np.full((256, 448, 3), 200, np.uint8)
    bc = BlurChecker(BatchedModel(), (7, 4))
    assert bc.check_image_quality(img) == 1.0
    assert calls == [(28, 128, 128, 3)]
    bc.annotate_image(img, reuse=True)
    assert len(calls) == 1  # reused without re-running inference

    calls.clear()

    class BatchedGlare(BatchedModel):
        postprocessings = [lambda row: ('NO', 0.99)]

    gc = GlareChecker(BatchedGlare(), (7, 4))
    assert gc.check_image_quality(img) == 0.0
    assert calls == [(28, 128, 128, 3)]
    gc.annotate_image(img, reuse=True)
    assert len(calls) == 1


def test_wave2_yolo_segmentor_emits_uint8_masks():
    from document_processing.processing.postprocessing import YOLOSegmentorPostprocessing

    post = YOLOSegmentorPostprocessing(mask_filter=0.5)
    proto = np.ones((16, 16, 4), dtype=np.float32)
    masks = np.ones((1, 4), dtype=np.float32)
    bbox = np.array([[2, 2, 14, 14]], dtype=np.float32)
    out_masks, segs = post(proto, masks, bbox, (0, 0), [16, 16])
    assert out_masks.dtype == np.uint8
    assert len(segs) == 1 and len(segs[0]) >= 4


def test_wave2_page_geometry_skips_fit_quad_lines_when_disabled(monkeypatch):
    from document_processing.pipeline_modules.page_registration import geometry as pg_mod

    def fail_if_called(*a, **kw):
        raise AssertionError('fit_quad_lines must not run when line_quads=False')

    monkeypatch.setattr(pg_mod, 'fit_quad_lines', fail_if_called)
    pg = pg_mod.PageGeometry(line_quads=False)
    square = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32)
    quads, infos = pg.quads([square], (200, 200))
    assert len(quads) == 1 and infos[0]['method'] == 'polygon'


def test_wave2_refine_by_lines_skips_fit_when_already_straight(monkeypatch):
    def fail_fit(*a, **kw):
        raise AssertionError('_fit must not run when page is already straight')

    monkeypatch.setattr(line_refine, '_fit', fail_fit)
    fake_meas = np.array([
        [100.0, 100.0, 300.0, 100.0, 0.0, 3.0, 0.0],
        [100.0, 200.0, 300.0, 200.0, 0.0, 3.0, 0.0],
    ], dtype=np.float64)
    monkeypatch.setattr(line_refine, 'measure', lambda gray, inset=0: fake_meas)
    Hm, info = line_refine.refine_by_lines(np.full((400, 400), 255, np.uint8))
    assert Hm is None and info['reason'] == 'already straight'
