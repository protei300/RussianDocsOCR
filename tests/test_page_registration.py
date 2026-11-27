"""Template-based page registration (internal passport)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from document_processing.pipeline_modules.page_registration import PageRegistrar
from document_processing.pipeline_modules.page_registration import page_registration as pr

SAMPLE = Path('../samples/INTPASSPORT_2011/6_CR_INTPASSPORT_2011.jpg')
NEGATIVE = Path('../samples/DL_2011/1_CR_DL_2010.jpg')

SPREAD_WITHDRAWN = pytest.mark.skip(
    reason='needs a two-page passport spread; the public sample under this name is a '
           'single page (the spread was withdrawn on 2026-08-25). Runs in the closed '
           'repository; returns here with a synthetic spread.')


@pytest.fixture(scope='module')
def registrar():
    return PageRegistrar()


@pytest.fixture(scope='module')
def spread_rgb():
    bgr = cv2.imread(str(SAMPLE))
    assert bgr is not None, SAMPLE
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = 1500 / max(h, w)
    return cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


class TestTemplates:
    def test_templates_load_with_features(self, registrar):
        assert registrar.page_names == ['page2', 'page3']
        assert (registrar.page_w, registrar.page_h) == (1000, 704)
        for name, refs in registrar.pages:
            assert len(refs) >= 1, name
            for tpl in refs:
                assert (tpl.width, tpl.height) == (1000, 704)
                assert len(tpl.kp) >= 300, f'{name}: too few static-print features'
                if name == 'page3':
                    assert tpl.mask[200:560, 60:320].sum() == 0
                else:
                    assert tpl.mask[400:600, 20:340].sum() == 0


class TestRegister:
    @SPREAD_WITHDRAWN
    def test_frontal_spread_both_pages(self, registrar, spread_rgb):
        regs = registrar.register(spread_rgb)
        assert [r.ok for r in regs] == [True, True]
        assert regs[0].inliers >= pr.MIN_REFINED_INLIERS
        assert regs[0].quad[:, 1].mean() < regs[1].quad[:, 1].mean()
        page = registrar.warp_page(spread_rgb, regs[1])
        assert page.shape == (registrar.out_h, registrar.out_w, 3) == (764, 1060, 3)

    @SPREAD_WITHDRAWN
    def test_known_homography_is_recovered(self, registrar, spread_rgb):
        base = registrar.register(spread_rgb)
        assert all(r.ok for r in base)
        h, w = spread_rgb.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[60, 40], [w - 30, 80], [w - 90, h - 50], [20, h - 20]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(spread_rgb, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        regs = registrar.register(warped)
        assert all(r.ok for r in regs)
        for b, r in zip(base, regs):
            expected = cv2.perspectiveTransform(b.quad.reshape(1, -1, 2), M).reshape(4, 2)
            page_w = np.linalg.norm(expected[1] - expected[0])
            centre_err = np.linalg.norm(expected.mean(axis=0) - r.quad.mean(axis=0))
            corner_err = np.linalg.norm(expected - r.quad, axis=1).max()
            assert centre_err < 0.025 * page_w, f'{r.name}: centre off by {centre_err:.1f} px'
            assert corner_err < 0.08 * page_w, f'{r.name}: corner off by {corner_err:.1f} px'

    @SPREAD_WITHDRAWN
    def test_borders_quads_are_used(self, registrar, spread_rgb):
        base = registrar.register(spread_rgb)
        quads = [r.quad for r in base]
        regs = registrar.register(spread_rgb, quads)
        assert all(r.ok for r in regs)
        assert all(r.method.startswith('quad') for r in regs)

    def test_non_passport_registers_nothing(self, registrar):
        bgr = cv2.imread(str(NEGATIVE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        regs = registrar.register(rgb)
        assert not any(r.ok for r in regs)


class TestPipeline:
    @pytest.fixture(scope='class')
    def pipe(self):
        from document_processing import Pipeline
        return Pipeline(model_format='ONNX', device='cpu', verbose=False)

    @SPREAD_WITHDRAWN
    def test_pipeline_canvas(self, pipe):
        assert pipe.page_registrar is not None, 'page_registration must be enabled by default'
        r = pipe.process_img(str(SAMPLE), ocr=False, find_text_fields=False, check_quality=False)
        assert r.doctype == 'INTPASSPORT_2011'
        meta = r.meta_results['PageRegistration']
        assert all(p['ok'] for p in meta['pages'])
        assert all(src in ('borders', 'template') for src in meta['sources'])
        pw, ph = meta['page_size']
        assert 0.25 < meta['scale'] <= 1.0
        assert (pw, ph) == (round(1060 * meta['scale']), round(764 * meta['scale']))
        assert r.img_with_fixed_perspective.shape == (2 * ph, pw, 3)
        assert meta['deskewed'] is False

    def test_pipeline_default_registers_and_opt_out_does_not(self, pipe, monkeypatch):
        """Registration is the default since 2026-09-17; page_registration=False
        gives the plain Borders canvas with no registration evidence."""
        import inspect
        from document_processing import Pipeline

        assert inspect.signature(Pipeline.__init__).parameters['page_registration'].default is True
        r = pipe.process_img(str(SAMPLE), ocr=False, find_text_fields=False, check_quality=False)
        assert 'PageRegistration' in r.meta_results
        monkeypatch.setattr(pipe, 'page_registrar', None)
        r = pipe.process_img(str(SAMPLE), ocr=False, find_text_fields=False, check_quality=False)
        assert 'PageRegistration' not in r.meta_results


def test_page_registration_handles_zero_sift_keypoints_and_singular_H():
    from document_processing.pipeline_modules.page_registration.page_registration import PageTemplate

    reg = object.__new__(PageRegistrar)
    quad = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
    kp, desc = reg._features_in_quad([], None, quad, (200, 200))
    assert kp == [] and desc is None

    tpl = object.__new__(PageTemplate)
    tpl.corners = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert tpl.quad_in_image(np.zeros((3, 3), dtype=np.float64)) is None
