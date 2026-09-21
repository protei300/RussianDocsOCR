"""Template-based page registration (internal passport).

Covers: the shipped templates load with usable static-print features; a page
under a known homography is recovered to within a few pixels; a non-passport
image registers nothing (so the pipeline keeps the Borders path); and the
default Pipeline(page_registration=True) yields the fixed-size two-page canvas
with the same document type as the plain Borders path (page_registration=False).
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from document_processing.pipeline_modules.page_registration import PageRegistrar
from document_processing.pipeline_modules.page_registration import page_registration as pr

SAMPLE = Path('../samples/INTPASSPORT_2011/6_CR_INTPASSPORT_2011.jpg')
NEGATIVE = Path('../samples/DL_2011/1_CR_DL_2010.jpg')

#: PARTLY OUT OF SERVICE IN THE PUBLIC REPOSITORY since 2026-09-21. The sample
#: under SAMPLE is a single anonymised personal page (the two-page spread that
#: carried this name left the public tree on 2026-08-25 with the rest of the
#: real-document material), and a one-page photo cannot exercise the two-page
#: registration these tests lock down: no page 2, no chain prior, no spread
#: canvas. Kept as skipping tests rather than deleted, so the gap shows up in
#: every run; they return with a synthetic two-page spread. What still runs here:
#: the shipped templates, the negative (non-passport) case and the default/opt-out
#: switch, none of which needs a spread.
SPREAD_WITHDRAWN = pytest.mark.skip(
    reason='needs a two-page passport spread; the public sample under this name is a '
           'single page (the spread was withdrawn on 2026-08-25). Runs in the closed '
           'repository; returns here with a synthetic spread.')


@pytest.fixture(scope='module')
def registrar():
    return PageRegistrar()


@pytest.fixture(scope='module')
def spread_rgb():
    """A frontal internal-passport spread, upright, at pipeline scale."""
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
                # personal zones are erased: the feature mask must not cover the
                # photo box of page 3 nor the stamp/signature area of page 2
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
        # page 2 sits above page 3 in the photo
        assert regs[0].quad[:, 1].mean() < regs[1].quad[:, 1].mean()
        page = registrar.warp_page(spread_rgb, regs[1])
        assert page.shape == (registrar.out_h, registrar.out_w, 3) == (764, 1060, 3)

    @SPREAD_WITHDRAWN
    def test_known_homography_is_recovered(self, registrar, spread_rgb):
        """Warp the photo with a known perspective; the recovered pages must
        land where the untouched photo's pages map to. The template fit is a
        page LOCATOR (the pipeline takes the geometry from the agreeing
        Borders quad): with a few dozen inliers its corners can be off by a
        few percent of the page, so the tolerance is the page centre within
        2.5% of the page width and every corner within 6%."""
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
            assert corner_err < 0.06 * page_w, f'{r.name}: corner off by {corner_err:.1f} px'

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
    @SPREAD_WITHDRAWN
    def test_pipeline_canvas(self):
        from document_processing import Pipeline
        pipe = Pipeline(model_format='ONNX', device='cpu', verbose=False, page_registration=True)
        r = pipe.process_img(str(SAMPLE), ocr=False, find_text_fields=False, check_quality=False)
        assert r.doctype == 'INTPASSPORT_2011'
        meta = r.meta_results['PageRegistration']
        # both pages found by the registrar; geometry from the agreeing Borders
        # quad where there is one, else from the template homography
        assert all(p['ok'] for p in meta['pages'])
        assert all(src in ('borders', 'template') for src in meta['sources'])
        pw, ph = meta['page_size']
        assert 0.25 < meta['scale'] <= 1.0
        assert (pw, ph) == (round(1060 * meta['scale']), round(764 * meta['scale']))
        assert r.img_with_fixed_perspective.shape == (2 * ph, pw, 3)
        # the per-page projection deskew is skipped on registered pages (line_refine did it)
        assert meta['deskewed'] is False   # line_refine straightened the pages; deskew skipped

    def test_pipeline_default_registers_and_opt_out_does_not(self):
        """Registration is the default since 2026-09-17; page_registration=False
        gives the plain Borders canvas with no registration evidence."""
        from document_processing import Pipeline
        pipe = Pipeline(model_format='ONNX', device='cpu', verbose=False)
        r = pipe.process_img(str(SAMPLE), ocr=False, find_text_fields=False, check_quality=False)
        assert 'PageRegistration' in r.meta_results
        plain = Pipeline(model_format='ONNX', device='cpu', verbose=False, page_registration=False)
        r = plain.process_img(str(SAMPLE), ocr=False, find_text_fields=False, check_quality=False)
        assert 'PageRegistration' not in r.meta_results
