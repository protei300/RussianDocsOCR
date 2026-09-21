"""Page geometry helpers of page_registration: quad from line fits to a
segmentation contour (quad_fit) and straightening a rectified page by its own
lines (line_refine). Synthetic, deterministic, no models."""
import cv2
import numpy as np
import pytest

from document_processing.pipeline_modules.page_registration import quad_fit, line_refine, line_dewarp
from document_processing.pipeline_modules.doc_detector.image_transformation import extract_quad, order_points


def _contour_of(mask):
    cs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return max(cs, key=len).reshape(-1, 2).astype(np.float32)


def _page_mask(shape, quad):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(quad))], 255)
    return m


class TestQuadFit:
    QUAD = np.float32([[300, 200], [1200, 230], [1180, 850], [280, 820]])   # a tilted page

    def test_clean_page_matches_polygon_corners(self):
        cnt = _contour_of(_page_mask((1100, 1500), self.QUAD))
        q, info = quad_fit.fit_quad_lines(cnt, (1100, 1500), 704 / 1000)
        assert info['method'] == 'lines'
        assert np.abs(q - order_points(self.QUAD)).max() < 3.0

    def test_thumb_on_the_edge_is_ignored(self):
        mask = _page_mask((1100, 1500), self.QUAD)
        # a thumb holding the bottom-left corner, merged into the mask (as in
        # the frames where the polygon corner lands on the thumb)
        cv2.ellipse(mask, (250, 800), (120, 95), 20, 0, 360, 255, -1)
        cnt = _contour_of(mask)
        poly = order_points(extract_quad(cnt))
        q, info = quad_fit.fit_quad_lines(cnt, (1100, 1500), 704 / 1000)
        assert info['method'] == 'lines'
        err_lines = np.abs(q - order_points(self.QUAD)).max()
        err_poly = np.abs(poly - order_points(self.QUAD)).max()
        assert err_lines < 4.0, err_lines
        assert err_lines < err_poly          # the polygon corner did move onto the thumb

    def test_side_clipped_by_frame_is_extrapolated(self):
        # page whose bottom runs out of the picture: the mask ends at the frame
        quad = np.float32([[300, 400], [1200, 420], [1200, 1150], [300, 1130]])   # h ~ 730 > frame
        mask = _page_mask((900, 1500), quad)
        cnt = _contour_of(mask)
        q, info = quad_fit.fit_quad_lines(cnt, (900, 1500), 730 / 900)
        assert info['method'] == 'lines'
        assert info['clipped'] == [2] and info['extrapolated'] == 2
        # bottom corners put back where the aspect ratio says, i.e. below the frame
        assert q[2, 1] > 1100 and q[3, 1] > 1100
        assert np.abs(q[:2] - order_points(quad)[:2]).max() < 3.0

    def test_partly_clipped_side_is_fitted_from_its_visible_part(self):
        # a rotated page whose bottom-right corner leaves the frame: the bottom
        # side is partly the frame border, partly the true edge
        quad = np.float32([[200, 100], [1300, 250], [1200, 950], [100, 800]])
        mask = _page_mask((900, 1500), quad)
        cnt = _contour_of(mask)
        q, info = quad_fit.fit_quad_lines(cnt, (900, 1500), 704 / 1000)
        assert info['method'] == 'lines'
        true = order_points(quad)
        # the three fully visible corners are exact; the clipped one is
        # extended along the true edges, so it lands near the true corner
        assert np.abs(q[[0, 1, 3]] - true[[0, 1, 3]]).max() < 3.0
        assert np.linalg.norm(q[2] - true[2]) < 25.0

    def test_degenerate_contour(self):
        assert quad_fit.fit_quad_lines(np.zeros((3, 2), np.float32))[0] is None


def _ruled_page(w=1060, h=764):
    """White page with dark horizontal rules and text-like dashes, plus two
    vertical rules - the structures a real page offers."""
    img = np.full((h, w), 235, np.uint8)
    for y in range(80, h - 60, 44):
        cv2.line(img, (70, y), (w - 70, y), 40, 2)
        for x in range(90, w - 90, 23):
            cv2.rectangle(img, (x, y - 18), (x + 14, y - 6), 60, -1)
    cv2.line(img, (60, 60), (60, h - 60), 40, 2)
    cv2.line(img, (w - 60, 60), (w - 60, h - 60), 40, 2)
    return img


class TestPageGeometryMode:
    """Pipeline(page_geometry=True): every document type gets line-fitted
    quads and straightening; the default pipeline is untouched."""

    @pytest.fixture(scope='class')
    def pipelines(self):
        from document_processing import Pipeline
        # registration is on by default and takes precedence on a passport;
        # switch it off here so the geometry mode itself is what runs
        return (Pipeline(model_format='ONNX', device='cpu', verbose=False, page_registration=False),
                Pipeline(model_format='ONNX', device='cpu', verbose=False, page_registration=False,
                         page_geometry=True))

    @pytest.mark.parametrize('image,doctype', [
        ('../samples/DL_2011/1_CR_DL_2010.jpg', 'DL_2011'),
        # OUT OF SERVICE IN THE PUBLIC REPOSITORY since 2026-09-21: the sample under
        # this name is a single anonymised page, not the two-page spread the
        # n_pages == 2 expectation below describes (the spread was withdrawn on
        # 2026-08-25). Skipped rather than dropped, so the gap stays visible; it
        # returns with a synthetic spread.
        pytest.param('../samples/INTPASSPORT_2011/6_CR_INTPASSPORT_2011.jpg', 'INTPASSPORT_2011',
                     marks=pytest.mark.skip(reason='needs a two-page passport spread; the public '
                                                   'sample is a single page (spread withdrawn '
                                                   '2026-08-25)')),
    ])
    def test_geometry_mode_keeps_type_and_reports_evidence(self, pipelines, image, doctype):
        base, geo = pipelines
        rb = base.process_img(image, ocr=False, check_quality=False)
        rg = geo.process_img(image, ocr=False, check_quality=False)
        assert rb.doctype == rg.doctype == doctype
        assert 'PageGeometry' not in rb.meta_results
        info = rg.meta_results['PageGeometry']
        assert info['n_pages'] == (2 if doctype.startswith('INTPASSPORT') else 1)
        assert all(q['method'] == 'lines' for q in info['quads'])
        assert len(info['straighten']) == info['n_pages']
        canvas = rg.img_with_fixed_perspective
        assert canvas.ndim == 3 and min(canvas.shape[:2]) > 100
        # the page detectors saw the geometry pages: fields were found
        assert (rg.text_fields_meta or {}).get('bbox')

    def test_aspect_table(self):
        from document_processing.pipeline_modules.page_registration.geometry import aspect_for
        assert abs(aspect_for('DL_2020') - 54.0 / 85.6) < 1e-6
        assert abs(aspect_for('INTPASSPORT_1997') - 88.0 / 125.0) < 1e-6
        assert aspect_for('BIRTHCERT_2018') is None and aspect_for(None) is None


class TestLineRefine:
    @pytest.mark.parametrize('rot_deg,p1', [(2.0, 0.0), (-1.5, 0.03), (0.0, -0.04)])
    def test_recovers_small_rotation_and_keystone(self, rot_deg, p1):
        page = _ruled_page()
        h, w = page.shape
        distort = line_refine._model([np.radians(rot_deg), 0.0, p1, 0.0], w, h)
        warped = cv2.warpPerspective(page, distort, (w, h), borderValue=235)
        Hm, info = line_refine.refine_by_lines(warped, inset=30)
        assert info['applied'], info
        fixed = line_refine.apply_refinement(warped, Hm)
        # rules are horizontal again and the correction undid the distortion
        _, after = line_refine.refine_by_lines(fixed, inset=30)
        assert after['before'] < 0.3, after
        back = Hm @ distort
        back /= back[2, 2]
        corners = np.float64([[100, 100], [w - 100, 100], [w - 100, h - 100], [100, h - 100]])
        moved = cv2.perspectiveTransform(corners.reshape(1, -1, 2), back).reshape(-1, 2)
        assert np.linalg.norm(moved - corners, axis=1).max() < 6.0

    def test_straight_page_is_left_alone(self):
        Hm, info = line_refine.refine_by_lines(_ruled_page(), inset=30)
        assert Hm is None and info['reason'] == 'already straight'

    def test_blank_page_has_no_evidence(self):
        Hm, info = line_refine.refine_by_lines(np.full((764, 1060), 240, np.uint8), inset=30)
        assert Hm is None and not info['applied']

    def test_bent_page_is_unbent(self):
        """A page bent near one edge (text lines curve, tilt varies with x and
        grows toward the bottom) is remapped so that the lines are straight
        again; a flat page is left alone."""
        page = _ruled_page()
        h, w = page.shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        xn, yn = (xs - w / 2) / (w / 2), (ys - h / 2) / (w / 2)
        bend = 18.0 * xn ** 2 * (0.6 + yn)          # px, quadratic in x, stronger at the bottom
        bent = cv2.remap(page, xs, ys - bend, cv2.INTER_LINEAR, borderValue=235)
        v, info = line_dewarp.dewarp_by_lines(bent, inset=30)
        assert info['applied'], info
        # the map undoes the bend (compare where it matters: inside the page)
        err = np.abs(v[60:-60, 60:-60] - bend[60:-60, 60:-60])
        assert np.median(err) < 2.0 and err.max() < 6.0, (np.median(err), err.max())
        fixed = line_dewarp.apply_dewarp(bent, v)
        _, again = line_dewarp.dewarp_by_lines(fixed, inset=30)
        assert not again['applied'] and again['before'] < 0.5, again
        _, flat = line_dewarp.dewarp_by_lines(page, inset=30)
        assert not flat['applied'] and flat['reason'] == 'flat enough'

    def test_profile_tilt_sign(self):
        """A band whose lines run down to the right (positive image angle)
        reports a positive tilt, like the LSD segments do."""
        page = _ruled_page()
        h, w = page.shape
        rot = cv2.warpAffine(page, cv2.getRotationMatrix2D((w / 2, h / 2), -3.0, 1.0), (w, h),
                             borderValue=235)   # cv2: negative angle = clockwise = lines go down-right
        band = rot[h // 3:2 * h // 3, 40:w - 40]
        tilt, ratio = line_refine._profile_tilt(band, axis=1)
        assert ratio > line_refine.PROFILE_MIN_PEAK
        assert abs(tilt - 3.0) < 0.4, tilt
        segs = line_refine._segments(rot, 100)
        assert np.median(line_refine._seg_angle(segs)[np.abs(line_refine._seg_angle(segs)) < 12]) > 2.5
