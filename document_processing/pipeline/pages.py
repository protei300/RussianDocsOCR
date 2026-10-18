"""Borders, the page canvas and its rectification: page registration of an
internal passport, the card of a vehicle registration certificate,
template-free page geometry, deskew, and the way back to the photo."""
import cv2
import numpy as np

from ..pipeline_modules import PageRegistrar
from ..geometry import Chain, Homography
from ..pipeline_modules.doc_detector.image_transformation import (DOC_MARGIN_FRAC, expand_quad,
                                                                   stitch_pages, stitched_geometry)
from ..pipeline_modules.page_registration.geometry import aspect_for


# A Borders quad overlapping the template-registered page by at least this
# much is the same page, and its edges (the physical page edges) give the
# geometry. The template homography replaces it only when the quad is
# clipped by the image border (the mask ran off the frame) and the two
# disagree strongly - otherwise the quad wins even when they differ, because
# on ordinary photos the template fit of page 3 is the less precise of the
# two (its static print is narrow and periodic; measured on samples/).
QUAD_SAME_PAGE_IOU = 0.40


QUAD_CLIPPED_IOU = 0.80


def _segments_payload(meta_results: dict):
    """Normalise DocDetector's contours into plain ``[[x, y], ...]`` lists.

    Only for the conformance probe (``borders.segments``). The stored contours are
    numpy arrays of shape ``(N, 1, 2)`` -- the layout ``cv2.findContours`` returns --
    which no other language reproduces, so a raw dump could not be compared. The
    reshape is O(number of points), a few hundred per document, and it is what makes
    the stage language-neutral rather than a convenience.

    Returns ``None`` when border detection did not run, which the checker treats as a
    stage that legitimately did not happen.
    """
    detector = (meta_results or {}).get('DocDetector') or {}
    segments = detector.get('segm')
    if not segments:
        return None
    out = []
    for contour in segments:
        if contour is None:
            out.append([])
            continue
        pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        out.append([[float(x), float(y)] for x, y in pts])
    return out


class PagesStage:
    """Pipeline methods: borders, the page canvas and its rectification: page registration of an
    internal passport, the card of a vehicle registration certificate,
    template-free page geometry, deskew, and the way back to the photo.

    A part of Pipeline, not usable on its own: the methods read the
    models and the results that Pipeline.__init__ and process_img set up.
    """

    def _doc_detector(self, img):
        """
        Detect document borders and fix perspective.

        Args:
            img: Input image

        Returns:
            np.ndarray: Image with fixed perspective
        """
        # Only internal passports are photographed as a two-page spread; all
        # other doc types are single cards/pages, so keep just the largest
        # segment (prevents background blobs from being stitched in). The
        # stitch direction is chosen automatically from the pages' layout.
        doc_type = (self.results.doctype or '')
        is_spread = 'intpassport' in doc_type.lower()
        max_pages = 2 if is_spread else 1
        result = self.doc_detector.predict_transform(img, stack='auto', max_pages=max_pages)
        self.results._meta_results = self.results._meta_results | result

        # img = result[self.doc_detector.model_name]['warped_img']
        # return img

    def _register_pages(self, img):
        """Rebuild the internal-passport canvas from template-registered pages.

        Runs after _doc_detector on the same upright image. The registrar
        decides WHICH pages are present and WHERE (template matching); the
        page's geometry then comes from the best available source:

        - the Borders quad of the same page (IoU >= QUAD_SAME_PAGE_IOU): its
          edges are the physical page edges and give the most faithful
          rectification (the segmentation is good on ordinary photos - the
          failures it has are page loss and clipped/bent edges);
        - the template homography itself when Borders has no quad for the
          page (flat scans) or its quad is clipped by the frame and disagrees
          (IoU < QUAD_CLIPPED_IOU);
        - a page the registrar could not find keeps a spare Borders quad
          (the next by vertical order), resized to the same frame.

        Every page is emitted into the canonical frame (same size, same
        cushion, one common scale - see PageRegistrar.native_scale), so the
        spread stacks consistently. If no page is available at all the Borders
        canvas stays as it is. The result replaces DocDetector.warped_img so
        every downstream stage sees it through img_with_fixed_perspective; the
        per-page evidence goes to meta_results['PageRegistration'].
        """
        doc_type = (self.results.doctype or '').lower()
        if doc_type.startswith('sts'):
            return self._register_card(img) if getattr(self, 'card_registration', True) else None
        if 'intpassport' not in doc_type or 'addr' in doc_type:
            return
        det = self.results._meta_results.get('DocDetector') or {}
        reg = self.page_registrar
        # page quads from the Borders contours: straight lines fitted to each
        # side (survive a thumb in the mask, a clipped edge) - see quad_fit
        quads, quad_infos = reg.page_quads(det.get('segm'), img.shape)
        regs = reg.register(img, quads)
        scale = reg.native_scale(regs)   # never upsample a page (see native_scale)
        used = set()
        plan = []                        # per page: ('quad', i) | ('template', reg) | None
        for r in regs:
            if not r.ok:
                plan.append(None)
                continue
            best_i, best_iou = None, 0.0
            for i, q in enumerate(quads):
                if i in used:
                    continue
                iou = reg.quad_iou(q, r.quad)
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i is not None and best_iou >= QUAD_SAME_PAGE_IOU:
                used.add(best_i)
                q = quads[best_i]
                h_img, w_img = img.shape[:2]
                clipped = bool((q[:, 0] <= 1).any() or (q[:, 1] <= 1).any()
                               or (q[:, 0] >= w_img - 2).any() or (q[:, 1] >= h_img - 2).any())
                if clipped and best_iou < QUAD_CLIPPED_IOU:
                    plan.append(('template', r, best_iou))
                else:
                    plan.append(('quad', best_i, best_iou))
            else:
                plan.append(('template', r, best_iou))
        spare = [i for i, _ in sorted(enumerate(quads), key=lambda t: t[1][:, 1].min())
                 if i not in used]
        pages, sources, quad_used, straightened, page_geometries = [], [], [], [], []
        for r, step in zip(regs, plan):
            if step is None:
                if spare:
                    qi = spare.pop(0)
                    M = reg.quad_matrix(expand_quad(quads[qi], DOC_MARGIN_FRAC), scale)
                    sources.append('borders-spare')
                    quad_used.append(qi)
                else:
                    sources.append(None)
                    quad_used.append(None)
                    continue
            elif step[0] == 'quad':
                M = reg.quad_matrix(expand_quad(quads[step[1]], DOC_MARGIN_FRAC), scale)
                sources.append('borders')
                quad_used.append(step[1])
            else:
                M = reg.page_matrix(r, scale)
                sources.append('template')
                quad_used.append(None)
            pages.append(reg.warp_matrix(img, M, scale))
            # straighten the page by its own lines / text-line profiles: catches a
            # quad corner that is off, and the rotation error of a template fit
            pages[-1], sinfo, straight = reg.straighten_with_geometry(pages[-1], scale)
            straightened.append(sinfo)
            # page -> the image the registrar received (geometry.py)
            page_geometries.append(Chain((Homography(M),)).then(straight))
        info = {'pages': [r.as_dict() for r in regs], 'sources': sources,
                'quad_iou': [round(step[2], 3) if step else None for step in plan],
                'all_registered': all(r.ok for r in regs), 'deskewed': None,
                'scale': round(scale, 4), 'page_size': list(reg.out_size(scale)),
                'quads': [{'quad': q.round(1).tolist(), **qi} for q, qi in zip(quads, quad_infos)],
                'quad_used': quad_used, 'straighten': straightened}
        self.results._meta_results['PageRegistration'] = info
        if pages:
            # hand the pages over the same way DocDetector does (pages, the
            # photo quads they came from, their placements on the canvas): the
            # per-page deskew and field detection then work on THESE pages
            page_quads = [quads[qi] if qi is not None else r.quad
                          for r, qi, src in zip(regs, quad_used, sources) if src]
            stitched, placements = stitch_pages(pages, page_quads, stack='vertical')
            det['pages'] = pages
            det['page_quads'] = page_quads
            det['page_placements'] = placements
            det['page_geometries'] = page_geometries
            det['geometry'] = stitched_geometry(pages, placements, page_geometries)
            det['warped_img'] = stitched
            self.results._meta_results['DocDetector'] = det

    #: Template matches a vehicle registration certificate needs before its template
    #: geometry replaces the Borders quad. The card is printed dense (captions,
    #: rules, guilloche), so a real match runs to a hundred or more points (median
    #: 141 on the client's test cards, 2026-10-02); 40 is the floor the form
    #: matching of the training-data side uses, below it the Borders canvas stays.
    CARD_MIN_INLIERS = 40

    #: Skew of the Borders canvas below which it is kept (see _card_skew): the card's
    #: corners found by the template, carried into that canvas, are fitted by a
    #: similarity; the residual as a share of the long side. The same measure the
    #: training-data side used on the client's canvases; visible skew starts at about
    #: 1.4 % (handoff of 2026-10-02).
    CARD_SKEW_KEEP = 0.01

    #: Least-squares re-fits after MAGSAC in the card's template match (PageRegistrar's
    #: refit_rounds): the skew decision above must not move with MAGSAC's samples.
    CARD_REFIT_ROUNDS = 5

    @staticmethod
    def _card_skew(reg, borders_quad, card_quad) -> float:
        """How far the Borders canvas bends the card out of a rectangle (share of its long side)."""
        M = reg.quad_matrix(borders_quad)
        found = cv2.perspectiveTransform(np.float32(card_quad).reshape(1, -1, 2), M).reshape(-1, 2)
        m = reg.margin
        ideal = np.float32([[m, m], [m + reg.page_w, m], [m + reg.page_w, m + reg.page_h], [m, m + reg.page_h]])
        A, _ = cv2.estimateAffinePartial2D(ideal, found, method=cv2.LMEDS)
        if A is None:
            return 1.0
        fit = ideal @ A[:, :2].T + A[:, 2]
        return float(np.sqrt(((fit - found) ** 2).sum(1).mean())) / max(reg.page_w, reg.page_h)

    def _card_registrar(self, doc_type: str):
        """The template registrar of one STS type, built on first use; None without templates."""
        if not hasattr(self, '_card_registrars'):
            self._card_registrars = {}
        if doc_type not in self._card_registrars:
            try:
                self._card_registrars[doc_type] = PageRegistrar(
                    doc_type, refit_rounds=self.CARD_REFIT_ROUNDS)
            except FileNotFoundError:
                self._card_registrars[doc_type] = None
        return self._card_registrars[doc_type]

    def _register_card(self, img):
        """Rebuild a vehicle registration certificate's canvas from its printed blank.

        The opposite policy to the passport's. There the Borders quad keeps the
        geometry whenever it agrees with the template, because the template fit of
        a sparsely printed page turns by degrees. Here the card often lies in a
        plastic sleeve or lamination and the Borders quad is the SLEEVE's edge: it
        overlaps the card well - it "agrees" - and still skews the canvas (on the
        client's 1217 cards ~12 % came out visibly skewed, 2026-10-02). The card is
        printed dense, so the template match is strong: when it reaches
        CARD_MIN_INLIERS the template geometry is taken and the page straightened
        by its own lines; otherwise the Borders canvas stays as it is.

        Where the card runs past the photo, the canvas is painted the card's own
        paper colour rather than the smeared edge (see PageRegistrar's ``fill``).
        """
        reg = self._card_registrar(self.results.doctype)
        det = self.results._meta_results.get('DocDetector') or {}
        if reg is None:
            return
        quads, quad_infos = reg.page_quads(det.get('segm'), img.shape)
        r = reg.register(img, quads)[0]
        info = {'pages': [r.as_dict()], 'sources': [None], 'policy': 'card',
                'min_inliers': self.CARD_MIN_INLIERS,
                'quads': [{'quad': q.round(1).tolist(), **qi} for q, qi in zip(quads, quad_infos)]}
        self.results._meta_results['PageRegistration'] = info
        if not r.ok or r.inliers < self.CARD_MIN_INLIERS:
            return
        # Where the Borders canvas is not skewed, keep it: re-cutting a canvas that
        # was right only resamples it (measured on the client's test cards: on the
        # 101 not skewed by Borders, 15 fields read better and 17 worse - noise; on
        # the 26 skewed by >= 1 %, 5 better, 1 worse). Skew, not offset: a sleeve
        # runs parallel to the card, so its edge sits 2-3 % off the card's even on a
        # straight canvas; what matters is whether the card comes out a rectangle.
        skews = [self._card_skew(reg, q, r.quad) for q in quads]
        info['borders_skew'] = round(min(skews), 4) if skews else None
        if skews and min(skews) < self.CARD_SKEW_KEEP:
            info['kept'] = 'borders: canvas not skewed'
            return
        scale = reg.native_scale([r])
        M = reg.page_matrix(r, scale)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.int32(np.round(r.quad))], 1)
        fill = tuple(float(np.median(img[..., c][mask > 0])) for c in range(3)) if mask.any() else None
        page = reg.warp_matrix(img, M, scale, fill=fill)
        page, sinfo, straight = reg.straighten_with_geometry(page, scale)
        page_geometry = Chain((Homography(M),)).then(straight)
        info.update(sources=['template'], scale=round(scale, 4), page_size=list(reg.out_size(scale)),
                    straighten=[sinfo])
        stitched, placements = stitch_pages([page], [r.quad], stack='vertical')
        det['pages'] = [page]
        det['page_quads'] = [r.quad]
        det['page_placements'] = placements
        det['page_geometries'] = [page_geometry]
        det['geometry'] = stitched_geometry([page], placements, [page_geometry])
        det['warped_img'] = stitched
        self.results._meta_results['DocDetector'] = det

    def _geometry_pages(self, img):
        """Template-free page geometry for any document type (page_geometry=True).

        Runs after _doc_detector (and after _register_pages, which takes
        precedence when it rebuilt the canvas). Every Borders page is
        re-cut from a quad of straight lines fitted to its contour, warped
        the way the Borders path warps it (own side lengths, 1 % cushion),
        then straightened by its own lines and unbent by their local tilt.
        Pages, quads and placements are handed over the way DocDetector does,
        so the per-page deskew and field detection work on these pages. The
        evidence goes to meta_results['PageGeometry'].
        """
        det = self.results._meta_results.get('DocDetector') or {}
        reg_info = self.results._meta_results.get('PageRegistration') or {}
        if any(reg_info.get('sources') or []):
            return                      # the registrar already rebuilt this canvas
        segm = det.get('segm') or []
        if not segm:
            return
        geo = self.page_geometry
        quads, quad_infos = geo.quads(segm, img.shape, aspect_for(self.results.doctype))
        pages, used, straightened, page_geometries = [], [], [], []
        for q, qi in zip(quads, quad_infos):
            # quad warp + straightening in one resampling; a side extrapolated
            # past the frame keeps its corners outside the photo
            page, sinfo, geometry = geo.rectify_with_geometry(
                img, q, keep_outside=qi.get('extrapolated') is not None)
            if page is None:
                continue
            pages.append(page)
            used.append(q)
            straightened.append(sinfo)
            page_geometries.append(geometry)
        if not pages:
            return
        stitched, placements = stitch_pages(pages, used, stack='auto')
        det['pages'] = pages if len(pages) >= 2 else []
        det['page_quads'] = used if len(pages) >= 2 else []
        det['page_placements'] = placements if len(pages) >= 2 else []
        det['page_geometries'] = page_geometries if len(pages) >= 2 else []
        det['geometry'] = stitched_geometry(pages, placements, page_geometries)
        det['warped_img'] = stitched
        self.results._meta_results['DocDetector'] = det
        self.results._meta_results['PageGeometry'] = {
            'quads': [{'quad': q.round(1).tolist(), **qi} for q, qi in zip(quads, quad_infos)],
            'straighten': straightened, 'n_pages': len(pages)}

    def _deskew(self, img):
        """Correct residual tilt of the perspective-fixed canvas and store it
        back so img_with_fixed_perspective returns the deskewed image.

        On a two-page spread each page is deskewed ON ITS OWN and the canvas is
        stitched back from the deskewed pages. Two reasons, both real: the pages
        of an open passport rarely lie at the same angle, and the per-page path
        needs the pages themselves (not the canvas) to hand to the field
        detector. The stitched result is still what every consumer sees.

        Deskew stays on after page registration, on purpose: registration
        aligns the printed BLANK, but the personalisation (names, MRZ, the
        vertical series/number) is printed in a separate pass and can sit a
        degree or two off the blank. The projection-profile deskew follows
        the text lines, which is what the word crops and the OCR need
        (measured: skipping it garbled the MRZ's second line on samples/).
        """
        info = self.results._meta_results.get('PageRegistration')
        if info is not None and any(info.get('sources') or []) and self.page_registrar is not None \
                and self.page_registrar.line_refine:
            # The registration pages were already straightened by their own
            # lines (line_refine, blob-tolerant). The projection-profile deskew
            # is NOT run on top of them: on a small photo (canvas ~500 px) it
            # took the dark cushion around a page for text and rotated the
            # straight page by 6 deg, garbling the MRZ (conformance case
            # 12_CR_INTPASSPORT_2011, measured 2026-09-17). The earlier note
            # that deskew had to stay on dates from before line_refine existed.
            info['deskewed'] = False
            return img
        if info is not None:
            info['deskewed'] = True
        meta = self.results._meta_results.get('DocDetector') or {}
        pages = meta.get('pages') or []
        if len(pages) >= 2:
            desked = [self.deskewer.deskew_with_geometry(p) for p in pages]
            desk_pages = [d[0] for d in desked]
            stitched, placements = stitch_pages(desk_pages, meta.get('page_quads') or [],
                                                stack='auto')
            if stitched is not None:
                # each page's own map followed by its own turn (geometry.py)
                page_geometries = [Chain((g,)).then(d[1]) if g is not None else Chain().then(d[1])
                                   for g, d in zip(meta.get('page_geometries') or [None] * len(pages),
                                                   desked)]
                meta['pages'] = desk_pages
                meta['page_placements'] = placements
                meta['page_geometries'] = page_geometries
                meta['geometry'] = stitched_geometry(desk_pages, placements, page_geometries)
                meta['warped_img'] = stitched
                return stitched
        desk, geometry = self.deskewer.deskew_with_geometry(img)
        if self.results._meta_results.get('DocDetector'):
            det = self.results._meta_results['DocDetector']
            det['warped_img'] = desk
            det['geometry'] = Chain((det.get('geometry'),)).then(geometry) \
                if det.get('geometry') is not None else Chain().then(geometry)
        else:
            # no doc_detector result (fallback path) -> stash under Angle90
            self.results._meta_results.setdefault('Angle90', {})['warped_img'] = desk
            self.results._meta_results['Deskew'] = {'geometry': geometry}
        return desk

    def _canvas(self, geometry):
        """The canvas later stages read is the previous one passed through `geometry`.

        Called wherever process_img takes a new image (see geometry.py); None
        means the image was handed on unchanged.
        """
        chain = self.results._meta_results.get('Geometry') or Chain()
        self.results._meta_results['Geometry'] = chain.then(geometry)

    def _borders_geometry(self):
        """Map from the current border canvas back to the image the border stage received.

        DocDetector keeps it up to date through page registration, page geometry
        and the deskew (its 'geometry' key); without a border result the only
        change is a deskew stashed under 'Deskew'.
        """
        det = self.results._meta_results.get('DocDetector')
        if det:
            return det.get('geometry')
        return (self.results._meta_results.get('Deskew') or {}).get('geometry')
