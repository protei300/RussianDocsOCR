"""A whole registration page: photo -> what is in the frame -> pages -> stamps -> fields.

The page side follows ``Melnikov/Text_fields/registration_page/PIPELINE_PAGE.md``
(the plan agreed with the user), the stamp side is ``stamp.StampReader``:

* **step -1, what is in the frame.** The library reads every document of the
  frame (``process_frame``). Documents of different types -> ``multiple_documents``,
  nothing is read; another document type -> ``not_registration``; type ``NONE``
  -> the stamp detector on the whole frame, a stamp at least
  ``stamp_conf_fallback`` sure -> ``stamp_fragment`` (a cut-out stamp, read as it
  is), else ``no_document``.
* **step 1, pages.** The library's rectified pages; a canvas wider than
  ``spread_aspect`` heights is a spread and is split at the fold; a page with more
  than ``cut_outside_share`` of it outside the frame is ``cut`` and not read; each
  page is classified on its own - only registration pages are read, and a page
  holding a stamp at least ``stamp_conf_fallback`` sure is one whatever the
  classifier says (the stamp detector is trusted over it, decision #185).
* **step 2, stamps.** The library's stamp detector on the rectified canvas, each
  stamp cut by its oriented box with no margin - exactly how the library cuts it.
* **step 10, order and links.** Numbered by place (left page first, top to bottom);
  listed freshest first - by date when every stamp has one, else by place
  (``order_by``); a ``deregistered`` stamp cancels the latest open registration,
  the latest registration nothing cancelled is ``current``.

The library is used as it is (``document_processing`` is not modified). It wants
RGB; the stamp reader was tuned on BGR crops, so a crop is converted before reading.
Coordinates in the answer are in the pixels of the uploaded image.
"""
from __future__ import annotations

import contextlib
import copy
import io
import threading
import time
from typing import Any

import cv2
import numpy as np

from .config import RegistrationConfig
from .stamp import FIELDS, StampReader

ADDR = 'INTPASSPORTADDR_ALL'

#: English labels of the stamp fields, as ``labels.FIELD_LABELS`` does for the library's
FIELD_DISPLAY = {
    'date': 'Date', 'region': 'Region', 'district': 'District', 'locality': 'Locality',
    'street': 'Street', 'house': 'House', 'building': 'Building', 'structure': 'Structure',
    'flat': 'Flat', 'code': 'Department code', 'addr_line': 'Address line', 'korp': 'Building',
}


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _round_quad(q) -> list[list[float]] | None:
    return None if q is None else np.asarray(q, np.float64).reshape(-1, 2).round(1).tolist()


class RegistrationReader:
    def __init__(self, cfg: RegistrationConfig | None = None):
        self.cfg = cfg or RegistrationConfig()
        self._stamps: StampReader | None = None
        self._lock = threading.Lock()

    @property
    def stamp_reader(self) -> StampReader:
        if self._stamps is None:
            self._stamps = StampReader(self.cfg)
        return self._stamps

    # -- inside the library lease: everything that touches the pipeline ---------------------------
    def cut(self, pipe, rgb: np.ndarray, *, docconf: float = 0.5, img_size: int = 1500) -> dict:
        """Steps -1, 1 and 2. Must run while the pipeline is leased; returns plain data
        (crops are copies), so the stamps can be read after the lease is released."""
        t0 = time.perf_counter()
        with _quiet():
            res = pipe.process_frame(rgb, ocr=False, find_text_fields=False, check_quality=True,
                                     docconf=docconf, img_size=img_size)
        types = sorted({r.doctype or 'NONE' for r in res})
        top = res[0]
        quality = {str(k): (float(v) if isinstance(v, (int, float, np.generic)) else str(v))
                   for k, v in (top.quality or {}).items()}
        out = {'doc_type': top.doctype or 'NONE', 'types_in_frame': types, 'quality': quality,
               'pages': [], 'stamps': [], 'timings': {}}
        t = top.doctype or 'NONE'
        if len(res) > 1 and len(types) > 1:
            out['decision'] = 'multiple_documents'
        elif t == ADDR:
            out['decision'] = 'registration'
        elif t != 'NONE':
            out['decision'] = 'not_registration'
        else:
            out['decision'] = ('stamp_fragment' if self._best_stamp(pipe, rgb) >= self.cfg.stamp_conf_fallback
                               else 'no_document')
        out['timings']['frame'] = time.perf_counter() - t0
        if out['decision'] not in ('registration', 'stamp_fragment'):
            return out
        t0 = time.perf_counter()
        if out['decision'] == 'registration':
            # the way back to the photo is copied: the caller zeroes every array the
            # pipeline kept once the lease ends (runtime rule 11), its matrices included,
            # and the stamps' boxes are mapped back after that
            geometry = copy.deepcopy(top.geometry)

            def to_photo(pts, g=geometry):
                pts = np.asarray(pts, np.float64).reshape(-1, 2)
                return pts if g is None else g.to_input(pts)
            canvas = top.img_with_fixed_perspective
            pages = self._pages(top, rgb)
            obbs = self._detect(pipe, canvas)
            for p in pages:
                if p['cut']:
                    p.update(decision='cut', page_type=None, page_conf=None, decided_by='cut', best_stamp_conf=None)
                else:
                    kind, t, c = self._page_kind(pipe, p['img'])
                    by = 'classifier'
                    # the stamp detector outranks the page classifier (user, decision #185): on half a
                    # spread the classifier often says NONE or another type while the stamps are sure
                    best = max((float(o[5]) for o in obbs if p['x0'] <= o[0] < p['x1']), default=0.0)
                    if kind != 'registration' and best >= self.cfg.stamp_conf_fallback:
                        kind, by = 'registration', 'stamps'
                    p.update(decision=kind, page_type=t, page_conf=c, decided_by=by,
                             best_stamp_conf=round(best, 4))
                p.pop('img', None)
        else:                                       # the photo itself is the stamp's page
            canvas, to_photo = rgb, (lambda pts: np.asarray(pts, np.float64).reshape(-1, 2))
            h, w = rgb.shape[:2]
            pages = [{'x0': 0.0, 'x1': float(w), 'y0': 0.0, 'quad': None, 'outside': 0.0, 'cut': False,
                      'decision': 'registration', 'page_type': ADDR, 'page_conf': None,
                      'decided_by': 'stamp_fragment', 'best_stamp_conf': None}]
            obbs = self._detect(pipe, canvas)
        out['pages'] = pages
        out['stamps'] = self._stamp_crops(canvas, to_photo, pages, obbs, pipe.registration_stamps)
        out['timings']['stamps_found'] = time.perf_counter() - t0
        return out

    def _best_stamp(self, pipe, img) -> float:
        det = pipe.registration_stamps
        if det is None:
            return 0.0
        with _quiet():
            obb = det.predict(img)[det.model_name]['obbox']
        return max((float(o[5]) for o in obb), default=0.0)

    def _pages(self, r, img) -> list[dict]:
        from document_processing.geometry import corners
        det = r._meta_results.get('DocDetector') or {}
        pages = det.get('pages') or []
        places = det.get('page_placements') or []
        if not pages:
            pages, places = [r.img_with_fixed_perspective], [(1.0, 0.0, 0.0)]
        if len(pages) == 1:                         # Borders may keep a whole spread as one page
            h, w = pages[0].shape[:2]
            if w > self.cfg.spread_aspect * h:
                s0, dx0, dy0 = places[0]
                half = w // 2
                pages = [pages[0][:, :half], pages[0][:, half:]]
                places = [(s0, dx0, dy0), (s0, dx0 + half * s0, dy0)]
        H, W = img.shape[:2]
        frame = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        out = []
        for pg, (s, dx, dy) in zip(pages, places):
            h, w = pg.shape[:2]
            q = r.to_input(corners(0, 0, w, h) * s + np.array([dx, dy]))
            outside = 0.0
            if q is not None:
                # a page shot edge to edge touches the frame and is whole; it is cut
                # only when a noticeable share of it lies outside the frame
                area = abs(cv2.contourArea(q.astype(np.float32)))
                inter, _ = cv2.intersectConvexConvex(q.astype(np.float32), frame)
                outside = 1.0 - inter / area if area > 0 else 0.0
            out.append({'img': pg, 'x0': float(dx), 'x1': float(dx + w * s), 'y0': float(dy),
                        'quad': _round_quad(q), 'outside': round(float(outside), 3),
                        'cut': bool(outside > self.cfg.cut_outside_share)})
        out.sort(key=lambda p: (p['x0'], p['y0']))  # left to right, top to bottom
        return out

    def _page_kind(self, pipe, page_img) -> tuple[str, str, float]:
        cls = pipe.doctype_angles
        with _quiet():
            meta = cls.predict(page_img)[cls.model_name]
        t, conf = meta['doc_type'] or 'NONE', float(meta['doc_type_confidence'])
        if t == ADDR:
            return 'registration', t, conf
        if t == 'NONE' and self._best_stamp(pipe, page_img) >= self.cfg.stamp_conf_fallback:
            return 'registration', t, conf
        return 'not_registration', t, conf

    @staticmethod
    def _detect(pipe, canvas) -> list:
        det = pipe.registration_stamps
        if det is None:
            return []
        with _quiet():
            return list(det.predict(canvas)[det.model_name]['obbox'])

    def _stamp_crops(self, canvas, to_photo, pages, obbs, det) -> list[dict]:
        from document_processing.pipeline_modules import AddressLinesDetector
        if det is None or not obbs:
            return []
        h, w = canvas.shape[:2]
        spread = w > self.cfg.spread_aspect * h
        usable = {i for i, p in enumerate(pages) if p['decision'] == 'registration'}

        def place(ob):
            return int(spread and ob[0] >= w / 2), float(ob[1]), float(ob[0])
        out = []
        for n, ob in enumerate(sorted(obbs, key=place), 1):
            page = next((i for i, p in enumerate(pages) if p['x0'] <= ob[0] < p['x1']), 0)
            if page not in usable:
                continue
            crop = AddressLinesDetector.crop_rotated(canvas, ob[:5])
            if crop is None or not crop.size:
                continue
            crop_map = AddressLinesDetector.crop_geometry(ob[:5])     # crop -> canvas
            ch, cw = crop.shape[:2]
            q_crop = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float64)
            out.append({'n': n, 'page': page, 'kind': det.KINDS.get(ob[7], 'unclear'), 'conf': float(ob[5]),
                        'crop_bgr': cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        'crop_to_photo': (lambda pts, m=crop_map: to_photo(m.to_input(
                            np.asarray(pts, np.float64).reshape(-1, 2)))),
                        'quad': _round_quad(to_photo(crop_map.to_input(q_crop)))})
        return out

    # -- after the lease: the stamps and the answer -----------------------------------------------
    def read_stamps(self, cut: dict) -> dict:
        """Read every cut stamp, order and link them, build the answer."""
        t0 = time.perf_counter()
        stamps = []
        for st in cut['stamps']:
            crop = st.pop('crop_bgr')
            try:
                with self._lock:                    # the dictionaries cache in plain dicts
                    res = self.stamp_reader.read(crop)
            finally:
                crop.fill(0)                        # a stamp is personal data: zeroed once read
            to_photo = st.pop('crop_to_photo')
            boxes = []
            for i, b in enumerate(res['boxes']):
                boxes.append({'id': f'{st["n"]}.{i}', 'label': b['field'], 'conf': b['conf'], 'text': b['text'],
                              'quad': _round_quad(to_photo(b['quad']))})
            fields = [{'name': f, 'display': FIELD_DISPLAY[f], 'value': res['fields'][f]}
                      for f in FIELDS if res['fields'].get(f)]
            stamps.append({'index': st['n'], 'page': st['page'], 'kind': st['kind'],
                           'conf': round(st['conf'], 4), 'quad': st['quad'],
                           'fields': fields, 'ocr': dict(res['fields']), 'boxes': boxes})
        order_by = self._order_and_links(stamps)
        notes = []
        q = cut['quality']
        if q.get('Glare') == 'bad' or q.get('Blur') == 'bad':
            notes.append('quality_low')
        d = cut['decision']
        notes += {'multiple_documents': ['one_document_expected'], 'not_registration': ['not_registration_page'],
                  'no_document': ['no_document']}.get(d, [])
        if cut['pages'] and all(p['decision'] == 'cut' for p in cut['pages']):
            notes.append('page_cut')
        timings = {k: round(v, 3) for k, v in cut['timings'].items()}
        timings['stamps_read'] = round(time.perf_counter() - t0, 3)
        return {
            'doc_type': cut['doc_type'],
            'decision': d,
            'recognised': d in ('registration', 'stamp_fragment'),
            'doc_conf': q.get('DocConf'),
            'types_in_frame': cut['types_in_frame'],
            'notes': notes,
            'coord_space': 'input',
            'pages': [{'index': i, **{k: p[k] for k in ('decision', 'decided_by', 'page_type', 'page_conf',
                                                         'best_stamp_conf', 'quad', 'cut', 'outside')}} for i, p in enumerate(cut['pages'])],
            'order_by': order_by,
            'stamps': stamps,
            'quality': {k: v for k, v in q.items() if k != 'DocConf'},
            'timings': timings,
        }

    @staticmethod
    def _order_and_links(stamps: list[dict]) -> str | None:
        from document_processing.pipeline.dates import canonical_date
        if not stamps:
            return None
        by_pos = sorted(stamps, key=lambda s: s['index'])
        keys = []
        for s in by_pos:
            v = canonical_date('Issue_date', s['ocr'].get('date', ''))
            keys.append(tuple(reversed(v.split('.'))) if v else None)
        order_by = 'date' if all(keys) else 'position'
        chrono = ([s for _, s in sorted(zip(keys, by_pos), key=lambda t: (t[0], t[1]['index']))]
                  if order_by == 'date' else by_pos)
        open_regs: list[dict] = []
        for s in chrono:
            s['cancels'], s['current'] = None, False
            if s['kind'] == 'registered':
                open_regs.append(s)
            elif s['kind'] == 'deregistered' and open_regs:
                s['cancels'] = open_regs.pop()['index']
        if open_regs:
            open_regs[-1]['current'] = True
        stamps[:] = list(reversed(chrono))      # freshest first
        return order_by
