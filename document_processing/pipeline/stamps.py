"""Registration stamps on the residence registration page (INTPASSPORTADDR, issue
#27): which stamps there are, which address lines belong to which stamp, and the
order of the stamps."""
import numpy as np

from ..geometry import corners
from ..pipeline_modules import AddressLinesDetector, RegistrationStamps


class StampsStage:
    """Pipeline methods: registration stamps on the residence registration page.

    A part of Pipeline, not usable on its own: the methods read the
    models and the results that Pipeline.__init__ and process_img set up.
    """

    #: A canvas wider than this many heights is a spread of two pages, split down
    #: the middle (the page-registration canvas is either one page or a spread).
    SPREAD_ASPECT = 1.2

    def _registration_stamps(self, img):
        """Find the stamps on the address canvas and tie the address lines to them.

        Runs after _address_lines, on the same canvas. A line belongs to the stamp
        its centre lies in (measured 2026-10-16 on the 19 held-out canvases of the
        issue #27 author: all 113 address lines fall inside a stamp); a line inside
        no stamp keeps ``stamp: None``. A «Снят с регистрационного учёта» stamp holds
        no address - it is tied to the registration it cancels (``cancels``), and
        the registration nothing cancels is the current one (``current``).

        Results: meta_results['RegistrationStamps'] - the list of stamps, each
        ``{'kind', 'confidence', 'quad', 'lines', 'address', 'cancels', 'current'}``,
        freshest first (_order_stamps); and ``stamp`` (an index into that list) on
        every entry of meta_results['Address_lines']. ``address`` is the stamp's own
        lines joined by a newline, the way OCR['Address'] joins all of them: that one
        follows the page's reading order, so the lines of two stamps standing side
        by side alternate in it (seen on the issue #27 canvases) - the per-stamp
        address is the one that keeps a registration's lines together.
        """
        if self.registration_stamps is None:
            return
        name = self.registration_stamps.model_name
        obboxes = self.registration_stamps.predict(img)[name]['obbox']
        quads = [self._obbox_quad(ob) for ob in obboxes]
        order = self._order_stamps(quads, img.shape[1], img.shape[0])
        position = {det: k for k, det in enumerate(order)}

        lines = self.results._meta_results.get('Address_lines') or []
        line_boxes = getattr(self, '_address_line_obboxes', None) or []
        stamps = [{'kind': RegistrationStamps.KINDS.get(obboxes[det][7], 'unclear'),
                   'confidence': float(obboxes[det][5]),
                   'quad': self._stamp_quad_on_input(quads[det]),
                   'lines': [],
                   'address': '',
                   'cancels': None,
                   'current': False}
                  for det in order]
        for li, (meta, ob) in enumerate(zip(lines, line_boxes)):
            det = self._containing_stamp((float(ob[0]), float(ob[1])), quads)
            meta['stamp'] = None if det is None else position[det]
            if det is not None:
                stamps[position[det]]['lines'].append(li)
        for stamp in stamps:
            stamp['address'] = '\n'.join(
                self.ADDRESS_HW_PLACEHOLDER if lines[li]['kind'] == 'handwritten'
                else (lines[li]['text'] or '') for li in stamp['lines'])
        self._tie_cancellations(stamps)
        self.results._meta_results['RegistrationStamps'] = stamps

    @classmethod
    def _order_stamps(cls, quads, width, height) -> list:
        """Detection indices, freshest stamp first.

        The page is filled in order: on a spread the left page before the right
        one, on a page top to bottom; the freshest stamp is the last one filled.
        Checked 2026-10-16 by the session «Новые серии документов» on 15 labelled
        spreads with at least three stamps and a «снят» each, dates read by eye:
        the chronology matched this order on all 15. Two columns of stamps on one
        page did not occur there and are not covered (x only breaks ties).
        Ordering by the date inside the stamp comes with stage 2 of issue #27.
        """
        spread = width > cls.SPREAD_ASPECT * height

        def filled(k):
            cx, cy = quads[k].mean(axis=0)
            page = int(spread and cx >= width / 2)
            return page, cy, cx
        return sorted(range(len(quads)), key=filled)[::-1]

    @staticmethod
    def _tie_cancellations(stamps) -> None:
        """Set 'cancels' on every «снят» stamp and 'current' on the registration in
        force. ``stamps`` is freshest first.

        A «снят» is put in the next free place right after the registration it
        cancels, so it cancels the nearest registration filled before it; one that
        opens the canvas cancels a registration on a page not in the photo, and
        keeps ``cancels: None`` (both from the same 15 spreads; over all the
        labelling, 163 of 171 «снят» share a page with the nearest registration).
        The current registration is the freshest one nothing cancels.
        """
        cancelled = set()
        for k, stamp in enumerate(stamps):
            if stamp['kind'] != 'deregistered':
                continue
            j = next((j for j in range(k + 1, len(stamps))
                      if stamps[j]['kind'] == 'registered'), None)
            stamp['cancels'] = j
            if j is not None:
                cancelled.add(j)
        current = next((k for k, stamp in enumerate(stamps)
                        if stamp['kind'] == 'registered' and k not in cancelled), None)
        if current is not None:
            stamps[current]['current'] = True

    @staticmethod
    def _obbox_quad(obbox) -> np.ndarray:
        """The stamp's corners on the canvas, clockwise from the top-left corner of
        its upright patch - the same convention as the address-line quads."""
        w_i, h_i = max(1, int(round(obbox[2]))), max(1, int(round(obbox[3])))
        crop = AddressLinesDetector.crop_geometry(obbox[:5])
        return crop.to_input(corners(0, 0, w_i, h_i))

    def _stamp_quad_on_input(self, quad):
        """The quad on the image passed to process_img, as a list; None when the
        run's way back is not known (see PipelineResults.to_input)."""
        on_input = self.results.to_input(quad)
        return None if on_input is None else on_input.tolist()

    @staticmethod
    def _containing_stamp(point, quads):
        """Index of the stamp whose quad holds the point; when quads overlap, the
        one whose centre is nearest. None when no quad holds it."""
        best, best_d = None, None
        for k, quad in enumerate(quads):
            edges = np.roll(quad, -1, axis=0) - quad
            rel = np.asarray(point) - quad
            cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
            if not ((cross >= 0).all() or (cross <= 0).all()):
                continue
            d = float(np.hypot(*(np.asarray(point) - quad.mean(axis=0))))
            if best_d is None or d < best_d:
                best, best_d = k, d
        return best
