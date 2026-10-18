"""Reading the machine-readable zone line by line, with a wider re-cut
when a line comes out the wrong length."""

class MrzReadingStage:
    """Pipeline methods: reading the machine-readable zone line by line, with a wider re-cut
    when a line comes out the wrong length.

    A part of Pipeline, not usable on its own: the methods read the
    models and the results that Pipeline.__init__ and process_img set up.
    """

    #: A line of a machine-readable zone is exactly 44 characters, and that is a
    #: rare luxury: the pipeline can tell that it read the line WRONG without being
    #: told, and try again. Anything shorter means the crop lost part of the line.
    MRZ_LINE_LEN = 44

    #: The MRZ is printed as ONE rectangle holding two lines, so both lines share
    #: the same horizontal span - but the detector does not know that. Measured
    #: over samples/: the two boxes of a zone start within 10 px of each other
    #: when the zone reads correctly and 90-182 px apart when it does not, and the
    #: characters outside the narrower box never reach the engine. That accounted
    #: for 23 of the 34 damaged lines; the engine was innocent (its CTC output had
    #: 117-552 time steps against the 57-71 a line needs).
    #:
    #: So the zone's own span is the FIRST retry candidate, then the ladder widens
    #: further. Deliberately a candidate and not a rewrite: forcing every MRZ box
    #: to the union span fixed the external passports and damaged an internal one
    #: (both its lines got shorter), because a wider crop can also pull in the page
    #: edge. Reading the detector's own crop first and widening only on a wrong
    #: length cannot lose a line that was already right.
    #: The ladder reaches 34% of the span on each side because that is what the
    #: worst measured case needed: a zone whose BOTH boxes are inset by nine
    #: characters (8_CR of EXTPASSPORTBIO) recovers its first line at +0.30 and not
    #: before. Wider steps cost nothing on a healthy document - the ladder is only
    #: walked for a line whose length is already wrong - and the crop is clamped to
    #: the canvas, so the last steps saturate instead of running away.
    MRZ_RETRY_GROWTH = (0.0, 0.05, 0.10, 0.16, 0.24, 0.34)

    #: The zone's alphabet is closed: capitals, digits and the filler. A line
    #: cannot begin or end with anything else, so a stray '.' or '_' at an edge is
    #: the page border caught by the crop, not text. Trimming those - and ONLY at
    #: the edges - is what keeps a widened crop from turning a correct
    #: 44-character line into a 45-character one.
    MRZ_ALPHABET = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')

    def _note_mrz_zone(self, text_fields: dict, img):
        """Remember the canvas and the MRZ boxes for the length self-check.

        Nothing is modified here: the boxes the detector produced stay exactly as
        they are, so `fields.bbox` and every other field are untouched. The only
        purpose is that _read_mrz can re-cut a line from the canvas later.

        Boxes are kept top to bottom, which is the order the OCR loop walks the
        patches in - the retry has to know which box a patch came from.
        """
        self._mrz_zone = None
        bboxes = text_fields.get('bbox') or []
        idx = [i for i, box in enumerate(bboxes) if box[-1] == 'MRZ']
        if not idx:
            return
        idx.sort(key=lambda i: (bboxes[i][1] + bboxes[i][3]) / 2)
        boxes = [list(bboxes[i][:4]) for i in idx]

        # The span of the zone as a whole: for a line whose own box is too narrow
        # this is where the missing characters are. Only from boxes that are
        # line-shaped - a box several line-heights tall is not a line, and its
        # edges say nothing about where the line ends (measured: the one such zone
        # reads worse from a widened crop).
        line_shaped = [b for b in boxes
                       if b[2] - b[0] >= 10 * max(1, b[3] - b[1])]
        span = None
        if len(line_shaped) > 1:
            span = (min(b[0] for b in line_shaped), max(b[2] for b in line_shaped))
        self._mrz_zone = {'canvas': img, 'boxes': boxes, 'span': span}

    @classmethod
    def _trim_to_mrz_alphabet(cls, text: str) -> str:
        """Drop edge characters that cannot occur in a machine-readable zone.

        Only the ends, and only characters outside the zone's closed alphabet: the
        page border sometimes lands inside a crop and comes back as '.' or '_'.
        Anything inside the line is left alone - a wrong character there is a
        reading error, and hiding it would be worse than showing it.
        """
        if not text:
            return text
        return text.strip(''.join(sorted(set(text) - cls.MRZ_ALPHABET)))

    def _read_mrz(self, line_index: int, text: str) -> str:
        """Re-read one MRZ line from a wider crop when it came out too short.

        The reading the detector's own crop produced is kept unless it is the
        wrong length; then the zone's full span is tried, then progressively wider
        crops, and the first result of exactly 44 characters wins. Falls back to
        the longest reading seen, which is still closer than the short one.

        Never invents a line: it only re-reads a box the detector found, so a zone
        whose second line was never detected stays a one-line zone. That matters -
        "read in" a missing line would turn a real miss into a plausible string.
        """
        text = self._trim_to_mrz_alphabet(text)
        if len(text) == self.MRZ_LINE_LEN:
            return text
        zone = self._mrz_zone
        if not zone or line_index >= len(zone['boxes']):
            return text

        canvas = zone['canvas']
        width = canvas.shape[1]
        x1, y1, x2, y2 = zone['boxes'][line_index]
        span = zone['span']
        best = text
        for growth in self.MRZ_RETRY_GROWTH:
            left, right = (span if span else (x1, x2))
            step = int(round((right - left) * growth))
            crop = canvas[y1:y2, max(0, left - step):min(width, right + step)]
            if crop.size == 0:
                continue
            candidate = self.ocr_lat.predict(crop)[self.ocr_lat.model_name]['ocr_output']
            candidate = self.ocr_lat.fix_errors(field_type='MRZ', text=candidate)
            candidate = self._trim_to_mrz_alphabet(candidate)
            if len(candidate) == self.MRZ_LINE_LEN:
                return candidate
            if len(candidate) > len(best):
                best = candidate
        return best
