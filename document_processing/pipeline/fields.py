"""Text fields on the canvas, the margin re-read, and the split of a field
into words (with the guard that reads a line whole when a word was lost)."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ..geometry import Chain, Offset, QuarterTurns, Scale, corners
from ..pipeline_modules.doc_detector.image_transformation import placement_rect


class FieldsStage:
    """Pipeline methods: text fields on the canvas, the margin re-read, and the split of a field
    into words (with the guard that reads a line whole when a word was lost).

    A part of Pipeline, not usable on its own: the methods read the
    models and the results that Pipeline.__init__ and process_img set up.
    """

    def _fields_detector(self, img, rotate_licence=False):
        """
        Detect text fields in document.

        Args:
            img: Input image
            rotate_licence: Whether to rotate license field

        Returns:
            dict: Detected text fields and patches
        """
        meta = self.results._meta_results.get('DocDetector') or {}
        pages = meta.get('pages') or []
        placements = meta.get('page_placements') or []
        if len(pages) >= 2 and len(placements) == len(pages):
            text_fields, frames = self._fields_from_pages(pages, placements)
            result = {self.text_fields.model_name: text_fields}
        else:
            result = self.text_fields.predict_transform(img)
            text_fields = result[self.text_fields.model_name]
            # a patch is the canvas cut at its box (see TextFieldsDetector)
            frames = [Offset(-float(box[0]), -float(box[1])) for box in text_fields['bbox']]
            self._read_margins(text_fields, frames, img)

        self._note_mrz_zone(text_fields, img)
        # Where each patch lies on the canvas (geometry.py): the map from the
        # patch as CUT (before the series/number turn below) to the canvas,
        # and the patch size then. Aligned with text_fields['bbox']; kept apart
        # from it because _split_words unpacks that dict by position.
        self.results._meta_results['FieldFrames'] = [
            (frame, patch.shape[1], patch.shape[0])
            for frame, patch in zip(frames, text_fields['warped_img'])]

        if rotate_licence:
            for i, field in enumerate(text_fields['bbox']):
                if field[-1] == 'Licence_number':
                    text_fields['warped_img'][i] = cv2.rotate(text_fields['warped_img'][i],
                                                              cv2.ROTATE_90_COUNTERCLOCKWISE)


        self.results._meta_results = self.results._meta_results | result

    def _fields_from_pages(self, pages, placements):
        """Run the field detector on each page and report boxes on the canvas.

        Why per page: the detector's input is 640x640 and the whole canvas is
        letterboxed into it, so on a stitched spread each page gets about half
        of that. Reading the pages separately gives each one the full input.

        The boxes come back in page coordinates and are moved onto the stitched
        canvas with the page's own (scale, dx, dy) from ``stitch_pages`` - every
        later stage (word splitting, drawing, the address path) addresses the
        canvas, so the boxes must speak its coordinates. The image PATCHES are
        kept as cut from the page: they are what OCR reads, and cutting them
        from the page rather than the canvas is exactly the point.
        """
        bbox, patches, frames = [], [], []
        for page, placement in zip(pages, placements):
            scale, dx, dy = placement
            got = self.text_fields.predict_transform(page)[self.text_fields.model_name]
            # The frame is written as the stages that make the PATCH out of the
            # CANVAS (geometry.py reads a chain that way round): take the page
            # off its place on the canvas, undo its resize, cut at the box.
            _, _, placed_w, placed_h = placement_rect(page, placement)
            unplace = (Offset(-dx, -dy), Scale(page.shape[1] / placed_w, page.shape[0] / placed_h))
            for box, patch in zip(got.get('bbox') or [], got.get('warped_img') or []):
                moved = list(box)
                moved[0] = int(round(box[0] * scale + dx))
                moved[1] = int(round(box[1] * scale + dy))
                moved[2] = int(round(box[2] * scale + dx))
                moved[3] = int(round(box[3] * scale + dy))
                bbox.append(moved)
                patches.append(patch)
                frames.append(Chain((*unplace, Offset(-float(box[0]), -float(box[1])))))
        return {'bbox': bbox, 'warped_img': patches}, frames

    def _read_margins(self, text_fields: dict, frames: list, img) -> None:
        """Re-cut the patch of a field labelled tight to its letters with a
        vertical margin, so the reading gets whole glyphs. In place.

        The box stays as the detector gave it - only the crop that is READ grows,
        and its frame (geometry.py) moves with it. The STS special marks are why:
        their lines stand so close that a labelling margin merged neighbours
        (39-41 % overlap on real canvases), so they were labelled tight, and the
        detector learnt to cut the tops and bottoms off the letters.

        The margin never reaches the next line: it stops halfway to the nearest
        box above or below that shares part of the width, of any label.
        Single-canvas path only; the per-page path reads passports, which have no
        such field.
        """
        margins = getattr(self.ocr_options, 'read_margin', None) or {}
        bboxes = text_fields.get('bbox') or []
        if not margins or not bboxes:
            return
        height = img.shape[0]
        for i, box in enumerate(bboxes):
            share = margins.get(box[-1])
            if not share:
                continue
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            pad = int(round((y2 - y1) * share))
            top, bottom = max(0, y1 - pad), min(height, y2 + pad)
            for j, other in enumerate(bboxes):
                if j == i or min(x2, other[2]) <= max(x1, other[0]):
                    continue                                  # no shared width
                if other[3] <= y1:                            # a line above
                    top = max(top, (int(other[3]) + y1 + 1) // 2)
                elif other[1] >= y2:                          # a line below
                    bottom = min(bottom, (y2 + int(other[1])) // 2)
            if (top, bottom) == (y1, y2):
                continue
            text_fields['warped_img'][i] = img[top:bottom, x1:x2]
            frames[i] = Offset(-float(x1), -float(top))

    #: An empty stretch on a line wider than this many typical words means the
    #: split dropped a word, and the line is read whole instead.
    #:
    #: Measured over 157 documents (all of samples/ plus 40 generator canons,
    #: where the defect lives), 815 fields: the widest gap on a line that really
    #: lost a long word has a median of 2.74 typical word widths, against 0.06 on
    #: intact lines - a factor of forty. The obvious alternative, the SHARE of the
    #: line covered by word boxes, was measured on the same run and rejected: it
    #: reads print DENSITY, not the integrity of the split, so the lowest coverage
    #: over samples/ belongs to a correctly read internal-passport Licence_number
    #: (0.64-0.75, three digit groups with wide gaps) and no threshold separates
    #: it from real damage. By the gap the same 24 fields sit at a median of 0.92
    #: and a maximum of 1.20, i.e. silent with room to spare.
    #:
    #: 3.0 rather than the 3.056 the selection rule produced: four significant
    #: digits pretend to a precision that 19 target fields cannot support. The
    #: rounding costs one extra false fire on the canons and none on samples/.
    #: Chosen on one half of the canons and reported on the other, so the number
    #: is not fitted to what it is measured on: 3 of 7 and 5 of 10 caught on the
    #: held-out half, which held the catch rate but raised the false count from 1
    #: to 4 - so the cost side of this number is an upper bound, not an
    #: expectation. Full report: notes/report_words_signal_choice_2026-08-25.md.
    #:
    #: Known limit, not a threshold to tune: when the missing word left NO gap
    #: because a neighbour's box swallowed it, geometry cannot see it at all.
    #: That is 8 of 19 measured cases - this guard covers about half of the
    #: complaint, the rest belongs to retraining the word detector.
    WORDS_MAX_GAP = 3.0

    #: Below this much fine-grained detail, a line carries no strokes and the
    #: fallback above must NOT re-read it.
    #:
    #: The guard treats "no word boxes" as "the split lost the text", but that
    #: observation has a second cause: there IS no text. On a real web sample
    #: (f2018_02) the name line is a blurred strip - anonymised at the source,
    #: no glyphs at all - and re-reading it whole produced «ЛВН»: an honest empty
    #: field turned into something that looks like an answer. For an identifier a
    #: plausible wrong value is worse than none.
    #:
    #: The measure is the variance of the Laplacian, i.e. how much fine detail
    #: the strip carries - strokes, not darkness. Three cheaper-looking
    #: alternatives were measured on the same run and rejected because they
    #: overlap: spread of brightness gives 18.8 on the blurred strip against a
    #: minimum of 18.1 on strips that do carry text, and the share of dark pixels
    #: 0.074 against 0.082. They measure how DARK a strip is; the question is
    #: whether it has strokes.
    #:
    #: The threshold comes from a blur ladder over 612 canon lines whose text is
    #: known to be there, blurred with a growing Gaussian (manufactured on
    #: purpose, so the number does not rest on the single real case): sharp text
    #: has a median of 1590, mild blur 249, and blur WIDER THAN THE STROKE 27 -
    #: an order-of-magnitude trough, because a stroke one or two pixels wide
    #: smeared over five stops being a stroke. 100 sits in that trough: it drops
    #: 0% of sharp lines, 1% of mildly blurred ones and all of the smeared ones,
    #: where 150 would cost 10% of the mildly blurred for nothing extra.
    #: Physics gives the trough; the point inside it is chosen by the cost of
    #: being wrong, which is asymmetric - refusing wrongly loses a repair and
    #: restores the pre-guard behaviour, accepting wrongly MANUFACTURES a value.
    #:
    #: Those percentages are measured on MANUFACTURED blur, which is easier than
    #: the real thing: a Gaussian is even, while a real blurred strip still
    #: carries compression, paper texture and sensor noise. On real documents
    #: this will therefore reject LESS than the ladder promises - it errs towards
    #: letting a blurred line through rather than dropping a line with text.
    #: Known limit, named before the measurement: an empty strip and a strip with
    #: VERY FAINT text look the same to this measure. Full report:
    #: notes/proposal_ink_check_2026-08-25.md.
    LINE_MIN_INK = 100.0

    @staticmethod
    def _widest_gap(word_boxes, line_width: float) -> float:
        """The widest empty stretch on the line, measured in typical word widths.

        A dropped word leaves a hole about as wide as a word; evenly spaced
        printing does not, however wide the spacing. That is the whole reason
        this is a ratio to the line's OWN median word width instead of a share
        of the line: an internal passport's Licence_number is three digit groups
        with wide gaps and is read correctly, and any absolute measure lumps it
        together with real damage.
        Edges count as gaps too - a word lost from the start or the end of a line
        leaves the hole at the border, not between boxes.
        Nothing found at all means the whole line is one hole, so the answer is
        infinity: that is the measured case where a field used to vanish without
        a trace.
        """
        if word_boxes is None or len(word_boxes) == 0:
            return float('inf')
        if not line_width:
            return 0.0
        spans = sorted((float(b[0]), float(b[2])) for b in word_boxes)
        widths = [b - a for a, b in spans if b > a]
        if not widths:
            return float('inf')
        typical = sorted(widths)[len(widths) // 2]
        if typical <= 0:
            return float('inf')
        gaps = [spans[0][0]]                          # empty stretch on the left
        end = spans[0][1]
        for a, b in spans[1:]:
            gaps.append(max(0.0, a - end))
            end = max(end, b)
        gaps.append(max(0.0, float(line_width) - end))  # and on the right
        return max(gaps) / typical

    @staticmethod
    def _line_ink(patch) -> float:
        """How much fine detail the line crop carries - strokes, not darkness.

        Variance of the Laplacian: a printed stroke is a sharp local change, and
        a strip that has none (blank paper, or a strip blurred past the width of
        a stroke) has almost no such change left, whatever its overall
        brightness.
        """
        if patch is None or getattr(patch, 'size', 0) == 0:
            return 0.0
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
        return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())

    def _split_words(self, text_fields: dict, doc_type:str):
        """
        Split text fields into words.

        Args:
            text_fields: Detected text fields

        Returns:
            dict: Text fields splitted into words
        """

        self._date_lines = {}   # filled below; never carried over from the previous image
        bboxes, patches = text_fields.values()
        frames = self.results._meta_results.get('FieldFrames') or [None] * len(bboxes)

        # The internal passport prints the series+number (and the FMS code)
        # twice, so the detector returns duplicate boxes; keep only the
        # highest-confidence one to avoid OCR'ing the same value twice.
        UNIQUE_FIELDS = ('Licence_number', 'Issue_organisation_code')
        drop = self._duplicate_field_indices(bboxes, UNIQUE_FIELDS)
        drop |= self._paired_duplicate_indices(bboxes)
        if drop:
            bboxes = [b for i, b in enumerate(bboxes) if i not in drop]
            patches = [p for i, p in enumerate(patches) if i not in drop]
            frames = [f for i, f in enumerate(frames) if i not in drop]

        # fields that will actually contribute to `result`. Multi-line fields
        # are labeled one detection per line (dataset convention), and the
        # per-class assembly below concatenates their words in list order -
        # so order the kept boxes top-to-bottom to get reading order (a pure
        # y-sort is safe: classes collect only their own boxes).
        kept = [i for i, bbox in enumerate(bboxes)
               if bbox[-1] in self.ocr_options.en_fields or bbox[-1] in self.ocr_options.ru_fields]
        kept.sort(key=lambda i: (bboxes[i][1] + bboxes[i][3]) / 2)

        # WordsDetector calls are independent (different crop each, same
        # reused session - ONNX Runtime sessions support concurrent run()
        # calls; verified no shared mutable state in pre/postprocessing, see
        # docs/progress-log.md), so dispatch them concurrently instead of one
        # at a time. Fields that don't need splitting need no call at all.
        split_idxs = [i for i in kept if bboxes[i][-1] in self.ocr_options.needed_split]
        words_by_idx = {}
        word_bbox_by_idx = {}
        if split_idxs:
            with ThreadPoolExecutor(max_workers=min(8, len(split_idxs))) as ex:
                futures = {i: ex.submit(self.words_detector.predict_transform, patches[i])
                          for i in split_idxs}
                for i, fut in futures.items():
                    detected = fut.result()[self.words_detector.model_name]
                    words_by_idx[i] = detected['warped_img']
                    word_bbox_by_idx[i] = detected['bbox']

        # Gap guard. A hole on the line wider than a few typical words means the
        # split dropped a word - measured twice: a 9 px crop where the detector
        # returned NO words (the field vanished without a trace), and a line whose
        # longest word («Тракторозаводский», 17 characters) was the one missed.
        # Reading the line whole recovers it; the price is that the engine emits
        # no spaces, so the line comes back glued, which is why this is a fallback
        # on a signal and not the default.
        #
        # SNILS is excluded BY CONSTRUCTION, not by hoping the threshold spares
        # it: there the engine is chosen by word-index parity (see _word_engines),
        # and a line read whole destroys the parity the routing depends on.
        fallback = []
        no_ink = []
        if doc_type != 'SNILS':
            # `kept` order is top-to-bottom, and a multi-line field collects its
            # lines in that same order below - so counting per label here gives
            # the line's ordinal WITHIN its field, which is what a reader of the
            # flag can act on. The raw box index would be meaningless outside
            # this function.
            seen = {}
            for i in kept:
                label = bboxes[i][-1]
                ordinal = seen.get(label, 0)
                seen[label] = ordinal + 1
                if i not in split_idxs:
                    continue
                boxes = word_bbox_by_idx.get(i)
                gap = self._widest_gap(boxes, patches[i].shape[1])
                if gap > self.WORDS_MAX_GAP and boxes is not None and len(boxes) == 0:
                    # No boxes at all has two causes, and only one of them is a
                    # lost split: the other is a line with nothing on it. Asked
                    # HERE only - where boxes were found the text is there by
                    # definition, and the question would be noise.
                    ink = self._line_ink(patches[i])
                    if ink < self.LINE_MIN_INK:
                        no_ink.append({'field': label, 'line': ordinal,
                                       'ink': round(ink, 2)})
                        continue
                if gap > self.WORDS_MAX_GAP:
                    words_by_idx[i] = [patches[i]]
                    fallback.append({'field': label, 'line': ordinal,
                                     'gap': None if gap == float('inf')
                                     else round(gap, 3)})
        if fallback:
            self.results._meta_results['WordsFallback'] = fallback
        if no_ink:
            self.results._meta_results['WordsNoInk'] = no_ink

        # Whole lines of the date fields that were read word by word, kept for
        # _reread_dates_whole: the split can drop a word without leaving a hole
        # wide enough for the gap guard (see there). SNILS is excluded for the
        # same parity reason as above; a line the guard already reads whole has
        # nothing to add.
        if doc_type != 'SNILS':
            for i in kept:
                label = bboxes[i][-1]
                if (i in split_idxs and 'date' in label.lower()
                        and not (len(words_by_idx.get(i, [])) == 1
                                 and words_by_idx[i][0] is patches[i])):
                    self._date_lines.setdefault(label, []).append(patches[i])

        result = {}
        word_bboxes = {}
        # Words per line, per field, in reading order: the join flattens the
        # lines, and a word torn by a line break is found at their boundary.
        self._field_lines = {}
        for i in kept:
            bbox = bboxes[i]
            words = words_by_idx[i] if i in words_by_idx else [patches[i]]
            self._field_lines.setdefault(bbox[-1], []).append(len(words))
            # None distinguishes "this field needs no splitting, so the whole patch is
            # the single word" from "the detector found exactly one word". Without it a
            # port that split a field it should not have would look like agreement.
            wb = word_bbox_by_idx[i] if i in word_bbox_by_idx else None

            if result.get(bbox[-1]):
                result[bbox[-1]]['patches'].extend(words)
                word_bboxes[bbox[-1]].append(wb)
            else:
                result[bbox[-1]] = {'patches': words,
                                    'ocr': []}
                word_bboxes[bbox[-1]] = [wb]

        # Probe only. Word PATCHES are deliberately not a stage (megabytes of pixels per
        # document, see conformance/spec/stages.md), but their boxes are a handful of
        # integers and they are what localises a wrong crop to the split rather than to
        # the OCR three stages later.
        for field_name, boxes in word_bboxes.items():
            self._emit(f'words.{field_name}.bbox', boxes)

        self.results._meta_results[self.words_detector.model_name] = result
        self._field_quads(bboxes, patches, kept, word_bbox_by_idx, frames)
        return result

    def _field_quads(self, bboxes, patches, kept, word_bbox_by_idx, frames):
        """Where the read fields and their word patches lie on the input image.

        A field patch is the canvas (or, on a spread, the page) cut at its box -
        its frame from _fields_detector says where (see geometry.py) - and the
        series/number patch is also turned (see _fields_detector); a word patch
        is its box cut from the field patch the way WordsDetector cuts it.
        """
        fields, words = {}, {}
        for i in kept:
            bbox = bboxes[i]
            label = bbox[-1]
            frame = frames[i] if i < len(frames) else None
            if frame is None:
                # no frame recorded (a caller that bypassed _fields_detector):
                # fall back to the canvas box itself
                x0, y0, x1, y1 = (float(v) for v in bbox[:4])
                frame = (Offset(-x0, -y0), x1 - x0, y1 - y0)
            to_canvas, fw, fh = frame
            field = self.results.to_input(to_canvas.to_input(corners(0.0, 0.0, fw, fh)))
            if field is None:
                # The way back is not known for this run, and it is not known for any
                # field of it: say so once, for the whole run, instead of per box.
                quads = self.results._meta_results.setdefault('Quads', {})
                quads['fields'] = quads['words'] = None
                return
            fields.setdefault(label, []).append(field)
            if i not in word_bbox_by_idx:
                words.setdefault(label, []).append(field)
                continue
            h, w = patches[i].shape[:2]
            patch = Chain((to_canvas,))
            if self.ocr_options.needs_licence_rotation and label == 'Licence_number':
                # the patch the words were found on is the field turned once: w x h of h x w
                patch = patch.then(QuarterTurns(h, w, 1))
            for box in word_bbox_by_idx[i]:
                wx0, wy0 = max(0, int(box[0])), max(0, int(box[1]))
                wx1, wy1 = min(w, int(box[2])), min(h, int(box[3]))
                quad = patch.to_input(corners(wx0, wy0, wx1, wy1))
                words.setdefault(label, []).append(self.results.to_input(quad))
        quads = self.results._meta_results.setdefault('Quads', {})
        quads['fields'], quads['words'] = fields, words

    @staticmethod
    def _duplicate_field_indices(bboxes, unique_fields) -> set:
        """For each field that must be unique, return the indices of all but the
        highest-confidence detection (bbox layout: [x1,y1,x2,y2,conf,cls,label]).
        Used to drop the duplicate series/number boxes on internal passports."""
        drop = set()
        for field in unique_fields:
            idxs = [i for i, b in enumerate(bboxes) if b[-1] == field]
            if len(idxs) > 1:
                best = max(idxs, key=lambda i: bboxes[i][4])
                drop.update(i for i in idxs if i != best)
        return drop

    #: IoU above which a <name>_ru box and a <name>_en box are one line read as
    #: both fields. Real pairs sit apart: ru/en lines of an external passport
    #: overlap at 0.2-0.3, of a driving licence at up to 0.5 (labels, 2026-09-26);
    #: the duplicates seen on a retrained detector overlap at 0.97-1.00.
    PAIRED_DUPLICATE_IOU = 0.8

    @classmethod
    def _paired_duplicate_indices(cls, bboxes) -> set:
        """Indices of the weaker box wherever a <name>_ru and a <name>_en box
        cover the same line (bbox layout: [x1,y1,x2,y2,conf,cls,label]).

        NMS runs per class on purpose (the ru/en pairs must not suppress each
        other), so nothing else removes a line the detector labels as BOTH
        languages. Seen on a retrained TextFields (2026-09-26): «Г. ВОЛГОГРАД/USSR»
        came out as Birth_place_ru at 0.92 and Birth_place_en at 0.62 on the same
        box, and the Latin engine read the Cyrillic line into Birth_place_en.
        The more confident label keeps the line.
        """
        def iou(a, b):
            w = min(a[2], b[2]) - max(a[0], b[0])
            h = min(a[3], b[3]) - max(a[1], b[1])
            if w <= 0 or h <= 0:
                return 0.0
            inter = w * h
            return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

        drop = set()
        ru = [i for i, b in enumerate(bboxes) if str(b[-1]).endswith('_ru')]
        for i in ru:
            name = str(bboxes[i][-1])[:-3]
            for j, b in enumerate(bboxes):
                if b[-1] == name + '_en' and iou(bboxes[i], b) > cls.PAIRED_DUPLICATE_IOU:
                    drop.add(j if bboxes[i][4] >= b[4] else i)
        return drop
