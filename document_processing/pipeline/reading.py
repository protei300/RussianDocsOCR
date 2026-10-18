"""Reading the words: which engine reads which word, the per-word and the
batched GPU paths, joining words into fields, the address lines."""
import re

from ..pipeline_modules import AddressLinesDetector
from ..geometry import corners
from .sts_marks import glue_torn_words, parse_leasing


class ReadingStage:
    """Pipeline methods: reading the words: which engine reads which word, the per-word and the
    batched GPU paths, joining words into fields, the address lines.

    A part of Pipeline, not usable on its own: the methods read the
    models and the results that Pipeline.__init__ and process_img set up.
    """

    #: What a handwritten address line reads as (OCR is printed-only). ASCII
    #: brackets on purpose: the report gets printed to consoles that are still
    #: cp1251/cp866 on Windows, and fancier delimiters (U+27E8/U+27E9) are not in
    #: those codepages - printing the result raised UnicodeEncodeError.
    ADDRESS_HW_PLACEHOLDER = '[рукопись]'

    def _address_lines(self, img):
        """
        Recognize the multi-line registration address (INTPASSPORTADDR).

        Runs the oriented-bbox address-line detector directly on the
        perspective-fixed canvas (it handles line tilt itself, so no deskew is
        needed), yielding upright line patches already ordered top-to-bottom.
        Each line is classified printed-vs-handwritten; printed lines are split
        into words and OCR'd with the printed Russian model, handwritten lines
        are flagged (OCR is printed-only) and kept as a placeholder so the line
        order/structure is preserved.

        Result is stored as results.meta_results['OCR']['Address'] — lines
        joined by newline. Per-line kinds are stored under
        meta_results['Address_lines'] and a handwritten flag on the OCR dict.

        Args:
            img: Perspective-fixed canvas.
        """
        result = self.address_lines.predict_transform(img)
        self.results._meta_results = self.results._meta_results | result
        line_patches = result[self.address_lines.model_name]['warped_img']
        obboxes = result[self.address_lines.model_name]['obbox']
        line_quads = []
        # ocr_device is only 'gpu' when ocr_gpu_batch=True was explicitly
        # requested (see __init__) - safe to gate on it directly.
        use_batch = self.ocr_device == 'gpu'

        line_meta = []
        self._address_line_obboxes = []   # one per line_meta entry
        line_slots = []          # 'hw' or ('printed', index into printed_words)
        printed_words = []       # list[list[word_patch]], one list per printed line
        has_handwritten = False

        for patch, obbox in zip(line_patches, obboxes):
            if patch is None or patch.size == 0:
                continue
            self._address_line_obboxes.append(obbox)
            # The line on the input image, in the order of line_meta
            crop = AddressLinesDetector.crop_geometry(obbox[:5])
            line_quad = self.results.to_input(crop.to_input(corners(0, 0, patch.shape[1], patch.shape[0])))
            # None once means None for the whole run (see PipelineResults.to_input).
            line_quads = None if line_quad is None else line_quads
            if line_quads is not None:
                line_quads.append(line_quad)
            kind, prob = self.address_textkind.predict(patch)[self.address_textkind.model_name]
            if kind == 'handwritten':
                has_handwritten = True
                line_meta.append({'kind': 'handwritten', 'p_handwritten': prob, 'text': None})
                line_slots.append(('hw', None))
                continue
            # printed line: split into words (left-to-right); OCR happens below
            words = self.words_detector.predict_transform(patch)[self.words_detector.model_name]['warped_img']
            if not words:
                words = [patch]
            words = [w for w in words if w is not None and w.size > 0]
            line_slots.append(('printed', len(printed_words)))
            printed_words.append(words)
            line_meta.append({'kind': 'printed', 'p_handwritten': prob, 'text': None})

        if use_batch:
            # one padded-batch call per engine across ALL address words (see
            # ocr_batch.py / _ocr_batched for why this matters on GPU).
            flat = [w for words in printed_words for w in words]
            cyr_texts = self.ocr_cyr.predict_batch(flat)
            lat_texts = self.ocr_lat.predict_batch(flat)
            flat_texts = [lat if self._is_number_token(lat) else ru
                          for ru, lat in zip(cyr_texts, lat_texts)]
            texts_by_line = []
            it = iter(flat_texts)
            for words in printed_words:
                texts_by_line.append([next(it) for _ in words])
        else:
            texts_by_line = [[self._route_word_ocr(w) for w in words] for words in printed_words]

        address_lines_text = []
        for slot, meta in zip(line_slots, line_meta):
            kind, idx = slot
            if kind == 'hw':
                address_lines_text.append(self.ADDRESS_HW_PLACEHOLDER)
                continue
            line_words = [t.strip() for t in texts_by_line[idx] if t]
            line_text = ' '.join(line_words)
            meta['text'] = line_text
            address_lines_text.append(line_text)

        self.results._meta_results['Address_lines'] = line_meta
        self.results._meta_results.setdefault('Quads', {})['address_lines'] = line_quads
        if address_lines_text:
            ocr_dict = self.results._meta_results.get('OCR') or {}
            ocr_dict['Address'] = '\n'.join(address_lines_text)
            ocr_dict['Address_has_handwritten'] = has_handwritten
            self.results._meta_results['OCR'] = ocr_dict

    def _route_word_ocr(self, word) -> str:
        """Pick the right OCR engine for an address word (serial/per-word path;
        used on CPU - see _address_lines for the batched GPU path).

        Run both the Cyrillic and Latin engines and keep the Latin result only
        when it is digit-dominated (a house/building/flat number), otherwise the
        Cyrillic result. The Latin engine reads digits and Latin letters
        cleanly, so its digit-dominated output is the reliable signal for
        numeric tokens; Cyrillic words are taken from the Cyrillic engine.
        """
        ru = self.ocr_cyr.predict(word)[self.ocr_cyr.model_name]['ocr_output']
        en = self.ocr_lat.predict(word)[self.ocr_lat.model_name]['ocr_output']
        return en if self._is_number_token(en) else ru

    def _engine_by_year(self, field_name: str):
        """The engine this field takes on this form year ('cyr' or 'lat'), or
        None when the ru_fields/en_fields routing decides."""
        rules = getattr(self.ocr_options, 'engine_by_year', None) or {}
        return (rules.get(field_name) or {}).get(getattr(self, '_doc_year', None))

    @staticmethod
    def _is_number_token(en_text: str) -> bool:
        """True if the eng+nums OCR output is digit-dominated (a number word
        such as a house/building/flat number), i.e. at least one digit and
        digits not outnumbered by letters among the alphanumeric characters."""
        alnum = [c for c in (en_text or '') if c.isalnum()]
        if not alnum:
            return False
        digits = sum(c.isdigit() for c in alnum)
        letters = len(alnum) - digits
        return digits >= 1 and digits >= letters

    def _ocr(self, words_dict: dict, doc_type: str):
        """
        Perform OCR on splitted words.

        With a v2 engine (accurate/fast) AND ocr_gpu_batch=True (so
        ocr_device=='gpu' - see __init__), routes through the batched path
        (_ocr_batched): padded-batch inference calls per engine instead of one
        call per word, which avoids ONNX Runtime recompiling the CUDA graph
        for every distinct patch width (measured 400-3700x faster on real word
        crops), at the cost of a small measured decode-drift risk (see
        pipeline_modules/ocr_batch.py and docs/progress-log.md). Otherwise
        (CPU) uses the per-word path (_ocr_serial), which is exact.

        Args:
            words: Text fields splitted into words

        Returns:
            dict: OCR text for input words
        """
        # ocr_device is only 'gpu' when ocr_gpu_batch=True was explicitly
        # requested (see __init__) - safe to gate on it directly.
        if self.ocr_device == 'gpu':
            self._ocr_batched(words_dict, doc_type)
        else:
            self._ocr_serial(words_dict, doc_type)
        if 'birthcert' in doc_type.lower():
            self._clean_ruler_artifacts()

    #: Characters the ruler dots come back as. Dots, dashes and underscores were
    #: the obvious ones; commas and quotes were added after looking at what the
    #: engine actually emits on this form - «28., ИЮЛЯ 2010», «08.,.АВГУСТА.,.2008»,
    #: «"""СЕМ","" ПОННИЛОВИЧ». The set is deliberately limited to marks observed
    #: as artifacts: every character added here is one that can no longer survive
    #: on its own in a birth-certificate field.
    _RULER_MARKS = r'.,_\-"'

    def _clean_ruler_artifacts(self):
        """The 1998 birth-certificate form prints dotted ruler lines under
        every field value; they land inside the field crops and OCR emits runs
        of those marks around the real words. They carry no information on this
        form, so collapse them (single in-word dots - the digit birth date,
        abbreviations - are left untouched).

        Only runs of two or more, plus marks standing alone between spaces, are
        removed. That is what keeps real punctuation: the comma in
        «Г. ИРКУТСК, ИРКУТСКАЯ ОБЛАСТЬ» is attached to a word, and the hyphen in
        a series like «II-МЮ» sits between letters, so neither matches.

        Runs only for birth certificates (see _ocr), so no other document type is
        affected by what is in _RULER_MARKS."""
        ocr = self.results._meta_results.get('OCR')
        if not ocr:
            return
        runs = re.compile(f'[{self._RULER_MARKS}]{{2,}}')
        lone = re.compile(f'(?:^|(?<=\\s))[{self._RULER_MARKS}](?=\\s|$)')
        for key, value in list(ocr.items()):
            if not isinstance(value, str):
                continue
            text = runs.sub(' ', value)       # runs of ruler marks
            text = lone.sub(' ', text)        # marks standing alone
            ocr[key] = re.sub(r'\s+', ' ', text).strip()

    def _word_engines(self, field_name: str, doc_type: str, n_words: int) -> list:
        """The engine that reads each word of a field: 'cyr', 'lat', or None when
        the word is not read at all. One rule for both OCR paths, in this order:
        the field's engine for this form year (engine_by_year), then ru_fields,
        then dates and en_fields to Latin.

        SNILS is the one doc type where a word-index parity check decides,
        regardless of field name: its date fields ("31 октября 1998", "26
        СЕНТЯБРЯ 1997 ГОДА") are printed as Russian words interleaved with
        digits, so odd-indexed words (the month name) go to the Cyrillic engine
        even though the field itself is date-routed. Only a Latin engine_by_year
        rule beats the parity.
        """
        by_year = self._engine_by_year(field_name)
        if by_year == 'lat':
            return ['lat'] * n_words
        if by_year == 'cyr' or field_name in self.ocr_options.ru_fields:
            return ['cyr'] * n_words
        rest = 'lat' if ('date' in field_name.lower()
                         or field_name in self.ocr_options.en_fields) else None
        if doc_type == 'SNILS':
            return ['cyr' if i % 2 == 1 else rest for i in range(n_words)]
        return [rest] * n_words

    def _ocr_serial(self, words_dict: dict, doc_type: str):
        """Per-word OCR calls (one predict() per patch). Used on CPU."""
        def read(kind, patch):
            engine = self.ocr_cyr if kind == 'cyr' else self.ocr_lat
            return engine.predict(patch)[engine.model_name]['ocr_output']
        self._ocr_fields(words_dict, doc_type, read)

    def _ocr_batched(self, words_dict: dict, doc_type: str):
        """Same field/word routing and fix_errors as _ocr_serial, but the OCR
        calls are batched: every Cyrillic-routed word and every Latin-routed
        word across ALL fields is collected first, each engine is called ONCE
        via predict_batch, then results are redistributed. Only called when
        device=='gpu' (guaranteed by _ocr). Not bit-exact against serial by
        design - see ocr_batch.py.
        """
        patches = {'cyr': [], 'lat': []}
        for field_name, words in words_dict.items():
            kinds = self._word_engines(field_name, doc_type, len(words['patches']))
            for kind, patch in zip(kinds, words['patches']):
                if kind is not None:
                    patches[kind].append(patch)
        # The texts come back in the order the patches went in, and
        # _ocr_fields walks the words in that same order.
        texts = {'cyr': iter(self.ocr_cyr.predict_batch(patches['cyr'])),
                 'lat': iter(self.ocr_lat.predict_batch(patches['lat']))}
        self._ocr_fields(words_dict, doc_type, lambda kind, _patch: next(texts[kind]))

    def _ocr_fields(self, words_dict: dict, doc_type: str, read):
        """Read every routed word with read(kind, patch) -> raw text, clean it up
        and join each field. The two OCR paths differ only in read()."""
        ocr_dict = {}
        for field_name, words in words_dict.items():
            ocred_words = []
            kinds = self._word_engines(field_name, doc_type, len(words['patches']))
            for i, (kind, patch) in enumerate(zip(kinds, words['patches'])):
                if kind is None:
                    continue
                engine = self.ocr_cyr if kind == 'cyr' else self.ocr_lat
                # text normalization: Sex_ru -> М/Ж, strip stray leading dots on
                # names, dates to dd.mm.yyyy. CER-validated win.
                result = engine.fix_errors(field_type=field_name, text=read(kind, patch))
                if kind == 'lat' and field_name == 'MRZ':
                    # On the batched path the retry reads one crop at a time,
                    # outside the batch: it runs only on a line whose length is
                    # already wrong, so the batching it breaks had failed anyway.
                    # i is the index among the field's patches, not ocred_words.
                    result = self._read_mrz(i, result)
                words['ocr'].append(result)
                ocred_words.append(result)

            ocred_words = self._glue_torn(field_name, ocred_words)
            self._join_field(ocr_dict, field_name, doc_type, ocred_words)
            # Per-field rather than per-word, under the same name on both paths,
            # so a dump is comparable regardless of which path produced it.
            self._emit(f'ocr.{field_name}.words', ocred_words)

        self._fix_fms(ocr_dict, doc_type)
        self._emit('join', ocr_dict)
        # merge, never assign: _address_lines runs earlier and may already have
        # written OCR['Address'] into the same dict (INTPASSPORTADDR)
        self.results._meta_results.setdefault('OCR', {}).update(ocr_dict)

    @staticmethod
    def _join_field(ocr_dict: dict, field_name: str, doc_type: str, ocred_words: list):
        """Join per-word OCR results into the field's final string (digit dates
        use '.', worded dates use ' ', MRZ uses a newline, everything else ' ')."""
        # The MRZ arrives as one detection per line, top to bottom. The line
        # boundary is load-bearing - every check digit lives at a fixed offset
        # in line 2 - so the lines are joined with a newline and nothing else
        # is done to the text (a space would be outside the MRZ alphabet).
        if field_name == 'MRZ':
            ocr_dict[field_name] = '\n'.join(w for w in ocred_words if w)
            return
        if 'date' in field_name.lower():
            # The separator follows the CONTENT of the date, not the doc type:
            # a digit date (DD.MM.YYYY) is written with dots, a date spelled out
            # in words is not. SNILS is worded by definition ('26 СЕНТЯБРЯ 1997
            # ГОДА') and stays hard-coded; birth certificates need both - the
            # 1998 form has a digit Birth_date but a worded Issue_date, and the
            # whole 2018 form is worded ('15 ОКТЯБРЯ 2020 Г.', the parents'
            # birth dates). The 1998 form hid this: its ruler dots merged with
            # the join dot into a run that _clean_ruler_artifacts wiped out; on
            # the cleaner 2018 crops the lone join dot survived, giving
            # '30.ОКТЯБРЯ.2020'. Multi-word dates are the only ones affected -
            # a digit date reaches this point as a single word ('22.06.2010').
            worded = doc_type == 'SNILS' or any(c.isalpha()
                                                for w in ocred_words for c in w)
            ocr_dict[field_name] = (' ' if worded else '.').join(ocred_words)
        else:
            if ocr_dict.get(field_name):
                ocr_dict[field_name] += ' ' + ' '.join(ocred_words)
            else:
                ocr_dict[field_name] = ' '.join(ocred_words)
        ocr_dict[field_name] = ocr_dict[field_name].replace('  ', ' ').strip()

    def _glue_torn(self, field_name: str, ocred_words: list) -> list:
        """Glue the words a line break tore apart (sts_marks.glue_torn_words),
        for the fields the options name. The words come flat; the line lengths
        recorded by _split_words cut them back into lines."""
        if field_name not in (getattr(self.ocr_options, 'glue_torn', None) or []):
            return ocred_words
        counts = (getattr(self, '_field_lines', None) or {}).get(field_name)
        if not counts or sum(counts) != len(ocred_words):
            return ocred_words
        lines, start = [], 0
        for n in counts:
            lines.append(ocred_words[start:start + n])
            start += n
        return glue_torn_words(lines)

    #: The parts of sts_marks.parse_leasing that reach results.leasing. Only the
    #: flag, by measurement (STS word-break synthetic, 240 shots, detector v10,
    #: 2026-10-07): found 54/64, false 0/176 - while the lessor was right in
    #: 7/58, the contract number in 3/56, its date in 2/46. The parser finds
    #: them where the reading is clean; the reading of the small special-marks
    #: print is not, and a wrong value is worse than none for an integrator.
    #: Widen this when the reading improves - the parser needs no change.
    LEASING_REPORTED = ('leasing',)

    def _read_leasing(self, doc_type: str):
        """The leasing flag from the STS special marks, alongside the reading
        (results.leasing); the special marks themselves stay as read."""
        if not doc_type.upper().startswith('STS'):
            return
        ocr = self.results._meta_results.get('OCR') or {}
        leasing = parse_leasing(ocr.get('Special_marks'))
        if leasing:
            self.results._meta_results['Leasing'] = {
                k: v for k, v in leasing.items() if k in self.LEASING_REPORTED}
        # Every STS back - the side with the special marks - emits the stage, null
        # when there is no leasing: a port that misses a leasing record must differ
        # from the reference, not be skipped. The front has no marks to read.
        if doc_type.upper().startswith('STSBACK'):
            self._emit('leasing', self.results._meta_results.get('Leasing'))

    @staticmethod
    def _fix_fms(ocr_dict: dict, doc_type: str):
        """Disabled: the OCR reading of the authority code/name is returned as is.

        This used to rewrite ``Issue_organisation_code``/``Issue_organization_ru``
        from the FMS dictionary. Two reasons it is off:

        * **Cost.** When the code is read exactly the lookup is O(1) and ~1 ms,
          but a single misread character falls through to a ``difflib`` scan of
          the whole ~16k-entry dictionary — one ``ratio()`` per entry per query
          word, measured at **3.3-5.1 s for one document** (sample
          ``INTPASSPORT_1997/15_CR_INTPASSPORT_2001.jpg``, code read as
          ``123-005``). That is the entire reason that sample took 3.8 s while
          every other one takes ~0.4 s.
        * **Soundness.** On that fall-through the dictionary does not correct the
          code — it *replaces* it with the code of whichever authority name
          scored highest, so a misread digit can silently become a confident,
          well-formed, wrong code. Reporting what OCR actually read is the
          honest answer.

        Kept as a no-op rather than deleted so the reasoning stays next to the
        decision, and so the language ports have a named counterpart to mirror.
        The dictionary itself is not part of this repository.
        """
        return
