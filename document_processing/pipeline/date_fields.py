"""Date fields: a date re-read whole, and dates brought to one form."""
from .dates import canonical_date, canonical_dates


class DateFieldsStage:
    """Pipeline methods: date fields: a date re-read whole, and dates brought to one form.

    A part of Pipeline, not usable on its own: the methods read the
    models and the results that Pipeline.__init__ and process_img set up.
    """

    def _reread_dates_whole(self):
        """Re-read a date field line by line WHOLE when the word-by-word reading
        is not a date and the whole reading is.

        The word splitter can drop a word without leaving a hole the gap guard
        sees: on a real 1998 birth certificate (2026-09-27) the issue
        date «ДД» МЕСЯЦА ГГГГ г. came out as month and year only - the day, pressed
        against the left edge of the field crop, was not taken for a word at all
        (padding the crop does not change that), and the empty stretch it left
        was 2.6 typical words wide against the guard's measured 3.0. Read whole,
        the same crop gives the day glued to the month and year, which converts.

        The rule checks itself: it replaces a reading only when that reading
        does NOT convert to dd.mm.yyyy and the whole-line one DOES, so a date
        that already converts is never touched. The price is the one the gap
        guard pays too - a line read whole comes back without spaces; the
        canonical view is unaffected. Every replacement is recorded in
        meta_results['DatesReadWhole'].
        """
        lines_by_field = getattr(self, '_date_lines', None) or {}
        ocr = self.results._meta_results.get('OCR')
        if not lines_by_field or not ocr:
            return
        done = []
        for field_name, lines in lines_by_field.items():
            split_reading = ocr.get(field_name, '')
            if canonical_date(field_name, split_reading) is not None:
                continue
            if field_name in self.ocr_options.ru_fields:
                engine = self.ocr_cyr
            else:
                engine = self.ocr_lat
            reads = [engine.fix_errors(field_type=field_name,
                                       text=engine.predict(line)[engine.model_name]['ocr_output'])
                     for line in lines]
            whole = ' '.join(r for r in reads if r).strip()
            if canonical_date(field_name, whole) is None:
                continue
            ocr[field_name] = whole
            done.append({'field': field_name, 'split': split_reading, 'whole': whole})
        if done:
            self.results._meta_results['DatesReadWhole'] = done

    def _normalize_dates(self):
        """Build the canonical ``dd.mm.yyyy`` view next to the reading.

        Runs once, after OCR, on the finished ``results.ocr`` - that is the whole
        point of the placement: the conversion happens when the RESULT is formed,
        not while decoding or checking OCR, so nothing upstream sees a rewritten
        value. Date fields are recognised by name, the same convention
        ``_join_field`` already uses.
        """
        ocr = self.results._meta_results.get('OCR')
        if not ocr:
            return
        fields = [name for name in ocr if 'date' in name.lower()]
        normalized = canonical_dates(ocr, fields)
        if normalized:
            self.results._meta_results['OCR_normalized'] = normalized
