package pipeline

import (
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// DateReread records one date field whose word-by-word reading was replaced by the
// reading of its whole line (meta_results['DatesReadWhole'] entries: field, split, whole).
type DateReread struct {
	Field string
	// Split is the reading that did not convert to dd.mm.yyyy.
	Split string
	// Whole is the whole-line reading that replaced it.
	Whole string
}

// rereadDatesWhole re-reads a date field line by line WHOLE when the word-by-word reading
// is not a date and the whole reading is. Port of Pipeline._reread_dates_whole (commit
// 1af0d270; the "is it a date" test goes through CanonicalDate by field since 75a082b).
//
// The word splitter can drop a word without leaving a hole the gap guard sees: on a real
// 1998 birth certificate the issue date «ДД» МЕСЯЦА ГГГГ г. came out as month and year
// only - the day, pressed against the left edge of the field crop, was not taken for a word
// at all (padding the crop does not change that), and the empty stretch it left was 2.6
// typical words wide against the guard's measured 3.0. Read whole, the same crop gives the
// day glued to the month and year, which converts.
//
// The rule checks itself: it replaces a reading only when that reading does NOT convert to
// dd.mm.yyyy and the whole-line one DOES, so a date that already converts is never touched.
// The price is the one the gap guard pays too - a line read whole comes back without
// spaces; the canonical view is unaffected.
//
// It rewrites the FINAL dict `ocr` (the reference rewrites meta_results['OCR']): the
// `ocr.<Field>.words` and `join` stages are emitted before this runs and keep the split
// reading, exactly as in the reference. The field keeps its engine: ru_fields go to the
// Cyrillic one, everything else to the Latin one (the SNILS parity rule is irrelevant -
// SNILS never collects date lines).
func rereadDatesWhole(fields []FieldWords, ocr map[string]string, opts OcrOptions,
	cyr, lat lineReader) ([]DateReread, error) {

	if len(ocr) == 0 {
		return nil, nil
	}
	var done []DateReread
	for _, fw := range fields {
		if len(fw.DateLines) == 0 {
			continue
		}
		split := ocr[fw.Label]
		if CanonicalDate(fw.Label, split) != "" {
			continue
		}
		engine := lat
		if contains(opts.RuFields, fw.Label) {
			engine = cyr
		}
		var reads []string
		for _, line := range fw.DateLines {
			text, err := engine.Predict(line)
			if err != nil {
				return done, err
			}
			if fixed := engine.FixErrors(fw.Label, text); fixed != "" {
				reads = append(reads, fixed)
			}
		}
		whole := strings.TrimSpace(strings.Join(reads, " "))
		if CanonicalDate(fw.Label, whole) == "" {
			continue
		}
		ocr[fw.Label] = whole
		done = append(done, DateReread{Field: fw.Label, Split: split, Whole: whole})
	}
	return done, nil
}

// lineReader is the part of an OCR engine the re-read needs. *modules.OcrEngine satisfies
// it; the tests pass a stand-in that returns a fixed text (the reference's tests do the
// same with a stub engine).
type lineReader interface {
	Predict(patch imaging.Image) (string, error)
	FixErrors(fieldType, text string) string
}
