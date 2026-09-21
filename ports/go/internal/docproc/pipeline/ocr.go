package pipeline

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// FieldText is one recognised field: the per-word strings and the joined value.
//
// A SLICE, not a map, because the field order is part of the output contract — it decides
// the order of the view model's `fields` array and of the service's search text. Go
// randomises map iteration, so an order that lives only in a map is not an order
// (CONVENTIONS §1).
// rulerRuns matches the dotted ruler lines the 1998 birth-certificate form prints
// under every value; they land inside the field crops and OCR emits runs of
// dots/dashes/underscores around the real words.
// The marks the ruler dots come back as. Commas and quotes are in the set
// because that is what the engine actually emits on this form («28., ИЮЛЯ 2010»,
// «08.,.АВГУСТА.,.2008», «"""СЕМ","" ПОННИЛОВИЧ»), not because they were
// expected. Mirrors Pipeline._RULER_MARKS.
var rulerRuns = regexp.MustCompile(`[.,_\-"]{2,}`)

// rulerLone is the same set, for the token filter below.
var rulerLone = map[string]bool{".": true, ",": true, "_": true, "-": true, `"`: true}

// CleanRulerArtifacts collapses ruler-dot runs out of a joined field value.
// Port of Pipeline._clean_ruler_artifacts (pipeline.py:1061-1076). The reference
// also drops LONE separators with a lookaround pattern Go's RE2 cannot express
// ((?:^|(?<=\s))[._\-](?=\s|$)); splitting into whitespace tokens, dropping
// tokens that are exactly one separator rune and re-joining is equivalent,
// because the reference finishes by collapsing all whitespace and trimming.
// Single in-word dots - the digit birth date, abbreviations - stay untouched,
// exactly as in the reference.
func CleanRulerArtifacts(value string) string {
	t := rulerRuns.ReplaceAllString(value, " ")
	fields := strings.Fields(t)
	kept := fields[:0]
	for _, tok := range fields {
		if rulerLone[tok] {
			continue
		}
		kept = append(kept, tok)
	}
	return strings.Join(kept, " ")
}

type FieldText struct {
	Label string
	Words []string
	Value string
}

// MrzLineLen is MRZ_LINE_LEN: a line of a machine-readable zone is exactly 44 characters,
// and that is a rare luxury - the pipeline can tell that it read the line WRONG without
// being told, and try again. Anything shorter means the crop lost part of the line.
const MrzLineLen = 44

// mrzRetryGrowth is MRZ_RETRY_GROWTH (pipeline.py:834-853). The MRZ is printed as ONE
// rectangle holding two lines, so both lines share the same horizontal span - but the
// detector does not know that. Measured over samples/: the two boxes of a zone start
// within 10 px of each other when the zone reads correctly and 90-182 px apart when it
// does not, and the characters outside the narrower box never reach the engine (23 of
// the 34 damaged lines; the engine was innocent). So the zone's own span is the FIRST
// retry candidate, then the ladder widens further - a candidate and not a rewrite,
// because forcing every MRZ box to the union span fixed the external passports and
// damaged an internal one. The ladder reaches 34% of the span on each side because that
// is what the worst measured case needed; the crop is clamped to the canvas, so the last
// steps saturate instead of running away.
var mrzRetryGrowth = []float64{0.0, 0.05, 0.10, 0.16, 0.24, 0.34}

// mrzAlphabet is MRZ_ALPHABET: capitals, digits and the filler. A line cannot begin or
// end with anything else, so a stray '.' or '_' at an edge is the page border caught by
// the crop, not text.
const mrzAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

// MrzZone remembers the canvas and the MRZ boxes for the length self-check.
// Port of Pipeline._mrz_zone as filled by _note_mrz_zone (pipeline.py:862-890).
//
// Nothing is modified when it is built: the boxes the detector produced stay exactly as
// they are, so fields.bbox and every other field are untouched. Its only purpose is that
// ReadMrz can re-cut a line from the canvas later. The canvas is BORROWED (owned by the
// run), never closed here.
type MrzZone struct {
	Canvas imaging.Image
	// Boxes are the MRZ boxes [x1, y1, x2, y2], top to bottom - the order the OCR loop
	// walks the patches in, so the retry knows which box a patch came from.
	Boxes [][4]int
	// Span is the horizontal extent of the zone as a whole, from the line-shaped boxes
	// only; nil when fewer than two are line-shaped.
	Span *[2]int
}

// NoteMrzZone builds the zone from the field detections, or nil when there is no MRZ.
func NoteMrzZone(boxes []postprocess.Box, canvas imaging.Image) *MrzZone {
	var idx []int
	for i, b := range boxes {
		if b.Label == "MRZ" {
			idx = append(idx, i)
		}
	}
	if len(idx) == 0 {
		return nil
	}
	// list.sort by vertical centre: stable.
	sort.SliceStable(idx, func(a, b int) bool {
		return (boxes[idx[a]].Y1+boxes[idx[a]].Y2)/2 < (boxes[idx[b]].Y1+boxes[idx[b]].Y2)/2
	})
	zone := &MrzZone{Canvas: canvas}
	for _, i := range idx {
		b := boxes[i]
		zone.Boxes = append(zone.Boxes, [4]int{int(b.X1), int(b.Y1), int(b.X2), int(b.Y2)})
	}
	// The span of the zone as a whole: for a line whose own box is too narrow this is
	// where the missing characters are. Only from boxes that are line-shaped - a box
	// several line-heights tall is not a line, and its edges say nothing about where the
	// line ends (measured: the one such zone reads worse from a widened crop).
	var lineShaped [][4]int
	for _, b := range zone.Boxes {
		h := b[3] - b[1]
		if h < 1 {
			h = 1
		}
		if b[2]-b[0] >= 10*h {
			lineShaped = append(lineShaped, b)
		}
	}
	if len(lineShaped) > 1 {
		left, right := lineShaped[0][0], lineShaped[0][2]
		for _, b := range lineShaped[1:] {
			if b[0] < left {
				left = b[0]
			}
			if b[2] > right {
				right = b[2]
			}
		}
		zone.Span = &[2]int{left, right}
	}
	return zone
}

// trimToMrzAlphabet drops edge characters that cannot occur in a machine-readable zone.
// Port of Pipeline._trim_to_mrz_alphabet: only the ends, and only characters outside the
// zone's closed alphabet. Anything inside the line is left alone - a wrong character
// there is a reading error, and hiding it would be worse than showing it.
func trimToMrzAlphabet(text string) string {
	return strings.TrimFunc(text, func(r rune) bool {
		return !strings.ContainsRune(mrzAlphabet, r)
	})
}

// readMrz re-reads one MRZ line from a wider crop when it came out too short.
// Port of Pipeline._read_mrz (pipeline.py:905-942).
//
// The reading the detector's own crop produced is kept unless it is the wrong length;
// then the zone's full span is tried, then progressively wider crops, and the first
// result of exactly 44 characters wins. Falls back to the longest reading seen. Never
// invents a line: it only re-reads a box the detector found.
func readMrz(zone *MrzZone, lat *modules.OcrEngine, lineIndex int, text string) (string, error) {
	text = trimToMrzAlphabet(text)
	if utf8.RuneCountInString(text) == MrzLineLen {
		return text, nil
	}
	if zone == nil || lineIndex >= len(zone.Boxes) {
		return text, nil
	}
	width := zone.Canvas.Width()
	b := zone.Boxes[lineIndex]
	x1, y1, x2, y2 := b[0], b[1], b[2], b[3]
	best := text
	for _, growth := range mrzRetryGrowth {
		left, right := x1, x2
		if zone.Span != nil {
			left, right = zone.Span[0], zone.Span[1]
		}
		// int(round(...)): Python's round is half-to-even.
		step := int(math.RoundToEven(float64(right-left) * growth))
		cx1, cx2 := left-step, right+step
		if cx1 < 0 {
			cx1 = 0
		}
		if cx2 > width {
			cx2 = width
		}
		crop, err := imaging.ClampedCrop(zone.Canvas, cx1, y1, cx2, y2)
		if err != nil {
			return "", err
		}
		if crop.Empty() || crop.Width() == 0 || crop.Height() == 0 {
			_ = crop.Close()
			continue
		}
		candidate, err := lat.Predict(crop)
		_ = crop.Close()
		if err != nil {
			return "", err
		}
		candidate = trimToMrzAlphabet(lat.FixErrors("MRZ", candidate))
		if utf8.RuneCountInString(candidate) == MrzLineLen {
			return candidate, nil
		}
		if utf8.RuneCountInString(candidate) > utf8.RuneCountInString(best) {
			best = candidate
		}
	}
	return best, nil
}

// OcrFields recognises every field's words and joins them.
// Port of Pipeline._ocr_serial (pipeline.py:1830-1870).
//
// docType is the label with its year suffix already stripped, which matters: the routing
// below tests `docType == "SNILS"` against the bare type. zone may be nil (no MRZ on the
// document); with one, every MRZ line of the wrong length goes through readMrz.
func OcrFields(fields []FieldWords, docType string, opts OcrOptions,
	cyr, lat *modules.OcrEngine, zone *MrzZone) ([]FieldText, error) {

	out := make([]FieldText, 0, len(fields))
	for _, fw := range fields {
		var words []string
		for i, patch := range fw.Patches {
			// Three branches, and the FIRST one carries a precedence subtlety worth
			// spelling out: Python's `doc_type == 'SNILS' and i % 2 == 1 or field_name in
			// ru_fields` binds `and` tighter than `or`, so it reads
			// `(SNILS and odd) or (in ru_fields)`.
			//
			// SNILS is the one type where a WORD-INDEX PARITY check decides the engine
			// regardless of field name: its dates read "31 октября 1998", Russian month
			// names interleaved with digits, so odd-indexed words must go to the Cyrillic
			// engine even though the field itself is date-routed below.
			switch {
			case (docType == "SNILS" && i%2 == 1) || contains(opts.RuFields, fw.Label):
				text, err := cyr.Predict(patch)
				if err != nil {
					return nil, err
				}
				words = append(words, cyr.FixErrors(fw.Label, text))

			case strings.Contains(strings.ToLower(fw.Label), "date"):
				text, err := lat.Predict(patch)
				if err != nil {
					return nil, err
				}
				words = append(words, lat.FixErrors(fw.Label, text))

			case contains(opts.EnFields, fw.Label):
				text, err := lat.Predict(patch)
				if err != nil {
					return nil, err
				}
				text = lat.FixErrors(fw.Label, text)
				if fw.Label == "MRZ" {
					// i is the line's index within the field: the MRZ is never split, so
					// its patches are its detections, top to bottom, exactly as the zone
					// recorded them.
					if text, err = readMrz(zone, lat, i, text); err != nil {
						return nil, err
					}
				}
				words = append(words, text)

				// No default: a field in neither list contributes NO word, and the field
				// still appears with an empty value. That is the reference's behaviour --
				// the loop simply skips -- and it is reachable, because SplitWords admits a
				// field only if it is in one of the lists, but the parity branch above can
				// leave an even-indexed SNILS word unmatched.
			}
		}
		out = append(out, FieldText{Label: fw.Label, Words: words})
	}

	// Joining is separate from recognition so the per-word strings survive for the
	// ocr.<Field>.words stage, which is what localises a single bad word.
	joined := map[string]string{}
	for i := range out {
		out[i].Value = joinField(joined, out[i].Label, docType, out[i].Words)
	}
	return out, nil
}

// joinField assembles a field's final string.
// Port of Pipeline._join_field.
//
// The date separator follows the CONTENT, not the doc type: a digit date joins with '.'
// to give "01.02.1998", a date spelled out in words joins with spaces. SNILS is worded by
// definition ("31 октября 1998") and stays hard-coded; birth certificates need both -
// the 1998 blank has a digit Birth_date next to a worded Issue_date, and every date on
// the 2018 blank is worded ("15 ОКТЯБРЯ 2020 Г."). Only multi-word dates are affected:
// a digit date reaches this point as a single word ("22.06.2010").
//
// `joined` accumulates across calls because a field detected twice appends with a space
// rather than replacing -- the reference relies on the dict already holding a value.
func joinField(joined map[string]string, label, docType string, words []string) string {
	// The MRZ arrives as one detection per line, top to bottom. The line boundary is
	// load-bearing - every check digit lives at a fixed offset in line 2 - so the lines
	// are joined with a newline and nothing else is done to the text: a space would be
	// outside the MRZ alphabet, and the double-space squeeze below must not touch it.
	if label == "MRZ" {
		var lines []string
		for _, w := range words {
			if w != "" {
				lines = append(lines, w)
			}
		}
		value := strings.Join(lines, "\n")
		joined[label] = value
		return value
	}

	isDate := strings.Contains(strings.ToLower(label), "date")
	worded := docType == "SNILS"
	for _, w := range words {
		for _, r := range w {
			if unicode.IsLetter(r) {
				worded = true
			}
		}
	}

	var value string
	switch {
	case isDate && !worded:
		value = strings.Join(words, ".")
	case isDate:
		value = strings.Join(words, " ")
	default:
		if prev := joined[label]; prev != "" {
			value = prev + " " + strings.Join(words, " ")
		} else {
			value = strings.Join(words, " ")
		}
	}

	// A SINGLE pass of "  " -> " ", not a loop and not a regex: Python's str.replace is
	// one pass too, so three consecutive spaces leave one behind in both. Collapsing
	// fully here would produce a different string.
	value = strings.TrimSpace(strings.ReplaceAll(value, "  ", " "))
	joined[label] = value
	return value
}

// FixFms is a deliberate no-op, carried over rather than dropped.
//
// It used to rewrite Issue_organisation_code/Issue_organization_ru from the FMS
// dictionary. Disabled in the reference for two reasons worth keeping next to the code:
//
//   - COST. An exactly-read code is an O(1) lookup, but a single misread character falls
//     through to a difflib scan of the whole ~16k-entry dictionary — measured at 3.3-5.1 s
//     for ONE document. That is the entire reason INTPASSPORT_1997/15_CR_INTPASSPORT_2001
//     took 3.8 s while every other sample took ~0.4 s.
//   - SOUNDNESS. On that fall-through the dictionary does not correct the code, it
//     REPLACES it with the code of whichever authority name scored highest — so a misread
//     digit silently becomes a confident, well-formed, wrong code.
//
// Kept as a no-op so re-enabling is one line and the reasoning stays attached. The whole
// FMS dictionary is otherwise not ported (it has no live callers).
func FixFms(_ []FieldText, _ string) {}
