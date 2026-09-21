package pipeline

import (
	"math"
	"sort"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// WordsMaxGap is WORDS_MAX_GAP (pipeline.py:759-786): an empty stretch on a line wider
// than this many typical words means the split dropped a word, and the line is read
// whole instead. Measured over 157 documents, 815 fields: the widest gap on a line that
// really lost a long word has a median of 2.74 typical word widths, against 0.06 on
// intact lines - a factor of forty. The share of the line covered by word boxes was
// measured on the same run and rejected: it reads print DENSITY, not the integrity of
// the split (a correctly read internal-passport Licence_number sits at 0.64-0.75).
// Known limit: when the missing word left NO gap because a neighbour's box swallowed it,
// geometry cannot see it at all - 8 of 19 measured cases.
const WordsMaxGap = 3.0

// LineMinInk is LINE_MIN_INK (pipeline.py:788-826): below this much fine-grained detail
// (variance of the Laplacian) a line carries no strokes and the fallback above must NOT
// re-read it. "No word boxes" has a second cause besides a lost split: there IS no text
// (an anonymised, blurred name strip), and re-reading it whole manufactured «ЛВН». From a
// blur ladder over 612 lines: sharp text median 1590, mild blur 249, blur wider than the
// stroke 27; 100 sits in that trough. Known limit: an empty strip and a strip with VERY
// FAINT text look the same to this measure.
const LineMinInk = 100.0

// LineFlag names one line of one field the gap guard acted on (or declined to).
// Mirrors the entries of PipelineResults.words_fallback / words_no_ink.
type LineFlag struct {
	Field string
	// Line is the ordinal of the line WITHIN its field, top to bottom.
	Line int
	// Gap is the widest empty stretch in typical word widths; NaN stands for the
	// reference's None (no words found at all - the ratio has no denominator).
	Gap float64
	// Ink is the Laplacian variance, set on no-ink entries only.
	Ink float64
}

// SplitFlags is what the gap guard reports: part of the contract, not debug output. A
// measurement that corrects for the missing spaces of a line read whole must apply that
// correction ONLY to these lines (pipeline.py:184-211).
type SplitFlags struct {
	Fallback []LineFlag
	NoInk    []LineFlag
}

// FieldWords is one OCR field: every word patch that will be read for it, in order.
//
// Detections merge by LABEL, and a label can be detected more than once, so WordBoxes is
// a list PER DETECTION while Patches is the flat concatenation the OCR loop walks. A nil
// entry in WordBoxes means that detection needed no splitting, so its whole patch is the
// single word — a different thing from a detector that found exactly one word.
type FieldWords struct {
	Label     string
	Patches   []imaging.Image
	WordBoxes [][]postprocess.Box
}

// SplitWords turns detected fields into per-field word patches.
// Port of Pipeline._split_words (pipeline.py:763-830).
//
// Returns fields in FIRST-DETECTION order. That order is the output contract, not a
// detail: it decides the order of `join`, of the view model's `fields` array and of the
// search text the service builds. Go randomises map iteration, so the order is carried
// in a slice and never recovered from a map (CONVENTIONS §1).
//
// Ownership: the returned patches are freshly cropped and owned by the caller, EXCEPT
// the whole-patch fallback, which borrows the field's own patch. Close via
// FieldWordsClose before closing the fields themselves.
//
// docType is the BARE type ('SNILS', not 'SNILS_1996'): the gap guard compares it with
// == 'SNILS', exactly as the OCR routing does.
func SplitWords(fields []modules.Field, opts OcrOptions,
	words *modules.WordsDetector, docType string) ([]FieldWords, SplitFlags, error) {

	var flags SplitFlags
	drop := duplicateFieldIndices(fields)

	// Fields that will actually contribute. Everything else is detected but never
	// read — Face and Signature are the obvious cases. Multi-line fields are labeled
	// one detection per line, and the per-label assembly below concatenates their
	// words in list order - so the kept boxes are ordered top-to-bottom by their
	// vertical centre (a STABLE sort on a pure y key, as the reference's list.sort;
	// pipeline.py:1538-1541).
	var kept []int
	for i := range fields {
		if drop[i] {
			continue
		}
		if opts.IsOcrField(fields[i].Box.Label) {
			kept = append(kept, i)
		}
	}
	sort.SliceStable(kept, func(a, b int) bool {
		ca := (fields[kept[a]].Box.Y1 + fields[kept[a]].Box.Y2) / 2
		cb := (fields[kept[b]].Box.Y1 + fields[kept[b]].Box.Y2) / 2
		return ca < cb
	})

	// Word-detector calls are independent (a different crop each, one reused session)
	// so they are dispatched as a group. Fields that need no splitting need no call.
	var splitIdxs []int
	for _, i := range kept {
		if opts.NeedsSplit(fields[i].Box.Label) {
			splitIdxs = append(splitIdxs, i)
		}
	}

	type split struct {
		boxes   []postprocess.Box
		patches []imaging.Image
	}
	byIdx := make(map[int]split, len(splitIdxs))
	if len(splitIdxs) > 0 {
		tasks := make([]func() (split, error), len(splitIdxs))
		for k, i := range splitIdxs {
			i := i
			tasks[k] = func() (split, error) {
				b, p, err := words.PredictTransform(fields[i].Patch)
				return split{boxes: b, patches: p}, err
			}
		}
		results, err := RunGroup(MinLimit(8, len(splitIdxs)), tasks)
		if err != nil {
			// The crops of the tasks that DID succeed are already allocated, and nothing
			// downstream will ever see them — releasing them here is the only chance. This is
			// why RunGroup returns partial results on error.
			for k := range results {
				for j := range results[k].patches {
					_ = results[k].patches[j].Close()
				}
			}
			return nil, flags, err
		}
		for k, i := range splitIdxs {
			byIdx[i] = results[k]
		}
	}

	// Gap guard (pipeline.py:1565-1610). A hole on the line wider than a few typical
	// words means the split dropped a word - measured twice: a 9 px crop where the
	// detector returned NO words (the field vanished without a trace), and a line whose
	// longest word («Тракторозаводский», 17 characters) was the one missed. Reading the
	// line whole recovers it; the price is that the engine emits no spaces, so the line
	// comes back glued, which is why this is a fallback on a signal and not the default.
	//
	// SNILS is excluded BY CONSTRUCTION, not by hoping the threshold spares it: there
	// the engine is chosen by word-index parity (see OcrFields), and a line read whole
	// destroys the parity the routing depends on.
	if docType != "SNILS" {
		// `kept` order is top-to-bottom, and a multi-line field collects its lines in
		// that same order below - so counting per label here gives the line's ordinal
		// WITHIN its field, which is what a reader of the flag can act on.
		seen := map[string]int{}
		for _, i := range kept {
			label := fields[i].Box.Label
			ordinal := seen[label]
			seen[label] = ordinal + 1
			s, split := byIdx[i]
			if !split {
				continue
			}
			gap := widestGap(s.boxes, float64(fields[i].Patch.Width()))
			if gap > WordsMaxGap && len(s.boxes) == 0 {
				// No boxes at all has two causes, and only one of them is a lost
				// split: the other is a line with nothing on it. Asked HERE only -
				// where boxes were found the text is there by definition.
				ink := imaging.LaplacianVariance(fields[i].Patch)
				if ink < LineMinInk {
					flags.NoInk = append(flags.NoInk, LineFlag{Field: label, Line: ordinal,
						Ink: tensor.RoundHalfEven(ink, 2)})
					continue
				}
			}
			if gap > WordsMaxGap {
				// The whole patch becomes the single word; the detected boxes (an
				// empty or a holed list) stay as the stage payload, exactly as the
				// reference leaves word_bbox_by_idx untouched.
				for j := range s.patches {
					_ = s.patches[j].Close()
				}
				s.patches = []imaging.Image{fields[i].Patch.Clone()}
				byIdx[i] = s
				g := gap
				if !math.IsInf(gap, 1) {
					g = tensor.RoundHalfEven(gap, 3)
				} else {
					g = math.NaN()
				}
				flags.Fallback = append(flags.Fallback, LineFlag{Field: label, Line: ordinal, Gap: g})
			}
		}
	}

	var out []FieldWords
	pos := map[string]int{}
	for _, i := range kept {
		label := fields[i].Box.Label

		var patches []imaging.Image
		var boxes []postprocess.Box
		if s, ok := byIdx[i]; ok {
			patches, boxes = s.patches, s.boxes
			// An empty detection still yields an empty word list here, exactly as the
			// reference does — it does NOT fall back to the whole patch. The fallback
			// belongs to fields that were never split at all.
		} else {
			// CLONED, not borrowed. The reference aliases the field's own patch here and
			// Python's GC makes that free; in a port, a borrowed Mat in a list the caller
			// closes is a double free that surfaces only in bulk. One copy per unsplit
			// field buys uniform ownership, which is worth far more than the copy.
			patches = []imaging.Image{fields[i].Patch.Clone()}
			boxes = nil
		}

		if j, seen := pos[label]; seen {
			out[j].Patches = append(out[j].Patches, patches...)
			out[j].WordBoxes = append(out[j].WordBoxes, boxes)
			continue
		}
		pos[label] = len(out)
		out = append(out, FieldWords{Label: label, Patches: patches,
			WordBoxes: [][]postprocess.Box{boxes}})
	}
	return out, flags, nil
}

// widestGap is the widest empty stretch on the line, in typical word widths.
// Port of Pipeline._widest_gap (pipeline.py:944-977).
//
// A dropped word leaves a hole about as wide as a word; evenly spaced printing does
// not, however wide the spacing. That is the whole reason this is a ratio to the line's
// OWN median word width instead of a share of the line. Edges count as gaps too - a
// word lost from the start or the end of a line leaves the hole at the border. Nothing
// found at all means the whole line is one hole, so the answer is +Inf.
func widestGap(boxes []postprocess.Box, lineWidth float64) float64 {
	if len(boxes) == 0 {
		return math.Inf(1)
	}
	if lineWidth == 0 {
		return 0
	}
	type span struct{ a, b float64 }
	spans := make([]span, len(boxes))
	for i, b := range boxes {
		spans[i] = span{b.X1, b.X2}
	}
	// sorted() on (x1, x2) tuples: lexicographic, stable.
	sort.SliceStable(spans, func(i, j int) bool {
		if spans[i].a != spans[j].a {
			return spans[i].a < spans[j].a
		}
		return spans[i].b < spans[j].b
	})
	var widths []float64
	for _, s := range spans {
		if s.b > s.a {
			widths = append(widths, s.b-s.a)
		}
	}
	if len(widths) == 0 {
		return math.Inf(1)
	}
	sort.Float64s(widths)
	typical := widths[len(widths)/2]
	if typical <= 0 {
		return math.Inf(1)
	}
	gaps := []float64{spans[0].a} // empty stretch on the left
	end := spans[0].b
	for _, s := range spans[1:] {
		gaps = append(gaps, math.Max(0, s.a-end))
		end = math.Max(end, s.b)
	}
	gaps = append(gaps, math.Max(0, lineWidth-end)) // and on the right
	best := gaps[0]
	for _, g := range gaps[1:] {
		if g > best {
			best = g
		}
	}
	return best / typical
}

// duplicateFieldIndices marks all but the highest-confidence detection of each field
// that must be unique.
//
// The internal passport prints its series and number — and the FMS code — twice, so the
// detector legitimately returns duplicate boxes and OCR'ing both would read the same
// value twice.
//
// The tie-break matters: Python's `max(idxs, key=...)` returns the FIRST maximum, so on
// equal confidence the EARLIER detection survives. Reproduced with a strict `>`.
func duplicateFieldIndices(fields []modules.Field) map[int]bool {
	uniqueFields := []string{"Licence_number", "Issue_organisation_code"}

	drop := map[int]bool{}
	for _, field := range uniqueFields {
		var idxs []int
		for i := range fields {
			if fields[i].Box.Label == field {
				idxs = append(idxs, i)
			}
		}
		if len(idxs) <= 1 {
			continue
		}
		best := idxs[0]
		for _, i := range idxs[1:] {
			if fields[i].Box.Conf > fields[best].Box.Conf {
				best = i
			}
		}
		for _, i := range idxs {
			if i != best {
				drop[i] = true
			}
		}
	}
	return drop
}

// FieldWordsClose releases every word crop.
//
// Unconditional, because SplitWords owns all of them — the unsplit fallback is cloned
// precisely so this function needs no special case and no aliasing analysis.
func FieldWordsClose(fw []FieldWords) {
	for i := range fw {
		for j := range fw[i].Patches {
			_ = fw[i].Patches[j].Close()
		}
	}
}
