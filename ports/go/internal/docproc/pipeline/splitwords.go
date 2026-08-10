package pipeline

import (
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

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
func SplitWords(fields []modules.Field, opts OcrOptions,
	words *modules.WordsDetector) ([]FieldWords, error) {

	drop := duplicateFieldIndices(fields)

	// Fields that will actually contribute, in original order. Everything else is
	// detected but never read — Face and Signature are the obvious cases.
	var kept []int
	for i := range fields {
		if drop[i] {
			continue
		}
		if opts.IsOcrField(fields[i].Box.Label) {
			kept = append(kept, i)
		}
	}

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
			return nil, err
		}
		for k, i := range splitIdxs {
			byIdx[i] = results[k]
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
	return out, nil
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
