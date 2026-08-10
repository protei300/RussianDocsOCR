package viewmodel

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// buildBoxes renders the detections, deciding which box owns each label's text.
// Port of transform.py::_build_boxes.
//
// Several boxes can share one label: split fields such as Birth_place_ru, and the doubled
// Licence_number on internal passports where the pipeline deduplicates the FIELD, not the
// boxes. The correspondence cannot be recovered from the library's output, so the text is
// attached to the highest-confidence box only and the rest are flagged `ambiguous` — a
// client greys them out instead of repeating the same string under every box.
func buildBoxes(raw []postprocess.Box, ocr map[string]string) []Box {
	if len(raw) == 0 {
		// An empty ARRAY, not null: the key is non-nullable in the contract.
		return []Box{}
	}

	// Which box owns the text for each label. Strict `>`, so on equal confidence the
	// EARLIER box wins -- matching the reference's `prev is None or conf > prev_conf`.
	bestByLabel := make(map[string]int, len(raw))
	for i, b := range raw {
		prev, seen := bestByLabel[label(b)]
		if !seen || b.Conf > raw[prev].Conf {
			bestByLabel[label(b)] = i
		}
	}

	out := make([]Box, 0, len(raw))
	for i, b := range raw {
		lbl := label(b)
		ownsText := bestByLabel[lbl] == i
		_, inOcr := ocr[lbl]

		box := Box{
			ID:      fmt.Sprintf("b%d", i),
			Label:   lbl,
			Display: FieldDisplay(lbl),
			// Face and Signature are detected but never OCR'd, so they carry no text.
			Kind: kindOf(lbl),
			// int(), truncating -- the coordinates are already whole numbers by here.
			X1:   intp(int(b.X1)),
			Y1:   intp(int(b.Y1)),
			X2:   intp(int(b.X2)),
			Y2:   intp(int(b.Y2)),
			Conf: num(b.Conf),
			Cls:  intp(b.Cls),
			// nil, NOT the empty string: a box with no text and a box whose text is ""
			// are different claims.
			Text:      nil,
			Ambiguous: inOcr && !ownsText,
		}
		if ownsText {
			if v, ok := ocr[lbl]; ok {
				box.Text = str(v)
			}
		}
		out = append(out, box)
	}
	return out
}

func label(b postprocess.Box) string { return b.Label }

func kindOf(lbl string) string {
	if nonTextLabels[lbl] {
		return "visual"
	}
	return "text"
}
