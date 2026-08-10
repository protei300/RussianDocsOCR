package pipeline

import (
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// Payload shaping for the stage sink.
//
// These live beside the pipeline rather than in the CLI because the CLI is no longer the only
// caller: the wire shape of a stage is part of the conformance contract, and a second copy in
// a command would be free to drift from the one the goldens were made with.

// SegmentsPayload renders the selected border contours as plain [[x, y], ...] lists.
// Mirrors pipeline.py::_segments_payload. nil (JSON null) when nothing was detected, which
// is what the reference emits too.
func SegmentsPayload(segments [][]imaging.Point) any {
	if len(segments) == 0 {
		return nil
	}
	out := make([][][2]float64, 0, len(segments))
	for _, contour := range segments {
		pts := make([][2]float64, 0, len(contour))
		for _, p := range contour {
			pts = append(pts, [2]float64{p.X, p.Y})
		}
		out = append(out, pts)
	}
	return out
}

// BoxesPayload renders detections in the reference's row layout,
// [x1, y1, x2, y2, conf, cls, label], as a list of heterogeneous arrays.
//
// The first four and the class are INTEGERS in the JSON, not floats: the reference coerces
// them with int() before attaching the label, and the checker compares discrete values
// exactly — 402 and 402.0 are the same number but not the same JSON.
func BoxesPayload(boxes []postprocess.Box) []any {
	out := make([]any, 0, len(boxes))
	for _, b := range boxes {
		out = append(out, []any{
			int(b.X1), int(b.Y1), int(b.X2), int(b.Y2),
			b.Conf, b.Cls, b.Label,
		})
	}
	return out
}

// WordBoxesPayload renders one field's word boxes, one entry per DETECTION of that field.
//
// A nil entry stays JSON null and means "this field needs no splitting, so its whole patch is
// the single word" — a different claim from "the detector found exactly one word", and a port
// that split a field it should not have would otherwise look like agreement.
func WordBoxesPayload(wordBoxes [][]postprocess.Box) []any {
	out := make([]any, 0, len(wordBoxes))
	for _, boxes := range wordBoxes {
		if boxes == nil {
			out = append(out, nil)
			continue
		}
		out = append(out, BoxesPayload(boxes))
	}
	return out
}
