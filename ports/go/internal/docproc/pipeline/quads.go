package pipeline

import (
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
)

// Port of the way back to the input image: pipeline/fields.py::Pipeline._fields_detector's
// FieldFrames, pipeline/fields.py::Pipeline._field_quads and
// pipeline/results.py::PipelineResults.to_input / field_quads / word_quads
// (PR #19, issue #18; conformance stage `quads`).

// fieldFrame says where the patch cut for one detection lies on the canvas it was cut from:
// Map is the map from the patch AS CUT (before the series/number turn) to that canvas, W and H
// the patch's size then. The map is written the way geometry.go reads a chain - the stages that
// make the patch out of the canvas, in the order they ran.
type fieldFrame struct {
	Map  geometry.Geometry
	W, H int
}

// singleCanvasFrame is the frame of a detection on a single canvas: the canvas cut at its box.
func singleCanvasFrame(f modules.Field, turned bool) fieldFrame {
	w, h := f.Patch.Width(), f.Patch.Height()
	if turned { // the patch was turned a quarter after it was cut: it was h x w then
		w, h = h, w
	}
	return fieldFrame{Map: geometry.Offset{DX: -math.Trunc(f.Box.X1), DY: -math.Trunc(f.Box.Y1)}, W: w, H: h}
}

// Quad is a quadrilateral on the input image, corners following the canvas box from its
// top-left.
type Quad = geometry.Quad

// Quads is where the read fields and their word patches lie on the input image
// (PipelineResults.field_quads / word_quads).
//
// Fields and Words are nil together when the way back to the input image is not known for this
// run: a caller draws these over a photo, so a quadrilateral that silently landed elsewhere would
// be worse than no quadrilateral at all. Said once for the run, never per box.
type Quads struct {
	// Fields is label -> one quadrilateral per detection, in the order the fields were read (top
	// to bottom).
	Fields map[string][]Quad
	// Words is label -> the quadrilateral of each patch of the field's words, same order as the
	// words.
	Words map[string][]Quad
	// AddressLines are the registration-address lines, in the order of Results.Address.Lines.
	// HasAddressLines says the address path ran (the stage then carries the key); a nil list
	// with it set is null: the way back is not known, said once for the run, like Fields.
	AddressLines    []Quad
	HasAddressLines bool
}

// Known reports whether the way back was known for the run.
func (q *Quads) Known() bool { return q != nil && q.Fields != nil }

func toQuad(pts []geometry.Point) Quad { return Quad{pts[0], pts[1], pts[2], pts[3]} }

// buildQuads is Pipeline._field_quads. toInput is PipelineResults.to_input (canvas -> input
// photo); records are the kept detections in reading order, frames those of every detection of
// the fields passed to SplitWords.
func buildQuads(records []SplitRecord, frames []fieldFrame, toInput func([]geometry.Point) ([]geometry.Point, bool),
	needsLicenceRotation bool) *Quads {

	fields := map[string][]Quad{}
	words := map[string][]Quad{}
	for _, rec := range records {
		frame := frames[rec.Index]
		onCanvas, _ := frame.Map.ToInput(geometry.Corners(0, 0, float64(frame.W), float64(frame.H)))
		field, ok := toInput(onCanvas)
		if !ok {
			// The way back is not known for this run, and it is not known for any field of it:
			// say so once, for the whole run, instead of per box.
			return &Quads{}
		}
		fields[rec.Label] = append(fields[rec.Label], toQuad(field))
		if !rec.Split {
			words[rec.Label] = append(words[rec.Label], toQuad(field))
			continue
		}
		h, w := rec.PatchH, rec.PatchW
		patch := geometry.Chain{Maps: []geometry.Geometry{frame.Map}}
		if needsLicenceRotation && rec.Label == "Licence_number" {
			// the patch the words were found on is the field turned once: w x h of h x w
			patch = patch.Then(geometry.QuarterTurns{Width: h, Height: w, Turns: 1})
		}
		for _, box := range rec.WordBoxes {
			wx0, wy0 := math.Max(0, math.Trunc(box.X1)), math.Max(0, math.Trunc(box.Y1))
			wx1, wy1 := math.Min(float64(w), math.Trunc(box.X2)), math.Min(float64(h), math.Trunc(box.Y2))
			quad, _ := patch.ToInput(geometry.Corners(wx0, wy0, wx1, wy1))
			mapped, ok := toInput(quad)
			if !ok {
				return &Quads{}
			}
			words[rec.Label] = append(words[rec.Label], toQuad(mapped))
		}
	}
	return &Quads{Fields: fields, Words: words}
}

// QuadsPayload renders the `quads` stage: {"fields": {<Field>: [quad, ...]}, "words": {...}},
// each quad [[x, y] x 4] from the box's top-left corner, unrounded; both null when the way back
// is not known. When the address path ran (the registration page) the stage also carries
// "address_lines": one quad per line, or null when the way back is not known.
func QuadsPayload(q *Quads) any {
	out := map[string]any{"fields": nil, "words": nil}
	if q != nil && q.HasAddressLines {
		out["address_lines"] = nil
		if q.AddressLines != nil {
			out["address_lines"] = renderQuadList(q.AddressLines)
		}
	}
	if !q.Known() {
		return out
	}
	render := func(m map[string][]Quad) map[string][][4][2]float64 {
		res := make(map[string][][4][2]float64, len(m))
		for label, quads := range m {
			res[label] = renderQuadList(quads)
		}
		return res
	}
	out["fields"], out["words"] = render(q.Fields), render(q.Words)
	return out
}

// renderQuadList is a list of quadrilaterals, [[x, y] x 4] each.
func renderQuadList(quads []Quad) [][4][2]float64 {
	list := make([][4][2]float64, 0, len(quads))
	for _, qd := range quads {
		var c [4][2]float64
		for i, p := range qd {
			c[i] = [2]float64{p.X, p.Y}
		}
		list = append(list, c)
	}
	return list
}

// ToInput is PipelineResults.to_input: points of the canvas (Canvas) on the image passed to
// Run. ok is false when a stage of this run changed the image in a way no point map expresses
// (geometry.Unknown) - the canvas and what was read are fine, the way back is not known.
func (r *Results) ToInput(points []geometry.Point) ([]geometry.Point, bool) {
	if r.Geometry == nil {
		return points, true
	}
	return r.Geometry.ToInput(points)
}
