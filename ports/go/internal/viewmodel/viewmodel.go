package viewmodel

import (
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// coordSpaceNote is a LITERAL string and is compared BY VALUE by the checker.
//
// It records a real limitation rather than boilerplate: the library does not retain the
// deskew angle, so boxes cannot be mapped back onto the original upload. DocDetector's
// contour lives in pre-warp space and, if ever exposed, must be tagged "prewarp" so
// nobody draws it on the canvas.
const coordSpaceNote = "Box coordinates are in canvas pixel space and match the canvas " +
	"image exactly. They cannot be mapped onto the original upload: the library does not " +
	"retain the deskew angle."

// Payload is the view model: exactly fourteen top-level keys.
//
// Three things the spec calls out as easy to get wrong, all of which this struct encodes:
// there is NO top-level doc_conf (it is quality.DocConf), there is NO `original` block
// (the service adds it from the uploaded bytes, which the library never sees), and
// `device` IS part of the view model.
//
// Every JSON tag is written by hand. Go, C# and Kotlin have three different default
// naming policies and ~60 wire names must agree byte for byte, so an automatic policy is
// forbidden (CONVENTIONS §1). No `omitempty` anywhere, for the same reason: a key must be
// present and null, not absent.
type Payload struct {
	DocType     *string `json:"doc_type"`
	DocTypeBase *string `json:"doc_type_base"`
	DocTypeEra  *string `json:"doc_type_era"`
	Recognised  bool    `json:"recognised"`
	Device      *string `json:"device"`
	Canvas      Canvas  `json:"canvas"`

	CoordSpace     string `json:"coord_space"`
	CoordSpaceNote string `json:"coord_space_note"`

	Boxes  []Box   `json:"boxes"`
	Fields []Field `json:"fields"`

	Ocr     map[string]string  `json:"ocr"`
	Quality map[string]any     `json:"quality"`
	Timings map[string]float64 `json:"timings"`
	Address *Address           `json:"address"`

	// Debug is the only optional key: absent unless --include-debug is passed, which is
	// how the reference behaves too.
	Debug *Debug `json:"debug,omitempty"`
}

// Canvas carries the dimensions the client needs to scale the box overlay. The canvas
// IMAGE is not included — it is persisted separately by the artifact layer.
type Canvas struct {
	Width      *int `json:"width"`
	Height     *int `json:"height"`
	IsFallback bool `json:"is_fallback"`
}

// Box is an axis-aligned detection in CANVAS pixel space.
type Box struct {
	ID      string   `json:"id"`
	Label   string   `json:"label"`
	Display string   `json:"display"`
	Kind    string   `json:"kind"`
	X1      *int     `json:"x1"`
	Y1      *int     `json:"y1"`
	X2      *int     `json:"x2"`
	Y2      *int     `json:"y2"`
	Conf    *float64 `json:"conf"`
	Cls     *int     `json:"cls"`
	// Text is attached to the highest-confidence box of a label only.
	Text *string `json:"text"`
	// Ambiguous marks a box whose label's text belongs to a different box.
	Ambiguous bool `json:"ambiguous"`
}

// Field is one recognised field, linked to its box(es).
type Field struct {
	Name    string   `json:"name"`
	Display string   `json:"display"`
	Value   *string  `json:"value"`
	Script  string   `json:"script"`
	Conf    *float64 `json:"conf"`
	// BoxIds is a LIST because one field legitimately owns several boxes.
	BoxIds []string `json:"box_ids"`
}

type Debug struct {
	DocOutline DocOutline `json:"doc_outline"`
}

// DocOutline is EXPLICITLY tagged prewarp. Without the tag someone eventually draws this
// polygon on the canvas and files a bug.
type DocOutline struct {
	CoordSpace string     `json:"coord_space"`
	Polygon    [][][2]int `json:"polygon"`
}

// Input is everything the view model needs, and nothing else.
//
// A dedicated struct rather than the pipeline's own result type, mirroring transform.py's
// deliberate purity: the transform has no I/O and can be unit-tested from a literal,
// without loading 215 MB of models.
type Input struct {
	DocType string
	Device  string

	// CanvasW/CanvasH are the rectified canvas dimensions. The canvas IMAGE is
	// deliberately not part of this struct: the client only needs the dimensions to
	// scale its overlay, and taking the Mat would put an ownership question into a type
	// whose whole point is being pure and testable from a literal.
	CanvasW, CanvasH int
	// CanvasMissing is set when the run short-circuited before producing a canvas
	// (doc_type == 'NONE'), which surfaces as is_fallback.
	CanvasMissing bool

	// Boxes are the text-field detections, in the detector's own order.
	Boxes []postprocess.Box
	// Ocr maps field name to recognised value.
	Ocr map[string]string
	// Quality mixes strings (Glare, Blur, the two spoofing verdicts) and one float
	// (DocConf), exactly as the library's dict does.
	Quality map[string]any
	Timings map[string]float64

	// Segments are the pre-warp document contours, emitted only under --include-debug.
	Segments [][]imaging.Point
}

// Build assembles the view model.
func Build(in Input, includeDebug bool) Payload {
	ocr := in.Ocr
	if ocr == nil {
		// An empty OBJECT, not null: the key is non-nullable in the contract.
		ocr = map[string]string{}
	}

	boxes := buildBoxes(in.Boxes, ocr)

	quality := in.Quality
	if quality == nil {
		quality = map[string]any{}
	}
	timings := in.Timings
	if timings == nil {
		timings = map[string]float64{}
	}

	canvas := Canvas{}
	if in.CanvasMissing {
		// A short-circuited run never populates the warped image. In Python the property
		// RAISES rather than returning None, which is why the reference wraps it in a try
		// and sets the fallback flag from the except.
		canvas.IsFallback = true
	} else {
		canvas.Width = intp(in.CanvasW)
		canvas.Height = intp(in.CanvasH)
	}

	base := BaseDocType(in.DocType)
	var basePtr *string
	if base != "" {
		basePtr = str(base)
	}

	p := Payload{
		DocType:     str(in.DocType),
		DocTypeBase: basePtr,
		DocTypeEra:  DocTypeEra(in.DocType),
		// An unrecognised document is a legitimate outcome, not an error: 'NONE' is a
		// normal short return with a populated result, and the SPA renders it as a state.
		Recognised:     in.DocType != "" && in.DocType != "NONE",
		Device:         str(in.Device),
		Canvas:         canvas,
		CoordSpace:     "canvas",
		CoordSpaceNote: coordSpaceNote,
		Boxes:          boxes,
		Fields:         buildFields(in.DocType, ocr, boxes),
		Ocr:            ocr,
		Quality:        quality,
		Timings:        timings,
		Address:        nil,
	}

	if includeDebug {
		p.Debug = &Debug{DocOutline: DocOutline{
			CoordSpace: "prewarp",
			Polygon:    polygonOf(in.Segments),
		}}
	}
	return p
}

func polygonOf(segments [][]imaging.Point) [][][2]int {
	if segments == nil {
		return nil
	}
	out := make([][][2]int, 0, len(segments))
	for _, contour := range segments {
		pts := make([][2]int, 0, len(contour))
		for _, pt := range contour {
			// int(), which truncates toward zero — the reference casts with astype(int).
			pts = append(pts, [2]int{int(pt.X), int(pt.Y)})
		}
		out = append(out, pts)
	}
	return out
}
