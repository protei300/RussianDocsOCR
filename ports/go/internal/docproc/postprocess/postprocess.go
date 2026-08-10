// Package postprocess turns raw model outputs into the library's result types.
//
// Two design rules from CONVENTIONS, both deliberate:
//
//   - **One-method interface, no inheritance.** Python's class hierarchy has exactly
//     two real uses of inheritance and both are flattened here, because Go embedding
//     is not virtual dispatch and a "subclass" that overrides a method silently calls
//     the base one.
//   - **A closed set of result types**, not `any`. Python's postprocessors return
//     tuples of varying arity; using `any` everywhere would push a type switch into
//     every caller. Instead the module layer — which knows what it asked for — does
//     ONE checked assertion, in one place.
package postprocess

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Context carries what a postprocessor may need beyond the tensor itself. An explicit
// typed struct rather than a variadic bag of `any`, so the three ports read alike.
type Context struct {
	// PaddingMeta and ImgShape are used by the detector postprocessors to map boxes
	// back through the letterbox. Unused by the classification heads.
	Ratio     float64
	PadExtra  [2]int
	PadLetter [2]float64
	PaddedH   int
	PaddedW   int
	OrigH     int
	OrigW     int
	Resize    bool
}

// Result is the closed set of postprocessor outputs. A sealed interface: the
// unexported marker keeps the set closed to this package, which is what makes the
// single type assertion in the module layer safe to reason about.
type Result interface {
	isResult()
}

// ClassResult is a label plus its score — BinaryClassification and
// MultiLabelClassification.
type ClassResult struct {
	Label      string
	Confidence float64
}

func (ClassResult) isResult() {}

// MetricResult is the nearest-centroid verdict: the label, the distance to it, and the
// per-class threshold it was judged against.
//
// Distance and Threshold are reported, not just the label, because DocTypeAngles turns
// them into a confidence (`1 - dist/threshold`) and because a rejected match still
// carries useful diagnostics.
type MetricResult struct {
	Label     string
	Distance  float64
	Threshold float64
}

func (MetricResult) isResult() {}

// TextResult is a decoded string — the OCR heads.
type TextResult struct {
	Text string
}

func (TextResult) isResult() {}

// Postprocessor converts one model output.
type Postprocessor interface {
	Apply(out *tensor.Array, ctx Context) (Result, error)
}

// ErrNotImplemented marks a tag that is recognised but deliberately unimplemented.
//
// Such tags are WIRED rather than omitted: an omitted case reads as an oversight and
// gets "helpfully" added differently in each port (D-06).
var ErrNotImplemented = fmt.Errorf("postprocess: not implemented")

// NotImplemented is a placeholder for a recognised-but-unimplemented output type.
type NotImplemented struct{ Tag string }

func (n NotImplemented) Apply(*tensor.Array, Context) (Result, error) {
	return nil, fmt.Errorf("%w: output type %q", ErrNotImplemented, n.Tag)
}
