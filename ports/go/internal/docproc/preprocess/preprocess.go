// Package preprocess turns an image into a model input tensor.
//
// One-method interface plus free helper functions, no inheritance — Python's
// `BasePreprocessing.padding` becomes a free function here, because Go embedding is
// not virtual dispatch and translating the base class would make this port diverge
// from the C#/Kotlin ones immediately (CONVENTIONS §5).
//
// A note that explains an absence: `BasePreprocessing.normalization` exists in Python
// but is SHADOWED by a same-named tuple attribute set in __init__, so it is
// uncallable and no model is ever normalised in preprocessing. Every shipped graph
// bakes its own scaling in. That is documented dead code, deliberately not ported —
// adding normalisation here would silently change every model's input.
package preprocess

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Meta records what a preprocessor did to the geometry, so a detector's
// postprocessor can map boxes back. Zero for the classification paths.
type Meta struct {
	Ratio     float64
	PadExtra  [2]int
	PadLetter [2]float64
	PaddedH   int
	PaddedW   int
	OrigH     int
	OrigW     int
}

// Preprocessor converts an image into one model input.
type Preprocessor interface {
	Apply(img imaging.Image) (*tensor.Array, Meta, error)
}

// ErrNotImplemented marks a recognised but deliberately unimplemented input type.
var ErrNotImplemented = fmt.Errorf("preprocess: not implemented")

// NotImplemented is a placeholder for a recognised-but-unimplemented input type.
// Wired rather than omitted, so a port cannot quietly grow a different behaviour for
// it (D-06).
type NotImplemented struct{ Tag string }

func (n NotImplemented) Apply(imaging.Image) (*tensor.Array, Meta, error) {
	return nil, Meta{}, fmt.Errorf("%w: input type %q", ErrNotImplemented, n.Tag)
}

// Pad applies the symmetric constant border from a config's PaddingSize.
//
// Every shipped model.json declares PaddingSize [0,0], so this is a no-op in practice —
// ported because it is part of the contract and a future artifact could use it, and
// because Python returns the applied offsets for the postprocessor to undo.
//
// Note the halving: Python pads `pad_v//2` top AND bottom, so a PaddingSize of [4,6]
// adds 3 rows above and below, not 6 in total.
func Pad(img imaging.Image, paddingSize []int, paddingColor []int) (imaging.Image, [2]int) {
	if len(paddingSize) < 2 || (paddingSize[0] == 0 && paddingSize[1] == 0) {
		return img.Clone(), [2]int{0, 0}
	}
	padH, padV := paddingSize[0]/2, paddingSize[1]/2
	var r, g, b uint8
	if len(paddingColor) >= 3 {
		r, g, b = uint8(paddingColor[0]), uint8(paddingColor[1]), uint8(paddingColor[2])
	}
	return imaging.CopyMakeBorderConstant(img, padV, padV, padH, padH, r, g, b), [2]int{padH, padV}
}
