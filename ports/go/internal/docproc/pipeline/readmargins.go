package pipeline

import (
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
)

// readMargins is Pipeline._read_margins: re-cut the patch of a field labelled tight to its
// letters with a vertical margin, so the reading gets whole glyphs. In place.
//
// The box stays as the detector gave it - only the crop that is READ grows. The STS special
// marks are why: their lines stand so close that a labelling margin merged neighbours (39-41 %
// overlap on real canvases), so they were labelled tight, and the detector learnt to cut the
// tops and bottoms off the letters.
//
// The margin never reaches the next line: it stops halfway to the nearest box above or below
// that shares part of the width, of any label. Single-canvas path only; the per-page path
// reads passports, which have no such field.
//
// The reference's boxes carry integer coordinates, so every comparison is on the truncated
// values; `//` is floor division of non-negative numbers.
func readMargins(fields []modules.Field, opts OcrOptions, canvas imaging.Image, frames []fieldFrame) error {
	if len(opts.ReadMargin) == 0 || len(fields) == 0 {
		return nil
	}
	height := canvas.Height()
	for i := range fields {
		share, ok := opts.ReadMargin[fields[i].Box.Label]
		if !ok || share == 0 {
			continue
		}
		b := fields[i].Box
		x1, y1, x2, y2 := int(b.X1), int(b.Y1), int(b.X2), int(b.Y2)
		pad := int(math.RoundToEven(float64(y2-y1) * share))
		top, bottom := max(0, y1-pad), min(height, y2+pad)
		for j := range fields {
			o := fields[j].Box
			ox1, oy1, ox2, oy2 := int(o.X1), int(o.Y1), int(o.X2), int(o.Y2)
			if j == i || min(x2, ox2) <= max(x1, ox1) {
				continue // no shared width
			}
			if oy2 <= y1 { // a line above
				top = max(top, (oy2+y1+1)/2)
			} else if oy1 >= y2 { // a line below
				bottom = min(bottom, (y2+oy1)/2)
			}
		}
		if top == y1 && bottom == y2 {
			continue
		}
		patch, err := imaging.ClampedCrop(canvas, x1, top, x2, bottom)
		if err != nil {
			return err
		}
		_ = fields[i].Patch.Close()
		fields[i].Patch = patch
		// the frame (geometry.go) moves with the crop
		frames[i] = fieldFrame{Map: geometry.Offset{DX: -float64(x1), DY: -float64(top)},
			W: patch.Width(), H: patch.Height()}
	}
	return nil
}
