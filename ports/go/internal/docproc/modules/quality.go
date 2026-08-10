package modules

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// windowSize is the tile edge the quality classifiers were trained on. Fixed at 128 in
// the reference ("window size must be 128").
const windowSize = 128

// tileVerdict is one classified tile.
type tileVerdict struct {
	Label      string
	Confidence float64
}

// classifyTiles resizes the image to a canvas of canvasX x canvasY tiles of 128 px and
// classifies every tile.
//
// Port of QualityChecker.perform_image, which exists in TWO near-identical copies in
// the reference (blur_detector/quality.py and glare_detector/quality.py). They are
// unified here because the difference between them is only in how the tile verdicts are
// aggregated, which lives in the callers below.
//
// One line deserves attention. The reference does:
//
//	self.tested_image = cv2.cvtColor(cv2.resize(image, canvas_in_pixels), cv2.COLOR_BGR2RGB)
//
// The pipeline hands it an **RGB** image, so despite the constant's name this is an
// RGB->BGR swap. The two conversions are the same channel permutation, so the effect is
// unambiguous: the classifiers see BGR. Reproduced as ToBGR, because "correcting" the
// call to match its name would feed every quality model the wrong channel order.
func classifyTiles(m *models.Model, img imaging.Image, canvasX, canvasY int) ([]tileVerdict, error) {
	canvas := imaging.Resize(img, canvasX*windowSize, canvasY*windowSize, imaging.InterLinear)
	defer canvas.Close()

	swapped := imaging.ToBGR(canvas)
	defer swapped.Close()

	// x outer, y inner — the reference's loop order. Aggregation is order-independent
	// (a sum), but keeping the order identical means a future per-tile comparison
	// needs no re-derivation.
	out := make([]tileVerdict, 0, canvasX*canvasY)
	for xStep := 0; xStep < canvasX; xStep++ {
		for yStep := 0; yStep < canvasY; yStep++ {
			x, y := windowSize*xStep, windowSize*yStep
			tile, err := imaging.ClampedCrop(swapped, x, y, x+windowSize, y+windowSize)
			if err != nil {
				return nil, err
			}
			res, err := m.Predict(tile)
			_ = tile.Close()
			if err != nil {
				return nil, err
			}
			cls, ok := res[0].(postprocess.ClassResult)
			if !ok {
				return nil, fmt.Errorf("modules: quality tile output is %T, want ClassResult", res[0])
			}
			out = append(out, tileVerdict{Label: cls.Label, Confidence: cls.Confidence})
		}
	}
	return out, nil
}
