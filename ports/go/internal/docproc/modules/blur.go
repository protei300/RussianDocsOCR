package modules

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
)

// Blur scores image sharpness. Port of pipeline_modules/blur_detector.
//
// Verdict vocabulary is 'good' / 'bad'. The score runs 0 (extremely blurred) to 1
// (sharp), and 'good' requires **> 0.9** — i.e. at most 10 % blur.
type Blur struct {
	model *models.Model
}

// blurCanvas is the same 7x4 tile grid Glare uses.
var blurCanvas = [2]int{7, 4}

// blurGate is the sharpness a document must exceed to pass.
const blurGate = 0.9

func NewBlur(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*Blur, error) {
	dir, err := paths.Dir("Blur", format)
	if err != nil {
		return nil, err
	}
	m, err := models.Load(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: Blur: %w", err)
	}
	return &Blur{model: m}, nil
}

func (b *Blur) Close() error { return b.model.Close() }

// Predict returns the verdict and the sharpness score.
//
// Two details of the aggregation are load-bearing and neither is obvious:
//
//   - **Only three of the model's five labels contribute.** Blur5 weighs 0.5, Blur10
//     weighs 1, NonBlur weighs 0, and any other label is not counted AT ALL — it does
//     not even enter the denominator. So the mean is over recognised tiles only.
//   - **Zero recognised tiles returns 1.0, not a division by zero.** A degenerate or
//     untextured input yields no blur evidence either way, and defaulting to "sharp"
//     avoids rejecting a document on a classifier miss rather than on an actual blur
//     signal. The reference states this reasoning in a comment; it is behaviour, not a
//     guard.
func (b *Blur) Predict(img imaging.Image) (label string, score float64, err error) {
	tiles, err := classifyTiles(b.model, img, blurCanvas[0], blurCanvas[1])
	if err != nil {
		return "", 0, err
	}

	sum, counted := 0.0, 0
	for _, t := range tiles {
		switch t.Label {
		case "Blur5":
			sum += 0.5
			counted++
		case "Blur10":
			sum += 1
			counted++
		case "NonBlur":
			sum += 0
			counted++
		default:
			// Deliberately neither summed nor counted.
		}
	}
	if counted == 0 {
		return "good", 1.0, nil
	}
	score = 1 - sum/float64(counted)

	if score > blurGate {
		return "good", score, nil
	}
	return "bad", score, nil
}
