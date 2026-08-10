package modules

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
)

// Glare detects specular highlights. Port of pipeline_modules/glare_detector.
//
// Verdict vocabulary is 'good' / 'bad' — NOT 'REAL' / 'FAKE' like the spoofing heads.
// The quality dict genuinely mixes vocabularies and the conformance rules record that
// (conformance/spec/tolerances.md).
type Glare struct {
	model *models.Model
}

// glareCanvas is the tile grid: 7 across, 4 down, i.e. a 896x512 canvas of 28 tiles.
var glareCanvas = [2]int{7, 4}

// glareConfidenceGate is the per-tile confidence above which a GLARE verdict counts.
//
// A tile is only held against the document when the classifier is quite sure: one
// flash ruins recognition, but a hesitant verdict on one tile out of 28 should not
// reject a usable scan.
const glareConfidenceGate = 0.85

func NewGlare(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*Glare, error) {
	dir, err := paths.Dir("Glare", format)
	if err != nil {
		return nil, err
	}
	m, err := models.Load(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: Glare: %w", err)
	}
	return &Glare{model: m}, nil
}

func (g *Glare) Close() error { return g.model.Close() }

// Predict returns the verdict and the glare score.
//
// The score is the FRACTION OF TILES showing confident glare, and the verdict is 'bad'
// as soon as that is greater than zero — a single confident glare tile is enough. That
// is deliberate in the reference ("One flash can spoil the recognition process, and I
// set zero level of Glare to pass the quality test"), so the comparison is `> 0` and
// not a threshold.
func (g *Glare) Predict(img imaging.Image) (label string, score float64, err error) {
	tiles, err := classifyTiles(g.model, img, glareCanvas[0], glareCanvas[1])
	if err != nil {
		return "", 0, err
	}
	if len(tiles) == 0 {
		return "", 0, fmt.Errorf("modules: Glare classified no tiles")
	}

	// Per tile: 0 when it is confidently glared, 1 otherwise. The score is then
	// 1 - mean, i.e. the glared fraction. Written this way round to match
	// check_image_quality exactly rather than simplifying to a count.
	sum := 0.0
	for _, t := range tiles {
		if t.Label == "GLARE" && t.Confidence > glareConfidenceGate {
			sum += 0
		} else {
			sum += 1
		}
	}
	score = 1 - sum/float64(len(tiles))

	if score > 0 {
		return "bad", score, nil
	}
	return "good", score, nil
}
