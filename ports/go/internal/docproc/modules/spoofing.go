package modules

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// Spoofing covers both anti-spoofing heads. They are one type here because they differ
// in exactly one thing — whether the module applies a confidence gate on top of the
// model's own threshold — and Python's two near-identical files gain nothing by being
// separate.
//
// Verdict vocabulary is 'REAL' / 'FAKE', unlike Glare and Blur's 'good' / 'bad'. The
// quality dict genuinely mixes the two, and the conformance rules record it.
type Spoofing struct {
	name  string
	model *models.Model
	// gate, when non-zero, forces 'FAKE' whenever the score falls below it — applied
	// AFTER the model's own binary threshold. PrintSpoofing sets 0.9; LCDSpoofing
	// has none and reports the model's verdict unchanged.
	gate float64
}

// printSpoofingGate is PrintSpoofing.threshold in the reference.
//
// Note what this does in combination with the model's own 0.5 threshold from
// model.json: the labels are ["FAKE","REAL"], so the model already says REAL at
// p >= 0.5, and this gate then demands p >= 0.9. The effective decision boundary is
// therefore 0.9, and the two thresholds are NOT redundant — the model's picks the
// label, this one overrides it.
const printSpoofingGate = 0.9

func NewPrintSpoofing(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*Spoofing, error) {
	return newSpoofing("PrintSpoofing", printSpoofingGate, root, paths, format, device, threads)
}

// NewLCDSpoofing builds the screen-capture detector, which applies no extra gate.
func NewLCDSpoofing(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*Spoofing, error) {
	return newSpoofing("LCDSpoofing", 0, root, paths, format, device, threads)
}

func newSpoofing(name string, gate float64, root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*Spoofing, error) {
	dir, err := paths.Dir(name, format)
	if err != nil {
		return nil, err
	}
	m, err := models.Load(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: %s: %w", name, err)
	}
	return &Spoofing{name: name, model: m, gate: gate}, nil
}

func (s *Spoofing) Close() error { return s.model.Close() }
func (s *Spoofing) Name() string { return s.name }

// Predict classifies the whole image — no tiling, unlike Glare and Blur.
func (s *Spoofing) Predict(img imaging.Image) (label string, score float64, err error) {
	out, err := s.model.Predict(img)
	if err != nil {
		return "", 0, err
	}
	cls, ok := out[0].(postprocess.ClassResult)
	if !ok {
		return "", 0, fmt.Errorf("modules: %s output is %T, want ClassResult", s.name, out[0])
	}

	// The reported score is the RAW model score in both branches, never recomputed for
	// the overridden label — the pipeline stores it as-is and the view model exposes it.
	if s.gate > 0 && cls.Confidence < s.gate {
		return "FAKE", cls.Confidence, nil
	}
	return cls.Label, cls.Confidence, nil
}
