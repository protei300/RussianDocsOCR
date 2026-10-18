package models

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/preprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// ObbModel is a single-head oriented-box detector: letterbox in, rotated boxes out. Port of
// YOLOOBBDetectionModel (models.py:433-460). Users are AddressLines and RegistrationStamps.
//
// Its own type, like DetectionModel, because the return is a list of oriented boxes.
type ObbModel struct {
	Config *Config

	pre      preprocess.Preprocessor
	detector *postprocess.YoloObb
	session  *inference.Session
}

// LoadObb assembles a YOLOOBBDetection model through the shared dispatch.
func LoadObb(root, dir string, device inference.Device, threads int) (*ObbModel, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if cfg.ModelType != "YOLOOBBDetection" {
		return nil, fmt.Errorf("%w: %s is %q, not YOLOOBBDetection", ErrModelLoad, cfg.Name, cfg.ModelType)
	}
	if len(cfg.Inputs) != 1 || len(cfg.Outputs) != 1 {
		return nil, fmt.Errorf("%w: %s needs 1 input and 1 output, got %d and %d",
			ErrModelLoad, cfg.Name, len(cfg.Inputs), len(cfg.Outputs))
	}
	pre, err := newPreprocessor(cfg.Inputs[0])
	if err != nil {
		return nil, err
	}
	post, err := newPostprocessor(root, *cfg, cfg.Outputs[0])
	if err != nil {
		return nil, err
	}
	det, ok := post.(*postprocess.YoloObb)
	if !ok {
		return nil, fmt.Errorf("%w: %s output type %q is not an oriented-box detector",
			ErrModelLoad, cfg.Name, cfg.Outputs[0].Type)
	}
	sess, err := inference.Open(cfg.ModelPath(), device, threads)
	if err != nil {
		return nil, err
	}
	return &ObbModel{Config: cfg, pre: pre, detector: det, session: sess}, nil
}

func (m *ObbModel) Close() error { return m.session.Close() }

// Predict returns the oriented detections in the coordinates of img, in reading order. An
// empty result is normal.
func (m *ObbModel) Predict(img imaging.Image) ([]postprocess.ObbBox, error) {
	in, meta, err := m.pre.Apply(img)
	if err != nil {
		return nil, err
	}
	raw, err := m.session.Run([]*tensor.Array{in})
	if err != nil {
		return nil, err
	}
	if len(raw) < 1 {
		return nil, fmt.Errorf("models: %s returned no output", m.Config.Name)
	}
	res, err := m.detector.Apply(raw[0], postprocess.Context{
		Ratio:     meta.Ratio,
		PadExtra:  meta.PadExtra,
		PadLetter: meta.PadLetter,
		PaddedH:   meta.PaddedH,
		PaddedW:   meta.PaddedW,
		OrigH:     meta.OrigH,
		OrigW:     meta.OrigW,
		Resize:    true,
	})
	if err != nil {
		return nil, err
	}
	det, ok := res.(postprocess.ObbResult)
	if !ok {
		return nil, fmt.Errorf("models: %s detector output is %T", m.Config.Name, res)
	}
	return det.Boxes, nil
}
