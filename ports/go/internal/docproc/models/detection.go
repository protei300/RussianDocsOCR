package models

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/preprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// DetectionModel is a single-head YOLO detector: letterbox in, suppressed boxes out.
// Port of YOLODetectionModel (models.py:682-736). Users are TextFields and Words.
//
// Its own type rather than a mode on Model for the same reason SegmentationModel is: the
// return is a box list, not the closed Result set, and returning `any` would only move a
// type assertion into every caller.
type DetectionModel struct {
	Config *Config

	pre      preprocess.Preprocessor
	detector *postprocess.YoloDetector
	session  *inference.Session
}

// LoadDetection assembles a YOLODetection model, taking the NMS mode from the output's
// declared Type — "PerClassYOLODetector" for TextFields, "YOLODetector" for Words.
func LoadDetection(root, dir string, device inference.Device, threads int) (*DetectionModel, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if cfg.ModelType != "YOLODetection" {
		return nil, fmt.Errorf("%w: %s is %q, not YOLODetection",
			ErrModelLoad, cfg.Name, cfg.ModelType)
	}
	if len(cfg.Outputs) != 1 {
		return nil, fmt.Errorf("%w: %s needs 1 output, got %d",
			ErrModelLoad, cfg.Name, len(cfg.Outputs))
	}

	// Through the shared dispatch, not hand-built: the three switches in loader.go are
	// the whole portable core of this design (CONVENTIONS §2), and a wrapper that
	// constructs its own pre/postprocessor puts a second copy of that knowledge
	// somewhere the next port will not think to look.
	pre, err := newPreprocessor(cfg.Inputs[0])
	if err != nil {
		return nil, err
	}
	post, err := newPostprocessor(root, *cfg, cfg.Outputs[0])
	if err != nil {
		return nil, err
	}
	det, ok := post.(*postprocess.YoloDetector)
	if !ok {
		return nil, fmt.Errorf("%w: %s output type %q is not a detector",
			ErrModelLoad, cfg.Name, cfg.Outputs[0].Type)
	}

	sess, err := inference.Open(cfg.ModelPath(), device, threads)
	if err != nil {
		return nil, err
	}
	return &DetectionModel{Config: cfg, pre: pre, detector: det, session: sess}, nil
}

func (m *DetectionModel) Close() error { return m.session.Close() }

// Predict returns the detections in ORIGINAL-image coordinates, already in reading
// order (top-to-bottom, then left-to-right) and with labels attached.
//
// An empty result is normal, not an error: a field the detector cannot see simply does
// not appear, and a patch with no words falls back to the whole patch.
func (m *DetectionModel) Predict(img imaging.Image) ([]postprocess.Box, error) {
	in, meta, err := m.pre.Apply(img)
	if err != nil {
		return nil, err
	}
	raw, err := m.session.Run([]*tensor.Array{in})
	if err != nil {
		return nil, err
	}
	if len(raw) != 1 {
		return nil, fmt.Errorf("models: %s returned %d outputs, want 1", m.Config.Name, len(raw))
	}

	// resize=true always: the reference passes it unconditionally here, so the boxes
	// come back in the source image's own coordinates.
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
	det, ok := res.(postprocess.DetectResult)
	if !ok {
		return nil, fmt.Errorf("models: %s detector output is %T", m.Config.Name, res)
	}
	return det.Boxes, nil
}
