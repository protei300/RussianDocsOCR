package models

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/preprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// SegmentationModel wires a detection head and a mask head together.
// Port of YOLOSegmentionModel (models.py:785-834). Borders is the only user.
//
// Kept as its own type rather than a mode on Model because its Predict has a genuinely
// different signature — two outputs feeding each other — and hiding that behind a
// shared `any` return would push a type switch into the caller.
type SegmentationModel struct {
	Config *Config

	pre       preprocess.Preprocessor
	detector  *postprocess.YoloDetector
	segmentor *postprocess.YoloSegmentor
	session   *inference.Session
}

// LoadSegmentation assembles the Borders model.
func LoadSegmentation(root, dir string, device inference.Device, threads int) (*SegmentationModel, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if cfg.ModelType != "YOLOSegmentation" {
		return nil, fmt.Errorf("%w: %s is %q, not YOLOSegmentation",
			ErrModelLoad, cfg.Name, cfg.ModelType)
	}
	if len(cfg.Outputs) != 2 {
		return nil, fmt.Errorf("%w: %s needs 2 outputs (bbox, mask), got %d",
			ErrModelLoad, cfg.Name, len(cfg.Outputs))
	}

	// Through the shared dispatch, for the reason given in LoadDetection.
	pre, err := newPreprocessor(cfg.Inputs[0])
	if err != nil {
		return nil, err
	}
	detPost, err := newPostprocessor(root, *cfg, cfg.Outputs[0])
	if err != nil {
		return nil, err
	}
	det, ok := detPost.(*postprocess.YoloDetector)
	if !ok {
		return nil, fmt.Errorf("%w: %s output 0 is %q, not a detector",
			ErrModelLoad, cfg.Name, cfg.Outputs[0].Type)
	}
	// The reference calls the detector with numpy=True here, which skips label
	// attachment and the integer coercion of coordinates — the segmentor needs the
	// full-precision boxes.
	det = det.WithNumpyOnly()

	segPost, err := newPostprocessor(root, *cfg, cfg.Outputs[1])
	if err != nil {
		return nil, err
	}
	seg, ok := segPost.(*postprocess.YoloSegmentor)
	if !ok {
		return nil, fmt.Errorf("%w: %s output 1 is %q, not a segmentor",
			ErrModelLoad, cfg.Name, cfg.Outputs[1].Type)
	}

	sess, err := inference.Open(cfg.ModelPath(), device, threads)
	if err != nil {
		return nil, err
	}
	return &SegmentationModel{Config: cfg, pre: pre, detector: det,
		segmentor: seg, session: sess}, nil
}

func (m *SegmentationModel) Close() error { return m.session.Close() }

// Predict returns the surviving detections and their contours.
//
// Empty results are a normal outcome, not an error: a document whose borders cannot be
// found keeps its original image, and the caller relies on that.
func (m *SegmentationModel) Predict(img imaging.Image) ([]postprocess.Box, [][]imaging.Point, error) {
	in, meta, err := m.pre.Apply(img)
	if err != nil {
		return nil, nil, err
	}
	raw, err := m.session.Run([]*tensor.Array{in})
	if err != nil {
		return nil, nil, err
	}
	if len(raw) != 2 {
		return nil, nil, fmt.Errorf("models: %s returned %d outputs, want 2", m.Config.Name, len(raw))
	}

	ctx := postprocess.Context{
		Ratio:     meta.Ratio,
		PadExtra:  meta.PadExtra,
		PadLetter: meta.PadLetter,
		PaddedH:   meta.PaddedH,
		PaddedW:   meta.PaddedW,
		OrigH:     meta.OrigH,
		OrigW:     meta.OrigW,
		Resize:    true,
	}
	res, err := m.detector.Apply(raw[0], ctx)
	if err != nil {
		return nil, nil, err
	}
	det, ok := res.(postprocess.DetectResult)
	if !ok {
		return nil, nil, fmt.Errorf("models: %s detector output is %T", m.Config.Name, res)
	}
	if len(det.Boxes) == 0 {
		return nil, nil, nil
	}

	// The mask head's spatial reference is the PADDED, pre-letterbox size — not the
	// original image — which is why preprocess.Meta carries both.
	segments, err := m.segmentor.Segment(raw[1], det.Boxes, meta.PadExtra,
		meta.PaddedH, meta.PaddedW)
	if err != nil {
		return nil, nil, err
	}
	return det.Boxes, segments, nil
}

func derefOr(v *float64, def float64) float64 {
	if v == nil {
		return def
	}
	return *v
}
