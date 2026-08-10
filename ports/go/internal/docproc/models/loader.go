package models

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/preprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// ErrModelLoad is returned for an unrecognised tag in model.json.
//
// Python's `match` statements have no `else` and fall through returning None, so a
// typo in a tag becomes a nil dereference three stages later with a traceback pointing
// nowhere near the cause. Returning an error that NAMES the tag is a deliberate
// improvement over the reference (D-06).
var ErrModelLoad = fmt.Errorf("models: cannot load")

// Model is a loaded artifact: preprocessors, a session, and postprocessors.
type Model struct {
	Config *Config

	pre     []preprocess.Preprocessor
	post    []postprocess.Postprocessor
	session *inference.Session
}

// Load assembles a model from its directory.
//
// root is the library root, needed because the OCR mask is resolved from
// document_processing/config/ocr_alphabets.json — the model.json carries the model's FULL
// alphabet, a different thing from the characters a given document may emit.
func Load(root, dir string, device inference.Device, threads int) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}

	pre := make([]preprocess.Preprocessor, 0, len(cfg.Inputs))
	for _, in := range cfg.Inputs {
		p, err := newPreprocessor(in)
		if err != nil {
			return nil, err
		}
		pre = append(pre, p)
	}

	post := make([]postprocess.Postprocessor, 0, len(cfg.Outputs))
	for _, out := range cfg.Outputs {
		p, err := newPostprocessor(root, *cfg, out)
		if err != nil {
			return nil, err
		}
		post = append(post, p)
	}

	sess, err := inference.Open(cfg.ModelPath(), device, threads)
	if err != nil {
		return nil, err
	}

	// ModelType selects how inputs and outputs are wired together.
	//
	// The detector and segmentation wrappers have genuinely different Predict signatures
	// (a box list, and boxes plus contours) so they are separate types with their own
	// constructors, and asking for one through this entry point is a caller mistake
	// worth naming rather than a shape to guess at.
	switch cfg.ModelType {
	case "YOLODetection":
		_ = sess.Close()
		return nil, fmt.Errorf("%w: %s is a detector; use LoadDetection", ErrModelLoad, cfg.Name)
	case "YOLOSegmentation":
		_ = sess.Close()
		return nil, fmt.Errorf("%w: %s is a segmentor; use LoadSegmentation", ErrModelLoad, cfg.Name)
	case "YOLOOBBDetection":
		_ = sess.Close()
		return nil, fmt.Errorf("%w: ModelType %q (INTPASSPORTADDR) is deferred",
			postprocess.ErrNotImplemented, cfg.ModelType)
	default:
		// A REAL default, not an error: both "UnifiedModel" and "OCRUnified" land here
		// on purpose, exactly as ModelLoader.load_model's default case does.
		return &Model{Config: cfg, pre: pre, post: post, session: sess}, nil
	}
}

func (m *Model) Close() error { return m.session.Close() }

// Session exposes the underlying session, for callers that need its declared types.
func (m *Model) Session() *inference.Session { return m.session }

// Predict runs the model on one image and post-processes every output.
//
// Outputs are mapped to postprocessors POSITIONALLY, which is what gives DocTypeAngles
// its two heads: output 0 (the embedding) goes to Metric, output 1 (the angle scores)
// to MultiClass. Port of UnifiedModel.predict (models.py:523-582).
func (m *Model) Predict(img imaging.Image) ([]postprocess.Result, error) {
	if len(m.pre) != 1 {
		return nil, fmt.Errorf("models: %s has %d inputs; Predict handles one",
			m.Config.Name, len(m.pre))
	}
	in, meta, err := m.pre[0].Apply(img)
	if err != nil {
		return nil, err
	}

	raw, err := m.session.Run([]*tensor.Array{in})
	if err != nil {
		return nil, err
	}
	if len(raw) != len(m.post) {
		return nil, fmt.Errorf("models: %s returned %d output(s) but %d postprocessor(s) are configured",
			m.Config.Name, len(raw), len(m.post))
	}

	ctx := postprocess.Context{
		Ratio:    meta.Ratio,
		PadExtra: meta.PadExtra,
		OrigH:    meta.OrigH,
		OrigW:    meta.OrigW,
	}
	results := make([]postprocess.Result, len(raw))
	for i := range raw {
		r, err := m.post[i].Apply(raw[i], ctx)
		if err != nil {
			return nil, err
		}
		results[i] = r
	}
	return results, nil
}

// newPreprocessor dispatches on Inputs[i].Type.
//
// Cases are in the order of Python's `match` in ModelLoader.__load_preprocess, and the
// order is recorded in MAPPING.md so nobody sorts them alphabetically: keeping it makes
// the Go, C# and Kotlin files diff line-for-line.
func newPreprocessor(in Input) (preprocess.Preprocessor, error) {
	switch in.Type {
	case "Classification":
		return preprocess.NewClassification(in.Shape, in.PaddingSize, in.PaddingColor)
	case "YOLO":
		return preprocess.NewYolo(in.Shape, in.PaddingSize, in.PaddingColor)
	case "YOLOOBB":
		return preprocess.NotImplemented{Tag: in.Type}, nil // INTPASSPORTADDR, deferred
	case "OCR":
		// Legacy 31x200 grayscale. No shipped model.json declares it; wired so the
		// absence is explicit rather than looking like an omission.
		return preprocess.NotImplemented{Tag: in.Type}, nil
	case "OCRv2":
		height := 32
		if in.Height != nil {
			height = *in.Height
		}
		return preprocess.NewOcrV2(height, in.ColorOrder, in.Dtype)
	default:
		return nil, fmt.Errorf("%w: unknown input type %q", ErrModelLoad, in.Type)
	}
}

// newPostprocessor dispatches on Outputs[i].Type, in Python's case order.
func newPostprocessor(root string, cfg Config, out Output) (postprocess.Postprocessor, error) {
	switch out.Type {
	case "BinaryClassification":
		labels, err := out.LabelsAsStrings()
		if err != nil {
			return nil, err
		}
		// 0.5 is the documented default when Threshold is absent from the config.
		return postprocess.NewBinaryClass(labels, out.ThresholdOr(0.5))
	case "MultiLabelClassification":
		labels, err := out.LabelsAsStrings()
		if err != nil {
			return nil, err
		}
		return postprocess.NewMultiClass(labels)
	case "Metric":
		path := cfg.CentersPath(out)
		if path == "" {
			return nil, fmt.Errorf("%w: Metric output %q has no Centers", ErrModelLoad, out.Name)
		}
		return postprocess.NewMetric(path, out.Metric)
	case "YOLODetector":
		return newDetectorPost(out, postprocess.NmsClassAgnostic)
	case "PerClassYOLODetector":
		// Per-class suppression, because on external passports the ru/en field pairs
		// legitimately overlap at IOU 0.2-0.3 and class-agnostic NMS would silently drop
		// one field of each pair.
		return newDetectorPost(out, postprocess.NmsPerClass)
	case "YOLOOBBDetector":
		return postprocess.NotImplemented{Tag: out.Type}, nil // INTPASSPORTADDR, deferred
	case "YOLOSegmentor":
		return postprocess.NewYoloSegmentor(derefOr(out.MaskFilter, 0.8))
	case "OCR":
		return postprocess.NotImplemented{Tag: out.Type}, nil // legacy, removed in 3.0.0
	case "OCRFV":
		return postprocess.NotImplemented{Tag: out.Type}, nil // legacy, removed in 3.0.0
	case "OCRProbs":
		// The model.json carries the model's FULL alphabet; which of those characters
		// this document may emit comes from the shared per-country table. Conflating the
		// two silently disables masking.
		allowed, err := config.AllowedCharset(root, out.Script, out.Country)
		if err != nil {
			return nil, err
		}
		return postprocess.NewOcrProbs(out.Alphabet, allowed, out.BlankIndexOr(0))
	default:
		return nil, fmt.Errorf("%w: unknown output type %q", ErrModelLoad, out.Type)
	}
}

// newDetectorPost builds a detector head, shared by the two detector tags.
//
// The IOU/CLS defaults are the reference constructor's; both shipped configs set both
// keys explicitly, so the defaults exist only so a hand-written config behaves.
func newDetectorPost(out Output, mode postprocess.NmsMode) (postprocess.Postprocessor, error) {
	labels, err := out.LabelsAsStrings()
	if err != nil {
		return nil, err
	}
	return postprocess.NewYoloDetector(labels, derefOr(out.IOU, 0.2), derefOr(out.CLS, 0.5), mode)
}
