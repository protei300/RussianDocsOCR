// Package modules holds one thin type per ML module, mirroring pipeline_modules/.
package modules

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// DocTypeAngles classifies the document type and its 90-degree rotation.
// Port of pipeline_modules/doctype_angles_classificator/doctype_angles.py.
type DocTypeAngles struct {
	model *models.Model
	// angleLabels are the numeric labels of the second head, [0,90,180,270]. Kept
	// separately because MultiClass hands back the label as a STRING and the caller
	// needs the integer to decide how many times to rotate.
	angleLabels []int
}

// DocTypeResult is the module's flat payload, matching the dict the Python module
// returns under the key 'DocTypeAngles'.
//
// Field names are the wire names on purpose: this payload is emitted as the
// `doctype.label` conformance stage and compared against the Python golden, so the
// JSON tags are part of the contract (conformance/spec/stages.md).
type DocTypeResult struct {
	DocType           string  `json:"doc_type"`
	DocTypeConfidence float64 `json:"doc_type_confidence"`
	Angle             int     `json:"angle"`
	AngleConfidence   float64 `json:"angle_confidence"`
}

const moduleName = "DocTypeAngles"

// NewDocTypeAngles loads the module from the repository's model directory.
func NewDocTypeAngles(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*DocTypeAngles, error) {

	dir, err := paths.Dir(moduleName, format)
	if err != nil {
		return nil, err
	}
	m, err := models.Load(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: %s: %w", moduleName, err)
	}
	if len(m.Config.Outputs) != 2 {
		_ = m.Close()
		return nil, fmt.Errorf("modules: %s expects 2 outputs (embeddings, angle), got %d",
			moduleName, len(m.Config.Outputs))
	}
	angleLabels, err := m.Config.Outputs[1].LabelsAsInts()
	if err != nil {
		_ = m.Close()
		return nil, fmt.Errorf("modules: %s angle labels: %w", moduleName, err)
	}
	return &DocTypeAngles{model: m, angleLabels: angleLabels}, nil
}

func (d *DocTypeAngles) Close() error { return d.model.Close() }

// Predict returns the type and angle without rotating anything.
func (d *DocTypeAngles) Predict(img imaging.Image) (DocTypeResult, error) {
	out, err := d.model.Predict(img)
	if err != nil {
		return DocTypeResult{}, err
	}

	// ONE checked assertion per output, at the layer that knows what it asked for.
	// This is the alternative to threading `any` through every caller — see
	// CONVENTIONS §5.
	metric, ok := out[0].(postprocess.MetricResult)
	if !ok {
		return DocTypeResult{}, fmt.Errorf("modules: %s output 0 is %T, want MetricResult",
			moduleName, out[0])
	}
	angle, ok := out[1].(postprocess.ClassResult)
	if !ok {
		return DocTypeResult{}, fmt.Errorf("modules: %s output 1 is %T, want ClassResult",
			moduleName, out[1])
	}

	// confidence = round(1 - dist/threshold, 2), and 0.0 when threshold is the
	// "no centroid within radius" sentinel — dividing by it would be a
	// ZeroDivisionError in Python and +Inf here, and the case is maximally unknown
	// anyway (doctype_angles.py:38).
	//
	// RoundHalfEven, not math.Round: np.round breaks ties to EVEN, and two decimals
	// on a value derived from a float distance reaches ties in practice.
	confidence := 0.0
	if metric.Threshold > 0 {
		confidence = tensor.RoundHalfEven(1-metric.Distance/metric.Threshold, 2)
	}

	angleDeg, err := d.angleFromLabel(angle.Label)
	if err != nil {
		return DocTypeResult{}, err
	}

	return DocTypeResult{
		DocType:           metric.Label,
		DocTypeConfidence: confidence,
		Angle:             angleDeg,
		AngleConfidence:   float64(angle.Confidence),
	}, nil
}

// PredictTransform is Predict plus the image rotated upright.
//
// `angle // 90` counter-clockwise rotations, exactly as the reference does. The angle
// is a multiple of 90 by construction (the head's labels are 0/90/180/270), so this is
// 0-3 whole-image rotations and never an interpolating warp.
func (d *DocTypeAngles) PredictTransform(img imaging.Image) (DocTypeResult, imaging.Image, error) {
	meta, err := d.Predict(img)
	if err != nil {
		return DocTypeResult{}, imaging.Image{}, err
	}

	current := img.Clone()
	for i := 0; i < meta.Angle/90; i++ {
		next := imaging.Rotate90CCW(current)
		_ = current.Close()
		current = next
	}
	return meta, current, nil
}

// angleFromLabel maps the winning class back to degrees.
//
// MultiClass returns the label as a string because most heads have textual classes;
// this head's are numeric, so the string is matched against the declared labels rather
// than parsed. Matching beats parsing here: it fails loudly if the config and the
// model ever disagree, instead of silently producing a plausible number.
func (d *DocTypeAngles) angleFromLabel(label string) (int, error) {
	for _, v := range d.angleLabels {
		if fmt.Sprintf("%g", float64(v)) == label || fmt.Sprintf("%d", v) == label {
			return v, nil
		}
	}
	return 0, fmt.Errorf("modules: %s angle label %q is not one of %v",
		moduleName, label, d.angleLabels)
}
