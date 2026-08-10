package modules

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// Field is one detected text field: its box and the crop taken from the canvas.
type Field struct {
	Box postprocess.Box
	// Patch is owned by the caller and must be closed. Closing a Field slice is what
	// FieldsClose is for.
	Patch imaging.Image
}

// TextFieldsDetector finds the labelled text regions on a rectified canvas.
// Port of pipeline_modules/textfields_detector. Uses the TextFields artifact, whose
// output type is PerClassYOLODetector.
type TextFieldsDetector struct {
	model *models.DetectionModel
}

func NewTextFieldsDetector(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*TextFieldsDetector, error) {

	dir, err := paths.Dir("TextFieldsDetector", format)
	if err != nil {
		return nil, err
	}
	m, err := models.LoadDetection(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: TextFieldsDetector: %w", err)
	}
	return &TextFieldsDetector{model: m}, nil
}

func (d *TextFieldsDetector) Close() error { return d.model.Close() }

// PredictTransform returns every detection together with its crop.
//
// rotateLicence turns the Licence_number patch 90 degrees counter-clockwise, which the
// internal passport needs because it prints the series and number sideways. Note the
// reference rotates the PATCH only, leaving the box in canvas coordinates — so the box
// and its patch deliberately disagree in orientation, and the view model draws the box.
//
// Crops go through imaging.ClampedCrop and nothing else: the detector does not clamp
// boxes against the canvas dimensions, so an out-of-range box genuinely reaches here,
// and Python's slice silently clamps where every OpenCV binding throws
// (CONVENTIONS §6.6).
func (d *TextFieldsDetector) PredictTransform(img imaging.Image, rotateLicence bool) ([]Field, error) {
	boxes, err := d.model.Predict(img)
	if err != nil {
		return nil, err
	}

	fields := make([]Field, 0, len(boxes))
	for _, b := range boxes {
		patch, err := imaging.ClampedCrop(img, int(b.X1), int(b.Y1), int(b.X2), int(b.Y2))
		if err != nil {
			FieldsClose(fields)
			return nil, err
		}
		if rotateLicence && b.Label == "Licence_number" {
			rotated := imaging.Rotate90CCW(patch)
			_ = patch.Close()
			patch = rotated
		}
		fields = append(fields, Field{Box: b, Patch: patch})
	}
	return fields, nil
}

// FieldsClose releases every patch in a slice. Python's GC hid this entirely; a port
// that passes conformance and then dies after 500 documents is the failure mode.
func FieldsClose(fields []Field) {
	for i := range fields {
		_ = fields[i].Patch.Close()
	}
}
