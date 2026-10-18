package modules

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// StampKinds is RegistrationStamps.KINDS: the model's label -> the kind reported in the
// results. A label that is not here reads as "unclear".
var StampKinds = map[string]string{
	"Stamp_registered":   "registered",
	"Stamp_deregistered": "deregistered",
	"Stamp_unclear":      "unclear",
}

// RegistrationStamps finds the registration stamps of the residence registration page (issue
// #27). Port of pipeline_modules/registration_stamps: a YOLO-OBB detector of the same family as
// AddressLines, run on the same canvas. A stamp is one oriented box around the whole impression:
// registered, deregistered, or a stamp the detector could not tell apart (unclear). The stamp is
// located, not read.
type RegistrationStamps struct {
	model *models.ObbModel
}

// RegistrationStampsAvailable reports whether the weight set has the stamp detector. A set of
// models-v10 or older does not, and the page is then read without stamps, as before.
func RegistrationStampsAvailable(paths *config.ModelPaths, format string) bool {
	dir, err := paths.Dir("RegistrationStamps", format)
	if err != nil {
		return false
	}
	for _, name := range []string{"model.json", "model.onnx"} {
		if info, err := os.Stat(filepath.Join(dir, name)); err != nil || info.IsDir() {
			return false
		}
	}
	return true
}

func NewRegistrationStamps(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*RegistrationStamps, error) {

	dir, err := paths.Dir("RegistrationStamps", format)
	if err != nil {
		return nil, err
	}
	m, err := models.LoadObb(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: RegistrationStamps: %w", err)
	}
	return &RegistrationStamps{model: m}, nil
}

func (d *RegistrationStamps) Close() error { return d.model.Close() }

// Predict returns the stamps as oriented boxes, in reading order.
func (d *RegistrationStamps) Predict(img imaging.Image) ([]postprocess.ObbBox, error) {
	return d.model.Predict(img)
}
