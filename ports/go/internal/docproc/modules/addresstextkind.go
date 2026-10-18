package modules

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// AddressTextKind says whether an address-line crop is printed or handwritten. Port of
// pipeline_modules/address_textkind_classifier. Address lines are wide and short, so the crop
// is put on a square grey canvas (aspect kept) before the standard square classification
// resize, and the glyph proportions are not distorted.
type AddressTextKind struct {
	model *models.Model
}

func NewAddressTextKind(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*AddressTextKind, error) {

	dir, err := paths.Dir("AddressTextKindClassifier", format)
	if err != nil {
		return nil, err
	}
	m, err := models.Load(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: AddressTextKindClassifier: %w", err)
	}
	return &AddressTextKind{model: m}, nil
}

func (c *AddressTextKind) Close() error { return c.model.Close() }

// Predict returns "printed" or "handwritten" and the raw score of the model, P(handwritten),
// whichever side of the threshold it falls on.
func (c *AddressTextKind) Predict(patch imaging.Image) (string, float64, error) {
	// letterbox_square: the image on a size x size canvas of (114, 114, 114); the odd pixel of
	// the margin goes below and to the right, as the reference's floor division puts it.
	h, w := patch.Height(), patch.Width()
	size := max(h, w)
	top, left := (size-h)/2, (size-w)/2
	square := imaging.CopyMakeBorderConstant(patch, top, size-h-top, left, size-w-left, 114, 114, 114)
	defer square.Close()

	out, err := c.model.Predict(square)
	if err != nil {
		return "", 0, err
	}
	cls, ok := out[0].(postprocess.ClassResult)
	if !ok {
		return "", 0, fmt.Errorf("modules: AddressTextKind output is %T, want ClassResult", out[0])
	}
	return cls.Label, cls.Confidence, nil
}
