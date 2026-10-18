package modules

import (
	"fmt"
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/models"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// AddressLines finds the oriented address-line regions of the residence registration page
// (INTPASSPORTADDR). Port of pipeline_modules/address_lines_detector: a YOLO-OBB detector, so a
// tilted line is captured as it lies and no deskew is needed; every line is then cut out
// upright, in reading order (top to bottom).
type AddressLines struct {
	model *models.ObbModel
}

func NewAddressLines(root string, paths *config.ModelPaths, format string,
	device inference.Device, threads int) (*AddressLines, error) {

	dir, err := paths.Dir("AddressLinesDetector", format)
	if err != nil {
		return nil, err
	}
	m, err := models.LoadObb(root, dir, device, threads)
	if err != nil {
		return nil, fmt.Errorf("modules: AddressLinesDetector: %w", err)
	}
	return &AddressLines{model: m}, nil
}

func (d *AddressLines) Close() error { return d.model.Close() }

// Predict returns the oriented boxes alone (AddressLinesDetector.predict).
func (d *AddressLines) Predict(img imaging.Image) ([]postprocess.ObbBox, error) {
	return d.model.Predict(img)
}

// PredictTransform returns the boxes and the upright patch of each, in the same order
// (AddressLinesDetector.predict_transform). The patches are owned by the caller.
func (d *AddressLines) PredictTransform(img imaging.Image) ([]postprocess.ObbBox, []imaging.Image, error) {
	boxes, err := d.model.Predict(img)
	if err != nil {
		return nil, nil, err
	}
	patches := make([]imaging.Image, 0, len(boxes))
	for _, ob := range boxes {
		patch, err := CropRotated(img, ob)
		if err != nil {
			for i := range patches {
				_ = patches[i].Close()
			}
			return nil, nil, err
		}
		patches = append(patches, patch)
	}
	return boxes, patches, nil
}

// degrees is np.degrees on a float64: the factor 180 / pi is divided at run time, in float64,
// like numpy (a constant expression would round once, from the exact quotient).
func degrees(rad float64) float64 {
	pi := math.Pi
	return rad * (180.0 / pi)
}

// obbSize is (max(1, round(w)), max(1, round(h))) with Python's half-to-even round().
func obbSize(ob postprocess.ObbBox) (int, int) {
	return max(1, int(math.RoundToEven(ob.W))), max(1, int(math.RoundToEven(ob.H)))
}

// CropRotated cuts a rotated rectangle out of img and returns it upright
// (AddressLinesDetector.crop_rotated): the image turned about the centre by the box's angle,
// then the w x h window about that centre read at sub-pixel precision.
func CropRotated(img imaging.Image, ob postprocess.ObbBox) (imaging.Image, error) {
	wi, hi := obbSize(ob)
	rotated, _ := imaging.RotateAbout(img, ob.Cx, ob.Cy, degrees(ob.Angle))
	defer rotated.Close()
	return imaging.GetRectSubPix(rotated, wi, hi, ob.Cx, ob.Cy)
}

// CropGeometry is the map from a CropRotated patch back to the image it was cut from
// (AddressLinesDetector.crop_geometry): the same rotation and the same sub-pixel window, so a
// point of the patch lands on the image. The rotation ran first, so it is the FIRST map of the
// chain.
func CropGeometry(ob postprocess.ObbBox) geometry.Geometry {
	wi, hi := obbSize(ob)
	m := imaging.RotationMatrixValues(ob.Cx, ob.Cy, degrees(ob.Angle))
	window := geometry.Offset{
		DX: -(ob.Cx - float64(wi-1)/2.0),
		DY: -(ob.Cy - float64(hi-1)/2.0),
	}
	return geometry.Chain{Maps: []geometry.Geometry{geometry.NewAffine(m), window}}
}
