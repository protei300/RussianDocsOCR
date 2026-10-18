package preprocess

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Obb is the input path of the oriented-box detectors (AddressLines, RegistrationStamps). Port
// of OBBPreprocessing (preprocessing.py:236-249): the YOLO letterbox, then the layout the
// standard ultralytics ONNX export expects - float32, channels first, scaled by 1/255.
//
// The one place where a detector's input is NOT the baked-in uint8 NHWC tensor of Yolo. These
// two graphs were exported without the "wrap-and-bake" wrapper, so the conversion happens here.
// The division is a FLOAT32 one (numpy divides a float32 array by a Python float in float32),
// which Go does for a float32 operand and an untyped constant.
type Obb struct {
	yolo *Yolo
}

func NewObb(shape []int, paddingSize, paddingColor []int) (*Obb, error) {
	y, err := NewYolo(shape, paddingSize, paddingColor)
	if err != nil {
		return nil, err
	}
	return &Obb{yolo: y}, nil
}

func (o *Obb) Apply(img imaging.Image) (*tensor.Array, Meta, error) {
	nhwc, meta, err := o.yolo.Apply(img)
	if err != nil {
		return nil, Meta{}, err
	}
	if len(nhwc.Shape) != 4 {
		return nil, Meta{}, fmt.Errorf("preprocess: Obb expects NHWC from the letterbox, got %v", nhwc.Shape)
	}
	pix, err := nhwc.AsUint8()
	if err != nil {
		return nil, Meta{}, err
	}
	h, w, c := nhwc.Shape[1], nhwc.Shape[2], nhwc.Shape[3]
	out := make([]float32, len(pix))
	// [1, H, W, C] -> [1, C, H, W]
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for ch := 0; ch < c; ch++ {
				out[(ch*h+y)*w+x] = float32(pix[(y*w+x)*c+ch]) / 255
			}
		}
	}
	arr, err := tensor.Float32Of([]int{1, c, h, w}, out)
	if err != nil {
		return nil, Meta{}, err
	}
	return arr, meta, nil
}
