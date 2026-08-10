package preprocess

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Classification is the input path for every classifier — DocTypeAngles, Blur, Glare,
// PrintSpoofing, LCDSpoofing, AddressTextKind. Port of ClassificationPreprocessing
// (preprocessing.py:90-134).
//
// The whole pipeline is: pad (a no-op with the shipped configs) -> resize to the
// declared size -> add a batch dimension. Output is **uint8 NHWC in 0-255**, not
// normalised float: the scaling lives inside each ONNX graph, and the inference layer
// casts to whatever dtype the session declares.
//
// One trap worth naming: Python calls `cv2.resize(image, self.image_size[:2])`, and
// cv2's dsize is **(width, height)** while the declared Shape is [H, W, C]. Every
// shipped classifier is square (224 or 128), so an axis swap here would be invisible —
// which is exactly why the conformance suite includes a deliberately non-square resize.
type Classification struct {
	height       int
	width        int
	paddingSize  []int
	paddingColor []int
}

// NewClassification builds the preprocessor from a model.json input block.
func NewClassification(shape []int, paddingSize, paddingColor []int) (*Classification, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("preprocess: Classification needs a Shape of at least [H,W]")
	}
	return &Classification{
		height: shape[0], width: shape[1],
		paddingSize: paddingSize, paddingColor: paddingColor,
	}, nil
}

func (c *Classification) Apply(img imaging.Image) (*tensor.Array, Meta, error) {
	padded, extra := Pad(img, c.paddingSize, c.paddingColor)
	defer padded.Close()

	// Python indexes image_size[:2] as (h, w) and hands that tuple to cv2.resize,
	// whose dsize is (w, h). For the square shipped sizes the two readings coincide;
	// the ordering below is the one that matches the reference.
	resized := imaging.Resize(padded, c.width, c.height, imaging.InterLinear)
	defer resized.Close()

	buf, err := resized.Bytes()
	if err != nil {
		return nil, Meta{}, err
	}
	// The batch dimension is added here, giving [1,H,W,C] — the same
	// `np.expand_dims(image, 0)` the reference applies.
	arr, err := tensor.Uint8Of([]int{1, resized.Height(), resized.Width(), resized.Channels()}, buf)
	if err != nil {
		return nil, Meta{}, err
	}
	return arr, Meta{
		PadExtra: extra,
		OrigH:    img.Height(),
		OrigW:    img.Width(),
	}, nil
}
