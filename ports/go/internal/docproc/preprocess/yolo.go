package preprocess

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Yolo is the letterbox input path for the detectors — Borders, TextFields, Words.
// Port of YoloPreprocessing (preprocessing.py:139-249), called with auto=False,
// scaleFill=False, scaleup=True, stride=32.
//
// Output is **uint8 NHWC in 0-255**: the /255 normalisation is baked into these ONNX
// graphs by the project's "wrap-and-bake" export convention. AddressLines is the one
// exception and uses OBBPreprocessing instead.
type Yolo struct {
	height       int
	width        int
	paddingSize  []int
	paddingColor []int
}

// letterboxFill is the grey the reference pads with. Not black, and not configurable:
// the models were trained on it.
var letterboxFill = [3]uint8{114, 114, 114}

func NewYolo(shape []int, paddingSize, paddingColor []int) (*Yolo, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("preprocess: Yolo needs a Shape of at least [H,W]")
	}
	return &Yolo{height: shape[0], width: shape[1],
		paddingSize: paddingSize, paddingColor: paddingColor}, nil
}

func (y *Yolo) Apply(img imaging.Image) (*tensor.Array, Meta, error) {
	padded, extra := Pad(img, y.paddingSize, y.paddingColor)
	defer padded.Close()

	// The shape AFTER the extra padding and BEFORE the letterbox. The segmentor needs
	// it to undo the scaling, so it travels in Meta rather than being recomputed.
	paddedH, paddedW := padded.Height(), padded.Width()

	boxed, ratio, padTo := letterbox(padded, y.height, y.width)
	defer boxed.Close()

	buf, err := boxed.Bytes()
	if err != nil {
		return nil, Meta{}, err
	}
	arr, err := tensor.Uint8Of([]int{1, boxed.Height(), boxed.Width(), boxed.Channels()}, buf)
	if err != nil {
		return nil, Meta{}, err
	}
	return arr, Meta{
		Ratio:     ratio,
		PadExtra:  extra,
		PadLetter: padTo,
		PaddedH:   paddedH,
		PaddedW:   paddedW,
		OrigH:     img.Height(),
		OrigW:     img.Width(),
	}, nil
}

// letterbox scales an image into a target box, preserving aspect ratio, and pads the
// remainder with grey.
//
// Returns the scale ratio and the HALVED padding (dw, dh) — halved because that is what
// the reference passes on as `pad_to_size`, and the detector's coordinate mapping
// subtracts exactly that value.
//
// The `-0.1 / +0.1` asymmetry is the part to leave alone. When the leftover padding is
// odd, `int(round(dh - 0.1))` and `int(round(dh + 0.1))` split it so the extra pixel
// lands on the BOTTOM and the RIGHT. Splitting it evenly — the obvious tidy-up — shifts
// every detected box by a pixel. Verified against the reference: a 528x727 input
// produces borders (0, 0, 87, 88) and a 1789x1083 one (126, 127, 0, 0).
func letterbox(src imaging.Image, targetH, targetW int) (imaging.Image, float64, [2]float64) {
	h, w := src.Height(), src.Width()

	r := float64(targetH) / float64(h)
	if rw := float64(targetW) / float64(w); rw < r {
		r = rw
	}
	// scaleup=True, so r is used as-is and images smaller than the target are enlarged.

	newW := roundHalfEven(float64(w) * r)
	newH := roundHalfEven(float64(h) * r)

	dw := float64(targetW-newW) / 2
	dh := float64(targetH-newH) / 2

	// Resize only when the size actually changes, matching the reference's
	// `if shape[::-1] != new_unpad` guard. Ownership is tracked with a flag rather
	// than by inspecting the Mat, so exactly one Close happens.
	scaled, weOwnScaled := src, false
	if w != newW || h != newH {
		scaled = imaging.Resize(src, newW, newH, imaging.InterLinear)
		weOwnScaled = true
	}

	top := roundHalfEven(dh - 0.1)
	bottom := roundHalfEven(dh + 0.1)
	left := roundHalfEven(dw - 0.1)
	right := roundHalfEven(dw + 0.1)

	out := imaging.CopyMakeBorderConstant(scaled, top, bottom, left, right,
		letterboxFill[0], letterboxFill[1], letterboxFill[2])
	if weOwnScaled {
		_ = scaled.Close()
	}
	return out, r, [2]float64{dw, dh}
}

// roundHalfEven matches numpy/Python round(), which breaks ties to EVEN.
//
// Go's math.Round is half-away-from-zero. The difference is one pixel of padding, and
// therefore one pixel on every box (CONVENTIONS §6.5).
func roundHalfEven(v float64) int {
	f := float64(int64(v))
	switch d := v - f; {
	case d > 0.5:
		return int(f) + 1
	case d < 0.5:
		return int(f)
	default:
		if int64(f)%2 == 0 {
			return int(f)
		}
		return int(f) + 1
	}
}
