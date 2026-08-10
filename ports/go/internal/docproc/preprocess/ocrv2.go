package preprocess

import (
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// minOcrWidth is the floor on the resized patch width.
//
// The EdgeNext ('fast') backbone has three internal stride-2 downsample stages and
// errors out with "Invalid input shape" below 16 px (measured: fails up to 15, stable
// from 16). MobileNetV4 ('accurate') has no such floor but is unharmed by it, and no
// genuine text crop is narrower than this.
const minOcrWidth = 16

// OcrV2 preprocesses a word patch for the v2 OCR engines.
// Port of OCRv2Preprocessing (preprocessing.py:363-420).
//
// Fixed height, DYNAMIC width, no letterbox padding, normalisation baked into the ONNX
// graph rather than applied here. That dynamic width is what made this the spike's
// kill-shot test: the output length T varies with it, so the session must allocate the
// result rather than be told its shape.
type OcrV2 struct {
	height     int
	colorOrder string
	dtype      string
}

func NewOcrV2(height int, colorOrder, dtype string) (*OcrV2, error) {
	if height <= 0 {
		height = 32
	}
	if colorOrder == "" {
		colorOrder = "BGR"
	}
	if dtype == "" {
		dtype = "uint8"
	}
	return &OcrV2{height: height, colorOrder: colorOrder, dtype: dtype}, nil
}

// Apply returns a [1, height, W, 3] uint8 tensor.
//
// The channel swap is not incidental: the pipeline works in RGB but these models were
// trained on OpenCV BGR patches, so RGB input is flipped here. Dropping the swap changes
// every character the model reads while still producing plausible-looking text.
func (p *OcrV2) Apply(img imaging.Image) (*tensor.Array, Meta, error) {
	h, w := img.Height(), img.Width()

	// A zero-size crop is a real edge case (a zero-height box after clipping), not a
	// bug to crash on: OCR of a blank patch decodes to the empty string, which is the
	// honest answer. The shape is (height, 16, 3) — the minimum width, so the model
	// still accepts it.
	if h == 0 || w == 0 {
		blank, err := tensor.Uint8Of([]int{1, p.height, minOcrWidth, 3},
			make([]uint8, p.height*minOcrWidth*3))
		if err != nil {
			return nil, Meta{}, err
		}
		return blank, Meta{OrigH: h, OrigW: w, Ratio: 1}, nil
	}

	newW := minOcrWidth
	// round(), which is half-to-EVEN in Python. A width one pixel out changes T and
	// with it the CTC alignment (CONVENTIONS §6.5).
	if scaled := int(tensor.RoundHalfEven(float64(w)*float64(p.height)/float64(h), 0)); scaled > newW {
		newW = scaled
	}

	// The reference flips channels BEFORE resizing. Order matters here in principle
	// (INTER_LINEAR is per-channel, so it commutes with a channel permutation) but is
	// kept identical anyway rather than relying on that argument.
	src := img
	if p.colorOrder == "BGR" {
		src = imaging.ToBGR(img)
		defer src.Close()
	}

	resized := imaging.Resize(src, newW, p.height, imaging.InterLinear)
	defer resized.Close()

	buf, err := resized.Bytes()
	if err != nil {
		return nil, Meta{}, err
	}
	arr, err := tensor.Uint8Of(
		[]int{1, resized.Height(), resized.Width(), resized.Channels()}, buf)
	if err != nil {
		return nil, Meta{}, err
	}
	return arr, Meta{OrigH: h, OrigW: w, Ratio: 1}, nil
}
