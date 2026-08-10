package imaging

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

// ClampedCrop is the ONLY sanctioned way to crop in this port. Do not call
// gocv.Mat.Region directly.
//
// This exists because of the highest-risk divergence in the whole port
// (CONVENTIONS §6.6). The Python pipeline crops with a numpy slice:
//
//	img[box[1]:box[3], box[0]:box[2]]     # WordsDetector, TextFieldsDetector
//
// and numpy slicing is FORGIVING in two specific ways that no OpenCV binding is:
//
//   - an upper bound past the edge is silently clamped to the edge;
//   - a NEGATIVE start is interpreted as counting from the end.
//
// The detector clamps negatives to zero only when `resize` is set
// (`detect_res[detect_res < 0] = 0` in postprocessing.py) and never clamps the upper
// bound against the image dimensions at all. So out-of-range boxes DO reach this
// code path in practice.
//
// gocv's Region, OpenCvSharp's Mat[Rect] and the JVM's submat all throw instead. A
// port that "works" is therefore one that clamps — and it must clamp the way numpy
// effectively does, to [0, dim], rather than throwing or returning a differently
// sized crop, or the OCR input silently changes size and the text changes with it.
//
// The negative-start case is deliberately NOT reproduced as from-the-end indexing:
// that behaviour is an accident of numpy rather than intent, it would produce a crop
// from the opposite side of the document, and the detector already zeroes negatives
// on the path that matters. Negatives are clamped to 0 and that choice is recorded
// here rather than left to be rediscovered.
//
// An empty intersection is not an error: OCRv2Preprocessing has an explicit
// degenerate-crop path that returns a 32x16 black patch, so a zero-area box must be
// representable rather than fatal.
func ClampedCrop(src Image, x1, y1, x2, y2 int) (Image, error) {
	w, h := src.Width(), src.Height()

	// Order the corners: a box with x2 < x1 is meaningless, and numpy would produce
	// an empty slice rather than a negative-width one.
	if x2 < x1 {
		x1, x2 = x2, x1
	}
	if y2 < y1 {
		y1, y2 = y2, y1
	}

	x1 = clamp(x1, 0, w)
	x2 = clamp(x2, 0, w)
	y1 = clamp(y1, 0, h)
	y2 = clamp(y2, 0, h)

	if x2 <= x1 || y2 <= y1 {
		// Zero-area after clamping. Return an empty 0x0 image of the right type;
		// callers that need a placeholder patch build one themselves, exactly as
		// OCRv2Preprocessing does.
		return Image{mat: gocv.NewMatWithSize(0, 0, gocv.MatTypeCV8UC3)}, nil
	}

	region := src.mat.Region(image.Rect(x1, y1, x2, y2))
	defer region.Close()
	// Region is a VIEW into the parent. Cloning is mandatory: the parent canvas is
	// reused and rebound across stages, and a view would change under the caller --
	// the same class of bug the service hit by reading PipelineResults after the
	// lease was released.
	return Image{mat: region.Clone()}, nil
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// CropRect is a convenience wrapper for callers that already hold a rectangle.
func CropRect(src Image, r image.Rectangle) (Image, error) {
	return ClampedCrop(src, r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
}

// MustBe8UC3 guards the assumption most of this package makes.
func MustBe8UC3(i Image) error {
	if i.mat.Type() != gocv.MatTypeCV8UC3 {
		return fmt.Errorf("imaging: expected CV_8UC3, got %v", i.mat.Type())
	}
	return nil
}
