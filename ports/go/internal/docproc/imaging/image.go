// Package imaging is the ONLY package in this port allowed to import gocv.
//
// Everything else works with the types here. That boundary is what makes the
// OpenCV dependency swappable in principle and, more usefully, what keeps the
// ~30 operations the pipeline actually uses in one auditable place instead of
// scattered across twelve modules.
//
// It is a thin concrete package, not an interface. There will never be a second
// implementation, so an interface would buy nothing and cost the other ports an
// indirection they would have to replicate pointlessly (see CONVENTIONS §5). What
// the wrapper DOES add over the raw binding is deterministic disposal.
//
// # Disposal is not optional
//
// gocv's Mat, OpenCvSharp's Mat and the JVM's Mat are all unmanaged and all leak.
// Python's garbage collector hides this completely, which is exactly how a port
// that passes conformance still dies in production after 500 documents. Every
// Image owns its Mat and every allocation site must Close it:
//
//	img, err := imaging.Decode(data)
//	if err != nil { return err }
//	defer img.Close()
//
// There is a leak test in the package tests: process one sample 200 times and
// assert the resident set is flat.
package imaging

import (
	"fmt"

	"gocv.io/x/gocv"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// Image is an owned, 8-bit, 3-channel image. Channel ORDER IS NOT ENCODED in the
// type, deliberately: the pipeline works in RGB while the OCR models want BGR, and
// a type-level distinction would multiply conversions rather than prevent mistakes.
// Order is documented per function instead, exactly as the Python code does.
type Image struct {
	mat gocv.Mat
}

// Wrap takes ownership of a Mat. The caller must not Close it afterwards.
func Wrap(m gocv.Mat) Image { return Image{mat: m} }

// NewFilled makes an h x w RGB image of one colour.
//
// For tests and for the synthetic warm-up frame. Not a general constructor: real images
// come from LoadRGB or from a transform, and a second way to make one invites images whose
// channel order nobody knows.
func NewFilled(h, w int, r, g, b uint8) Image {
	mat := gocv.NewMatWithSizeFromScalar(
		gocv.NewScalar(float64(r), float64(g), float64(b), 0),
		h, w, gocv.MatTypeCV8UC3)
	return Image{mat: mat}
}

// Mat exposes the underlying matrix for use inside this package only. Returning it
// does not transfer ownership.
func (i Image) Mat() gocv.Mat { return i.mat }

// Close releases the underlying Mat. Safe to call on a zero Image.
func (i Image) Close() error {
	if i.mat.Empty() && i.mat.Ptr() == nil {
		return nil
	}
	return i.mat.Close()
}

func (i Image) Empty() bool   { return i.mat.Ptr() == nil || i.mat.Empty() }
func (i Image) Width() int    { return i.mat.Cols() }
func (i Image) Height() int   { return i.mat.Rows() }
func (i Image) Channels() int { return i.mat.Channels() }

// Clone returns an independently owned copy.
func (i Image) Clone() Image { return Image{mat: i.mat.Clone()} }

// Bytes returns the pixel buffer in row-major HxWxC order, which is the same layout
// numpy uses — so it can be handed straight to the .npy writer.
//
// Errors rather than returning a short buffer if the Mat is not continuous: a
// silently truncated tensor is far worse than a failed call, and everything this
// package produces is continuous.
func (i Image) Bytes() ([]byte, error) {
	if i.mat.Type() != gocv.MatTypeCV8UC3 && i.mat.Type() != gocv.MatTypeCV8U {
		return nil, fmt.Errorf("imaging: expected CV_8UC3 or CV_8U, got %v", i.mat.Type())
	}
	want := i.mat.Rows() * i.mat.Cols() * i.mat.Channels()
	buf := i.mat.ToBytes()
	if len(buf) != want {
		return nil, fmt.Errorf("imaging: %d bytes for %dx%dx%d (non-continuous Mat?)",
			len(buf), i.mat.Rows(), i.mat.Cols(), i.mat.Channels())
	}
	return buf, nil
}

// Shape returns [H, W, C], matching the numpy convention used by the goldens.
func (i Image) Shape() []int {
	return []int{i.mat.Rows(), i.mat.Cols(), i.mat.Channels()}
}

// OpenCVVersion reports the linked OpenCV. Worth surfacing in `info`: the Python
// reference runs 4.12 and this port links 4.13, and while the spike measured those
// as bit-identical on every operation the pipeline uses, a future divergence should
// be attributable at a glance rather than after a day of bisection.
func OpenCVVersion() string { return gocv.OpenCVVersion() }

// BindingVersion reports the gocv version.
func BindingVersion() string { return gocv.Version() }

// ToArray converts an image into an [H, W, 3] uint8 array.
//
// The byte layout of a continuous CV_8UC3 Mat is already numpy's, so this is a copy and a
// shape, not a transformation. Used by the stage sink to write .npy.
func ToArray(img Image) (*tensor.Array, error) {
	if err := MustBe8UC3(img); err != nil {
		return nil, err
	}
	data, err := img.Bytes()
	if err != nil {
		return nil, err
	}
	return tensor.Uint8Of([]int{img.Height(), img.Width(), 3}, data)
}
