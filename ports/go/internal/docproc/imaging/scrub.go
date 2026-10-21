package imaging

/*
#include "lsd_shim.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"image"
	"unsafe"

	"gocv.io/x/gocv"
)

// In-memory recognition (ports/MEMORY.md §5): a buffer that held a document is OVERWRITTEN with
// zeros before it is released, not merely released. OpenCV's allocator hands freed blocks back to
// the heap as they are, so a closed Mat leaves the pixels of the passport in memory until
// something else happens to reuse the block.

// ErrNotDecoded is DecodeRGBScrubbed's answer for bytes the decoder produced nothing from — the
// service maps it to 422, every other error to 500.
var ErrNotDecoded = errors.New("imaging: the decoder produced no image")

// Zero overwrites every pixel with 0. Safe on a zero or empty Image.
func (i Image) Zero() {
	if i.Empty() {
		return
	}
	m := i.mat
	m.SetTo(gocv.NewScalar(0, 0, 0, 0))
}

// ZeroAndClose is Zero, then Close: the release of a buffer that held a document.
func (i Image) ZeroAndClose() {
	i.Zero()
	_ = i.Close()
}

// Share returns a second handle to the SAME pixels: a view of the whole image, which keeps the
// pixel memory alive (OpenCV reference-counts it) until both handles are closed.
//
// For tests that must look at a buffer AFTER its owner zeroed and released it — the only way to
// prove the zeroing happened without reading freed memory. Nothing in the pipeline uses it.
func (i Image) Share() Image {
	return Image{mat: i.mat.Region(image.Rect(0, 0, i.Width(), i.Height()))}
}

// IsAllZero reports whether every byte of the pixel buffer is 0 (false for an empty image).
func (i Image) IsAllZero() bool {
	if i.Empty() {
		return false
	}
	buf, err := i.Bytes()
	if err != nil {
		return false
	}
	for _, b := range buf {
		if b != 0 {
			return false
		}
	}
	return true
}

// DecodeRGBScrubbed is DecodeRGB for a document that must leave no copy behind: the encoded bytes
// are read where they lie (gocv's IMDecode first copies them into a native vector it frees
// without zeroing) and the intermediate BGR image is zeroed before it is released. The pixels are
// the ones DecodeRGB produces — the same cv::imdecode(IMREAD_COLOR) and BGR->RGB conversion.
//
// The caller owns the result and releases it with ZeroAndClose.
func DecodeRGBScrubbed(data []byte) (Image, error) {
	if len(data) == 0 {
		return Image{}, ErrNotDecoded
	}
	dst := gocv.NewMat()
	rc := C.rdocs_imdecode_rgb_scrubbed((*C.uchar)(unsafe.Pointer(&data[0])), C.int(len(data)),
		unsafe.Pointer(dst.Ptr()))
	switch {
	case rc == 0:
		return Image{mat: dst}, nil
	case rc == 1:
		_ = dst.Close()
		return Image{}, ErrNotDecoded
	default:
		_ = dst.Close()
		return Image{}, fmt.Errorf("imaging: decode failed (%d)", int(rc))
	}
}

// EncodePNGFromRGBScrubbed encodes an RGB image as PNG into a Go-owned buffer, with the same
// compression as WritePNGFromRGB. The BGR copy the encoder needs and the encoder's own native
// output buffer are zeroed before they are released; the returned slice is the caller's to zero.
func EncodePNGFromRGBScrubbed(rgb Image) ([]byte, error) {
	bgr := gocv.NewMat()
	defer func() {
		bgr.SetTo(gocv.NewScalar(0, 0, 0, 0))
		_ = bgr.Close()
	}()
	gocv.CvtColor(rgb.mat, &bgr, gocv.ColorRGBToBGR)

	native, err := gocv.IMEncodeWithParams(gocv.PNGFileExt, bgr,
		[]int{int(gocv.IMWritePngCompression), 3})
	if native != nil {
		defer native.Close()
	}
	if err != nil {
		return nil, fmt.Errorf("imaging: png encode: %w", err)
	}
	// GetBytes is a view of the native buffer: copied out, then the native one is zeroed.
	view := native.GetBytes()
	out := make([]byte, len(view))
	copy(out, view)
	clear(view)
	return out, nil
}
