package imaging

/*
#include "lsd_shim.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"gocv.io/x/gocv"
)

// The operations of the oriented-box path (the address lines and the registration stamps of the
// residence registration page): cutting a rotated rectangle out of an image upright, and the
// intersection of two rotated rectangles for the suppression of overlapping detections.

// RotationMatrixValues is cv2.getRotationMatrix2D((cx, cy), angleDeg, 1.0) as plain numbers.
func RotationMatrixValues(cx, cy, angleDeg float64) [2][3]float64 {
	m := RotationMatrix2D(cx, cy, angleDeg, 1.0)
	defer m.Close()
	var out [2][3]float64
	for r := 0; r < 2; r++ {
		for c := 0; c < 3; c++ {
			out[r][c] = m.GetDoubleAt(r, c)
		}
	}
	return out
}

// RotateAbout is `cv2.warpAffine(img, cv2.getRotationMatrix2D((cx, cy), angleDeg, 1.0), (W, H),
// flags=cv2.INTER_LINEAR)`: the image turned about (cx, cy), bilinear, the area turned in from
// outside the frame black. The 2x3 matrix is returned with it, because the way back from a patch
// cut out of the result needs the same matrix (geometry.NewAffine).
func RotateAbout(src Image, cx, cy, angleDeg float64) (Image, [2][3]float64) {
	m := RotationMatrix2D(cx, cy, angleDeg, 1.0)
	defer m.Close()
	var matrix [2][3]float64
	for r := 0; r < 2; r++ {
		for c := 0; c < 3; c++ {
			matrix[r][c] = m.GetDoubleAt(r, c)
		}
	}
	return WarpAffine(src, m, src.Width(), src.Height(), false, BorderConstantZero), matrix
}

// GetRectSubPix is `cv2.getRectSubPix(src, (w, h), (cx, cy))`: the w x h window of src centred
// on a FRACTIONAL point, read bilinearly. gocv's own binding takes an integer centre, which would
// move every patch by up to half a pixel, so this goes through the shim of this package.
func GetRectSubPix(src Image, w, h int, cx, cy float64) (Image, error) {
	dst := gocv.NewMat()
	in := src.Mat()
	if rc := C.rdocs_get_rect_sub_pix(unsafe.Pointer(in.Ptr()), C.int(w), C.int(h),
		C.double(cx), C.double(cy), unsafe.Pointer(dst.Ptr())); rc != 0 {
		_ = dst.Close()
		return Image{}, fmt.Errorf("imaging: getRectSubPix failed (%d)", int(rc))
	}
	return Image{mat: dst}, nil
}

// RotatedIntersectionArea is cv2.contourArea(cv2.rotatedRectangleIntersection(a, b)[1]) for two
// rectangles (cx, cy, w, h, angle in DEGREES), float32 each as the reference hands them over; 0
// when they do not meet.
func RotatedIntersectionArea(a, b [5]float32) (float64, error) {
	area := float64(C.rdocs_rotated_intersection_area((*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0]))))
	if area < 0 {
		return 0, fmt.Errorf("imaging: rotatedRectangleIntersection failed")
	}
	return area, nil
}
