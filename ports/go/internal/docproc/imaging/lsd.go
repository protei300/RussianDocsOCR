package imaging

/*
#include "lsd_shim.h"
*/
import "C"

import (
	"unsafe"
)

// lsdSegments is cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0] reshaped to
// (N, 4) float64: the line segment detector of OpenCV's imgproc, which gocv does not bind. It
// is called through a small C++ shim of this package (lsd_shim.cpp) so that the segments are
// the reference's own - the same library code the Python wheel runs, not a stand-in.
func lsdSegments(gray Image) []Segment {
	if gray.Empty() {
		return nil
	}
	m := gray.Mat()
	var out *C.float
	n := int(C.rdocs_lsd_detect(unsafe.Pointer(m.Ptr()), &out))
	if n <= 0 {
		return nil
	}
	defer C.rdocs_lsd_free(out)
	flat := unsafe.Slice((*float32)(unsafe.Pointer(out)), 4*n)
	segs := make([]Segment, n)
	for i := range segs {
		segs[i] = Segment{float64(flat[4*i]), float64(flat[4*i+1]), float64(flat[4*i+2]), float64(flat[4*i+3])}
	}
	return segs
}

