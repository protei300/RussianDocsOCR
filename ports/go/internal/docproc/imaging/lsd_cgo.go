//go:build !customenv

package imaging

// The C++ shim of this package (lsd_shim.cpp) includes OpenCV's headers itself, so the
// package needs the same cgo directive gocv has: without it the CI build stopped at
// "opencv2/core.hpp: No such file or directory" (ports workflow, 2026-10-08). With the
// customenv tag the flags come from CGO_CPPFLAGS / CGO_LDFLAGS instead, as for gocv.

/*
#cgo !windows pkg-config: opencv4
*/
import "C"
