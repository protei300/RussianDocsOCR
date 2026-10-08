package imaging

import (
	"math"
	"image/color"

	"gocv.io/x/gocv"
)

// Segment is a straight line piece (x1, y1) - (x2, y2), pixel coordinates.
type Segment struct {
	X1, Y1, X2, Y2 float64
}

// DetectSegments is line_refine._segments: the line segments of a grayscale image that are at
// least minLen long, from OpenCV's line segment detector (LSD_REFINE_STD).
//
// The reference calls cv2.createLineSegmentDetector. gocv v0.43.0 does not bind it (no
// LineSegmentDetector symbol anywhere in the package), so lsd.go reaches the same OpenCV
// routine through a small C++ shim; the segments are the reference's own, not a substitute.
// (Until 2026-10-08 this was Canny + HoughLinesP, a different detector that found a different
// segment set - the cause of the line-refinement part of D-03.)
func DetectSegments(gray Image, minLen float64) []Segment {
	all := lsdSegments(gray)
	out := make([]Segment, 0, len(all))
	for _, s := range all {
		if math.Hypot(s.X2-s.X1, s.Y2-s.Y1) >= minLen {
			out = append(out, s)
		}
	}
	return out
}

// Blob is one connected-component's bounding box, in the OpenCV
// ConnectedComponentsWithStats column order.
type Blob struct {
	X, Y, W, H, Area int
}

// ConnectedComponents8 labels an 8-bit binary image (8-connectivity) and returns each
// component's bounding stats, INCLUDING background as index 0 (matching
// cv2.connectedComponentsWithStats's row 0), so callers can replicate `big[0] = False`.
func ConnectedComponents8(binary Image) (labels []int32, w, h int, blobs []Blob) {
	labelsMat := gocv.NewMat()
	defer labelsMat.Close()
	statsMat := gocv.NewMat()
	defer statsMat.Close()
	centroids := gocv.NewMat()
	defer centroids.Close()

	n := gocv.ConnectedComponentsWithStatsWithParams(binary.mat, &labelsMat, &statsMat, &centroids,
		8, gocv.MatTypeCV32S, gocv.CCL_DEFAULT)

	h, w = binary.Height(), binary.Width()
	labels = make([]int32, h*w)
	lb := labelsMat.ToBytes()
	// CV_32S, 4 bytes/pixel, little-endian (matches every platform this ships on).
	for i := 0; i < h*w; i++ {
		o := i * 4
		labels[i] = int32(lb[o]) | int32(lb[o+1])<<8 | int32(lb[o+2])<<16 | int32(lb[o+3])<<24
	}
	blobs = make([]Blob, n)
	for i := 0; i < n; i++ {
		blobs[i] = Blob{
			X:    int(statsMat.GetIntAt(i, 0)),
			Y:    int(statsMat.GetIntAt(i, 1)),
			W:    int(statsMat.GetIntAt(i, 2)),
			H:    int(statsMat.GetIntAt(i, 3)),
			Area: int(statsMat.GetIntAt(i, 4)),
		}
	}
	return labels, w, h, blobs
}

// RotateNearestZero rotates a single-channel image about its centre with
// nearest-neighbour interpolation and zero border fill - the angle-search primitive
// _profile_tilt/_cell_tilts use to test a candidate de-skew angle on a binary/valid
// mask without inventing grey values.
func RotateNearestZero(src Image, angleDeg float64) Image {
	cx, cy := float64(src.Width())/2.0, float64(src.Height())/2.0
	m := RotationMatrix2D(cx, cy, angleDeg, 1.0)
	defer m.Close()
	return WarpAffine(src, m, src.Width(), src.Height(), true, BorderConstantZero)
}

// RemapVerticalDisplacement moves each pixel (x, y) to sample source row y + v[y*w+x]
// at the same column (dst(x, y) = src(x, y + v(x, y))), linear interpolation, replicate
// border - the forward model cv2.remap(page, xs, ys + v, ...) implements in
// line_dewarp.apply_dewarp.
func RemapVerticalDisplacement(src Image, v []float32, w, h int) Image {
	xVals := make([]float32, h*w)
	yVals := make([]float32, h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			xVals[idx] = float32(x)
			yVals[idx] = float32(y) + v[idx]
		}
	}
	mapX, errX := gocv.NewMatFromBytes(h, w, gocv.MatTypeCV32F, float32sToBytes(xVals))
	if errX != nil {
		return src.Clone()
	}
	defer mapX.Close()
	mapY, errY := gocv.NewMatFromBytes(h, w, gocv.MatTypeCV32F, float32sToBytes(yVals))
	if errY != nil {
		return src.Clone()
	}
	defer mapY.Close()

	dst := gocv.NewMat()
	gocv.Remap(src.mat, &dst, &mapX, &mapY, gocv.InterpolationLinear, gocv.BorderReplicate, color.RGBA{})
	return Image{mat: dst}
}

// fillConvexPolyMask is unused directly by callers (kept for reference parity with
// _features_in_quad's cv2.fillConvexPoly) - the port instead tests point-in-polygon in
// pure Go (modules.PointInConvex), which needs no Mat at all. Present so a future
// reader searching for fillConvexPoly finds the reasoning, not a missing piece.
