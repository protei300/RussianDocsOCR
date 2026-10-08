package imaging

// The OpenCV boundary for the vehicle registration certificate (STS) straightened by its
// printed blank - Pipeline._register_card and PageRegistrar.warp_matrix(fill=...) in the
// reference. Four small operations, each a single cv2 call there.

import (
	"image"
	"image/color"
	"math"
	"sort"

	"gocv.io/x/gocv"
)

// WarpByHomographyFill is cv2.warpPerspective(img, M, (w, h), INTER_LINEAR,
// BORDER_CONSTANT, borderValue=fill): what the warp puts where the page runs past the
// photo is the given colour instead of the repeated edge pixels (PageRegistrar's `fill`).
//
// fill is given in the image's own channel order (RGB here), channel by channel, exactly
// as `tuple(int(v) for v in fill)` is handed to cv2. gocv's colour argument names its
// fields BGR (val1 = B), so the first channel travels in .B.
func WarpByHomographyFill(src Image, H [3][3]float64, width, height int, fill [3]uint8) Image {
	m := gocv.NewMatWithSize(3, 3, gocv.MatTypeCV64F)
	defer m.Close()
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			m.SetDoubleAt(r, c, H[r][c])
		}
	}
	dst := gocv.NewMat()
	gocv.WarpPerspectiveWithParams(src.mat, &dst, m, image.Pt(width, height),
		gocv.InterpolationLinear, gocv.BorderConstant,
		color.RGBA{B: fill[0], G: fill[1], R: fill[2], A: 0})
	return Image{mat: dst}
}

// PerspectiveTransformF32 is cv2.perspectiveTransform on a float32 point array: the
// result is float32 (returned as float64 values that are exactly float32-representable).
//
// This is not cosmetic. `PageTemplate.quad_in_image` feeds float32 corners to it, so the
// quad a registration reports - and the native scale taken from its side lengths - are
// float32 values; the same quad in float64 differs in the 8th digit, which moves a
// perspective warp's fixed-point weights and changes pixels of the canvas. The call goes
// through OpenCV itself rather than a re-derivation of its loop.
func PerspectiveTransformF32(H [3][3]float64, pts []Point) []Point {
	n := len(pts)
	if n == 0 {
		return nil
	}
	buf := make([]float32, 0, 2*n)
	for _, p := range pts {
		buf = append(buf, float32(p.X), float32(p.Y))
	}
	src, err := gocv.NewMatFromBytes(1, n, gocv.MatTypeCV32FC2, float32sToBytes(buf))
	if err != nil {
		return nil
	}
	defer src.Close()
	tm := gocv.NewMatWithSize(3, 3, gocv.MatTypeCV64F)
	defer tm.Close()
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			tm.SetDoubleAt(r, c, H[r][c])
		}
	}
	dst := gocv.NewMat()
	defer dst.Close()
	if err := gocv.PerspectiveTransform(src, &dst, tm); err != nil {
		return nil
	}
	out := bytesToFloat32s(dst.ToBytes())
	if len(out) != 2*n {
		return nil
	}
	res := make([]Point, n)
	for i := range res {
		res[i] = Point{X: float64(out[2*i]), Y: float64(out[2*i+1])}
	}
	return res
}

// EstimateAffinePartialLMEDS is cv2.estimateAffinePartial2D(from, to, method=cv2.LMEDS)
// with the Python defaults (reprojection threshold 3, 2000 iterations, confidence 0.99,
// 10 refinement iterations): a similarity (rotation, uniform scale, translation) fitted by
// least median of squares. Both point sets are float32 arrays on the Python side.
//
// ok is false when OpenCV finds no model (cv2 returns None).
func EstimateAffinePartialLMEDS(from, to []Point) (A [2][3]float64, ok bool) {
	toVec := func(pts []Point) gocv.Point2fVector {
		v := make([]gocv.Point2f, len(pts))
		for i, p := range pts {
			v[i] = gocv.Point2f{X: float32(p.X), Y: float32(p.Y)}
		}
		return gocv.NewPoint2fVectorFromPoints(v)
	}
	f, t := toVec(from), toVec(to)
	defer f.Close()
	defer t.Close()
	inliers := gocv.NewMat()
	defer inliers.Close()
	m := gocv.EstimateAffinePartial2DWithParams(f, t, inliers, int(gocv.HomographyMethodLMEDS), 3.0, 2000, 0.99, 10)
	defer m.Close()
	if m.Empty() || m.Rows() != 2 || m.Cols() != 3 {
		return A, false
	}
	for r := 0; r < 2; r++ {
		for c := 0; c < 3; c++ {
			A[r][c] = m.GetDoubleAt(r, c)
		}
	}
	return A, true
}

// PolygonMedianColour fills the polygon (the integer vertices, cv2.fillPoly with its
// defaults: LINE_8, no shift) into a mask the size of img and returns, per channel, the
// median of the pixels under it truncated to an integer - `int(np.median(img[..., c][mask
// > 0]))` in `_register_card`. ok is false when the polygon covers no pixel (the reference
// then leaves the fill unset).
//
// np.median of an even count is the mean of the two middle values, and int() of that
// truncates: an odd sum of the two loses its half.
func PolygonMedianColour(img Image, poly []image.Point) (fill [3]uint8, ok bool) {
	h, w := img.Height(), img.Width()
	mask := gocv.NewMatWithSize(h, w, gocv.MatTypeCV8U)
	defer mask.Close()
	pv := gocv.NewPointVectorFromPoints(poly)
	defer pv.Close()
	pvs := gocv.NewPointsVector()
	defer pvs.Close()
	pvs.Append(pv)
	if err := gocv.FillPolyWithParams(&mask, pvs, color.RGBA{B: 1}, gocv.Line8, 0, image.Point{}); err != nil {
		return fill, false
	}
	maskBytes := mask.ToBytes()
	pix, err := img.Bytes()
	if err != nil || len(pix) != h*w*3 {
		return fill, false
	}
	var vals [3][]uint8
	for i, mv := range maskBytes {
		if mv == 0 {
			continue
		}
		for c := 0; c < 3; c++ {
			vals[c] = append(vals[c], pix[i*3+c])
		}
	}
	if len(vals[0]) == 0 {
		return fill, false
	}
	for c := 0; c < 3; c++ {
		v := vals[c]
		sort.Slice(v, func(a, b int) bool { return v[a] < v[b] })
		n := len(v)
		if n%2 == 1 {
			fill[c] = v[n/2]
		} else {
			fill[c] = uint8((int(v[n/2-1]) + int(v[n/2])) / 2)
		}
	}
	return fill, true
}

// RoundPolygon is np.int32(np.round(quad)): round half to even, then truncate to int32.
func RoundPolygon(quad []Point) []image.Point {
	out := make([]image.Point, len(quad))
	for i, p := range quad {
		out[i] = image.Point{X: int(math.RoundToEven(p.X)), Y: int(math.RoundToEven(p.Y))}
	}
	return out
}
