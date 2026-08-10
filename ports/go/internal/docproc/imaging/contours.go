package imaging

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"gocv.io/x/gocv"
)

// Point is a float32 2-D point. Contours are integral by construction (findContours
// yields integer pixels; the reference merely views them as float32 without changing
// any value), but quads become fractional once expand_quad scales them — so the type
// carries floats throughout and only the OpenCV calls that require integers get them.
type Point struct {
	X, Y float64
}

// ThresholdOtsu binarises with an automatically chosen threshold and returns the value
// it picked.
//
// The chosen value is an INTEGER, which makes it an unusually unambiguous thing to
// compare across implementations — the conformance suite uses it as the first check on
// the deskew path.
func ThresholdOtsu(src Image, invert bool) (Image, float64) {
	typ := gocv.ThresholdOtsu
	if invert {
		typ |= gocv.ThresholdBinaryInv
	} else {
		typ |= gocv.ThresholdBinary
	}
	dst := gocv.NewMat()
	t := gocv.Threshold(src.mat, &dst, 0, 255, typ)
	return Image{mat: dst}, float64(t)
}

// FindExternalContours returns the outer contours of a binary mask, largest first is
// NOT guaranteed — callers that need the largest ask for it explicitly, as the
// reference does.
func FindExternalContours(src Image) [][]Point {
	cs := gocv.FindContours(src.mat, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer cs.Close()

	out := make([][]Point, 0, cs.Size())
	for i := 0; i < cs.Size(); i++ {
		pv := cs.At(i)
		pts := make([]Point, 0, pv.Size())
		for j := 0; j < pv.Size(); j++ {
			p := pv.At(j)
			pts = append(pts, Point{X: float64(p.X), Y: float64(p.Y)})
		}
		out = append(out, pts)
	}
	return out
}

// LargestContour returns the contour with the most points, or nil.
//
// "Most points" and not "largest area": the reference selects with
// `argmax([len(x) for x in c])`, which is a different criterion and occasionally a
// different contour. Reproduced as written.
func LargestContour(contours [][]Point) []Point {
	best := -1
	bestLen := -1
	for i, c := range contours {
		if len(c) > bestLen {
			best, bestLen = i, len(c)
		}
	}
	if best < 0 {
		return nil
	}
	return contours[best]
}

// ConvexHull returns the convex hull of a contour.
//
// Integer points are handed to OpenCV here, which is faithful rather than lossy: the
// reference views the contour as float32 but its values come from findContours and are
// whole numbers. The first fractional coordinate in this pipeline appears only after
// ExpandQuad, downstream of every call in this function.
func ConvexHull(pts []Point) []Point {
	pv := toPointVector(pts)
	defer pv.Close()

	hull := gocv.NewMat()
	defer hull.Close()
	// clockwise=false, matching cv2.convexHull's default. Not cosmetic: the hull's
	// ORIENTATION decides which vertices Douglas-Peucker keeps in ExtractQuad, so
	// passing true yields a different four-point quad on some contours — measured as a
	// 6 px canvas-width difference on an internal-passport spread.
	if err := gocv.ConvexHull(pv, &hull, false, true); err != nil {
		return nil
	}
	// The hull comes back as an Nx1 CV_32SC2 matrix.
	out := make([]Point, 0, hull.Rows())
	for i := 0; i < hull.Rows(); i++ {
		out = append(out, Point{
			X: float64(hull.GetIntAt(i, 0)),
			Y: float64(hull.GetIntAt(i, 1)),
		})
	}
	return out
}

// ArcLength is the closed perimeter of a contour.
func ArcLength(pts []Point) float64 {
	pv := toPointVector(pts)
	defer pv.Close()
	return gocv.ArcLength(pv, true)
}

// ApproxPolyDP is Douglas-Peucker simplification with a closed curve.
func ApproxPolyDP(pts []Point, epsilon float64) []Point {
	pv := toPointVector(pts)
	defer pv.Close()

	approx := gocv.ApproxPolyDP(pv, epsilon, true)
	defer approx.Close()

	out := make([]Point, 0, approx.Size())
	for i := 0; i < approx.Size(); i++ {
		p := approx.At(i)
		out = append(out, Point{X: float64(p.X), Y: float64(p.Y)})
	}
	return out
}

// MinAreaRectPoints is the four corners of the minimum-area rotated rectangle.
//
// This is the fallback in ExtractQuad, and unlike everything above it returns
// FRACTIONAL coordinates. Note also that OpenCV changed boxPoints' corner ORDER around
// the 4.5 series; the ordering does not matter here because OrderPoints canonicalises
// it immediately afterwards, which is exactly why the reference is safe too.
func MinAreaRectPoints(pts []Point) []Point {
	pv := toPointVector(pts)
	defer pv.Close()

	rect := gocv.MinAreaRect(pv)
	box := gocv.NewMat()
	defer box.Close()
	if err := gocv.BoxPoints(rect, &box); err != nil {
		return nil
	}
	out := make([]Point, 0, box.Rows())
	for i := 0; i < box.Rows(); i++ {
		out = append(out, Point{
			X: float64(box.GetFloatAt(i, 0)),
			Y: float64(box.GetFloatAt(i, 1)),
		})
	}
	return out
}

// WarpPerspectiveQuad maps a fractional source quad onto a width x height rectangle.
//
// Uses GetPerspectiveTransform2f rather than the integer variant: by this point the
// quad has been through ExpandQuad and clipping, so its corners are fractional, and
// rounding them would move the crop edge by up to a pixel on every document.
func WarpPerspectiveQuad(src Image, quad []Point, width, height int) (Image, error) {
	if len(quad) != 4 {
		return Image{}, fmt.Errorf("imaging: perspective transform needs 4 points, got %d", len(quad))
	}
	from := toPoint2fVector(quad)
	defer from.Close()
	to := toPoint2fVector([]Point{
		{0, 0},
		{float64(width - 1), 0},
		{float64(width - 1), float64(height - 1)},
		{0, float64(height - 1)},
	})
	defer to.Close()

	m := gocv.GetPerspectiveTransform2f(from, to)
	defer m.Close()

	dst := gocv.NewMat()
	if err := gocv.WarpPerspective(src.mat, &dst, m, image.Pt(width, height)); err != nil {
		dst.Close()
		return Image{}, fmt.Errorf("imaging: warpPerspective: %w", err)
	}
	return Image{mat: dst}, nil
}

// RotationMatrix2D builds OpenCV's 2x3 affine rotation matrix by hand.
//
// gocv's GetRotationMatrix2D takes an image.Point and therefore CANNOT express a
// fractional centre — but DocDeskewer rotates about (w/2.0, h/2.0), which is fractional
// for any odd dimension. Measured cost of rounding it: the deskew variance array shifts
// by up to 3.8e-3 relative, above the 1e-3 conformance policy. See DEVIATIONS D-08.
//
// OpenCV's own formula, positive angle meaning counter-clockwise:
//
//	alpha = scale*cos(a), beta = scale*sin(a)
//	[  alpha  beta   (1-alpha)*cx - beta*cy ]
//	[ -beta   alpha   beta*cx + (1-alpha)*cy ]
//
// Verified against cv2.getRotationMatrix2D to 1.6e-14.
func RotationMatrix2D(cx, cy, angleDeg, scale float64) gocv.Mat {
	a := angleDeg * math.Pi / 180.0
	alpha := scale * math.Cos(a)
	beta := scale * math.Sin(a)
	m := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV64F)
	m.SetDoubleAt(0, 0, alpha)
	m.SetDoubleAt(0, 1, beta)
	m.SetDoubleAt(0, 2, (1-alpha)*cx-beta*cy)
	m.SetDoubleAt(1, 0, -beta)
	m.SetDoubleAt(1, 1, alpha)
	m.SetDoubleAt(1, 2, beta*cx+(1-alpha)*cy)
	return m
}

// BorderMode selects how WarpAffine fills area rotated in from outside the image.
type BorderMode int

const (
	// BorderConstantZero fills with zeros. The deskew angle SCAN uses this, so that
	// rotated-in area contributes no text pixels and the projection variance stays
	// meaningful.
	BorderConstantZero BorderMode = iota
	// BorderReplicate extends edge pixels. The FINAL deskew rotation uses this, to
	// avoid introducing black wedges into the canvas that field detection then sees.
	BorderReplicate
)

// WarpAffine rotates with an explicit interpolation and border policy.
//
// The two callers deliberately differ: the angle scan uses nearest-neighbour with zero
// borders (it is measuring a binary mask, and interpolation would invent grey), while
// the final rotation uses linear with edge replication.
func WarpAffine(src Image, m gocv.Mat, width, height int,
	nearest bool, border BorderMode) Image {

	interp := gocv.InterpolationLinear
	if nearest {
		interp = gocv.InterpolationNearestNeighbor
	}
	mode := gocv.BorderConstant
	if border == BorderReplicate {
		mode = gocv.BorderReplicate
	}
	dst := gocv.NewMat()
	gocv.WarpAffineWithParams(src.mat, &dst, m, image.Pt(width, height),
		interp, mode, blackScalar())
	return Image{mat: dst}
}

// HStack and VStack join two images along an axis, for the two-page passport spread.
func HStack(a, b Image) (Image, error) {
	dst := gocv.NewMat()
	if err := gocv.Hconcat(a.mat, b.mat, &dst); err != nil {
		dst.Close()
		return Image{}, fmt.Errorf("imaging: hconcat: %w", err)
	}
	return Image{mat: dst}, nil
}

func VStack(a, b Image) (Image, error) {
	dst := gocv.NewMat()
	if err := gocv.Vconcat(a.mat, b.mat, &dst); err != nil {
		dst.Close()
		return Image{}, fmt.Errorf("imaging: vconcat: %w", err)
	}
	return Image{mat: dst}, nil
}

// RowSums returns the per-row sum of a single-channel 8-bit image, as int64.
//
// int64 and not float: numpy's `.sum(axis=1)` on a uint8 array produces exact integers,
// and the only float-sensitive step in the deskew score is the variance that follows.
// Accumulating here in float would introduce error the reference does not have.
func RowSums(src Image) ([]int64, error) {
	if src.mat.Type() != gocv.MatTypeCV8U {
		return nil, fmt.Errorf("imaging: RowSums needs CV_8U, got %v", src.mat.Type())
	}
	h, w := src.mat.Rows(), src.mat.Cols()
	buf := src.mat.ToBytes()
	if len(buf) != h*w {
		return nil, fmt.Errorf("imaging: %d bytes for %dx%d (non-continuous Mat?)", len(buf), h, w)
	}
	out := make([]int64, h)
	for y := 0; y < h; y++ {
		var s int64
		base := y * w
		for x := 0; x < w; x++ {
			s += int64(buf[base+x])
		}
		out[y] = s
	}
	return out, nil
}

func toPointVector(pts []Point) gocv.PointVector {
	ip := make([]image.Point, len(pts))
	for i, p := range pts {
		ip[i] = image.Pt(int(p.X), int(p.Y))
	}
	return gocv.NewPointVectorFromPoints(ip)
}

func toPoint2fVector(pts []Point) gocv.Point2fVector {
	fp := make([]gocv.Point2f, len(pts))
	for i, p := range pts {
		fp[i] = gocv.Point2f{X: float32(p.X), Y: float32(p.Y)}
	}
	return gocv.NewPoint2fVectorFromPoints(fp)
}

func blackScalar() color.RGBA { return color.RGBA{} }
