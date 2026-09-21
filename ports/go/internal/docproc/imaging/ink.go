package imaging

import (
	"image"
	"image/color"
	"math"

	"gocv.io/x/gocv"
)

// Two "is there anything printed here" measures. Both are thresholds in the reference,
// not values on the wire, so they are compared against a constant and never against a
// golden - which is why float64 accumulation is acceptable here where CONVENTIONS §6.7
// forbids it elsewhere.

// blankInkScalePx is INK_SCALE_PX: the image is downscaled to this longest side before
// the gradient is measured (doc_detector.py:26).
const blankInkScalePx = 400

// SegmentInk is the mean gradient magnitude inside a contour, on the image downscaled to
// blankInkScalePx, with the mask eroded so the segment's own edge does not count.
// Port of doc_detector.segment_ink (doc_detector.py:29-45).
//
// The image is the pipeline's RGB frame; the reference converts RGB->GRAY first.
func SegmentInk(rgb Image, contour []Point) float64 {
	if len(contour) < 3 {
		return 0
	}
	gray := ToGray(rgb)
	defer gray.Close()

	h, w := gray.Height(), gray.Width()
	longest := h
	if w > longest {
		longest = w
	}
	s := float64(blankInkScalePx) / float64(longest)
	sw, sh := int(float64(w)*s), int(float64(h)*s)
	if sw < 1 {
		sw = 1
	}
	if sh < 1 {
		sh = 1
	}
	small := Resize(gray, sw, sh, InterArea)
	defer small.Close()

	// fillPoly on np.int32(np.round(pts * s)): half-to-even rounding of the scaled
	// points, then the polygon rasterised, then a 5x5 erosion.
	pts := make([]image.Point, len(contour))
	for i, p := range contour {
		pts[i] = image.Pt(int(math.RoundToEven(p.X*s)), int(math.RoundToEven(p.Y*s)))
	}
	mask := gocv.NewMatWithSize(sh, sw, gocv.MatTypeCV8U)
	defer mask.Close()
	pv := gocv.NewPointsVectorFromPoints([][]image.Point{pts})
	defer pv.Close()
	if err := gocv.FillPoly(&mask, pv, color.RGBA{R: 255, G: 255, B: 255, A: 255}); err != nil {
		return 0
	}
	// np.ones((5, 5)) is a full rectangle; the default anchor is the centre.
	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(5, 5))
	defer kernel.Close()
	eroded := gocv.NewMat()
	defer eroded.Close()
	if err := gocv.Erode(mask, &eroded, kernel); err != nil {
		return 0
	}
	maskBytes := eroded.ToBytes()
	any := false
	for _, b := range maskBytes {
		if b != 0 {
			any = true
			break
		}
	}
	if !any {
		return 0
	}

	// cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3): default scale 1, delta 0,
	// BORDER_DEFAULT (reflect-101).
	gx := gocv.NewMat()
	defer gx.Close()
	gy := gocv.NewMat()
	defer gy.Close()
	if err := gocv.Sobel(small.Mat(), &gx, gocv.MatTypeCV32F, 1, 0, 3, 1, 0, gocv.BorderDefault); err != nil {
		return 0
	}
	if err := gocv.Sobel(small.Mat(), &gy, gocv.MatTypeCV32F, 0, 1, 3, 1, 0, gocv.BorderDefault); err != nil {
		return 0
	}
	fx := bytesToFloat32s(gx.ToBytes())
	fy := bytesToFloat32s(gy.ToBytes())
	var sum float64
	var n int
	for i := range maskBytes {
		if maskBytes[i] == 0 {
			continue
		}
		sum += math.Hypot(float64(fx[i]), float64(fy[i]))
		n++
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// LaplacianVariance is the variance of the Laplacian of a patch: how much fine detail
// (strokes, not darkness) the strip carries. Port of Pipeline._line_ink
// (pipeline.py:980-991): RGB->GRAY, float32, cv2.Laplacian(..., CV_32F) with the default
// 3x3 aperture, then `.var()` (population variance, two-pass).
func LaplacianVariance(rgb Image) float64 {
	if rgb.Empty() || rgb.Width() == 0 || rgb.Height() == 0 {
		return 0
	}
	gray := ToGray(rgb)
	defer gray.Close()
	f := gocv.NewMat()
	defer f.Close()
	grayMat := gray.Mat()
	grayMat.ConvertTo(&f, gocv.MatTypeCV32F)
	lap := gocv.NewMat()
	defer lap.Close()
	if err := gocv.Laplacian(f, &lap, gocv.MatTypeCV32F, 1, 1, 0, gocv.BorderDefault); err != nil {
		return 0
	}
	v := bytesToFloat32s(lap.ToBytes())
	if len(v) == 0 {
		return 0
	}
	var mean float64
	for _, x := range v {
		mean += float64(x)
	}
	mean /= float64(len(v))
	var acc float64
	for _, x := range v {
		d := float64(x) - mean
		acc += d * d
	}
	return acc / float64(len(v))
}
