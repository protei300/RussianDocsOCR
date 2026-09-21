package imaging

// SIFT feature matching and homography estimation, the OpenCV boundary for
// pipeline_modules/page_registration/page_registration.py. The matching/RANSAC-on-line
// BUSINESS LOGIC lives in the modules package (pure Go); this file only crosses into
// gocv and back, returning plain Go types.

import (
	"image"

	"gocv.io/x/gocv"
)

// SiftDetector owns one cv::SIFT instance (nfeatures budget fixed at construction,
// matching cv2.SIFT_create(nfeatures=...)).
type SiftDetector struct {
	sift gocv.SIFT
}

// NewSiftDetector builds a SIFT detector. nfeatures<=0 means "no cap" (SIFT's own
// default).
func NewSiftDetector(nfeatures int) *SiftDetector {
	if nfeatures <= 0 {
		s := gocv.NewSIFT()
		return &SiftDetector{sift: s}
	}
	n := nfeatures
	s := gocv.NewSIFTWithParams(&n, nil, nil, nil, nil)
	return &SiftDetector{sift: s}
}

func (s *SiftDetector) Close() error { return s.sift.Close() }

// DetectAndCompute finds keypoints and their 128-float descriptors over the whole
// image (no mask), one []float32 row per keypoint, in keypoint order.
func (s *SiftDetector) DetectAndCompute(img Image) ([]Point, [][]float32) {
	mask := gocv.NewMat()
	defer mask.Close()
	return s.detectAndCompute(img, mask)
}

// DetectAndComputeMasked restricts detection to where mask (CV_8U, same size as img,
// nonzero = search here) is nonzero.
func (s *SiftDetector) DetectAndComputeMasked(img Image, mask Image) ([]Point, [][]float32) {
	return s.detectAndCompute(img, mask.mat)
}

func (s *SiftDetector) detectAndCompute(img Image, mask gocv.Mat) ([]Point, [][]float32) {
	kp, desc := s.sift.DetectAndCompute(img.mat, mask)
	defer desc.Close()
	pts := make([]Point, len(kp))
	for i, k := range kp {
		pts[i] = Point{X: k.X, Y: k.Y}
	}
	return pts, descriptorRows(desc, len(kp))
}

// descriptorRows reshapes a CV_32F N x 128 descriptor Mat into N row slices. Empty
// (zero-keypoint) descriptors come back as a nil Mat from gocv, hence the explicit n.
func descriptorRows(desc gocv.Mat, n int) [][]float32 {
	if n == 0 || desc.Empty() {
		return nil
	}
	cols := desc.Cols()
	flat := bytesToFloat32s(desc.ToBytes())
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = flat[i*cols : (i+1)*cols]
	}
	return out
}

// Match is one knnMatch result: the index into the TRAIN set and the L2 distance.
type Match struct {
	TrainIdx int
	Distance float64
}

// L2Matcher wraps cv::BFMatcher(NORM_L2), the descriptor metric SIFT uses.
type L2Matcher struct {
	bf gocv.BFMatcher
}

func NewL2Matcher() *L2Matcher {
	return &L2Matcher{bf: gocv.NewBFMatcherWithParams(gocv.NormL2, false)}
}

func (m *L2Matcher) Close() error { return m.bf.Close() }

// KnnMatch2 returns, for each row of query, its k=2 nearest rows of train (fewer than
// 2 when train has fewer than 2 descriptors) - exactly the shape the ratio test in
// page_registration._match needs.
func (m *L2Matcher) KnnMatch2(query, train [][]float32) [][]Match {
	if len(query) == 0 || len(train) == 0 {
		return make([][]Match, len(query))
	}
	qMat := matFromRows(query)
	defer qMat.Close()
	tMat := matFromRows(train)
	defer tMat.Close()
	raw := m.bf.KnnMatch(qMat, tMat, 2)
	out := make([][]Match, len(raw))
	for i, ms := range raw {
		row := make([]Match, len(ms))
		for j, d := range ms {
			row[j] = Match{TrainIdx: d.TrainIdx, Distance: d.Distance}
		}
		out[i] = row
	}
	return out
}

func matFromRows(rows [][]float32) gocv.Mat {
	n := len(rows)
	cols := len(rows[0])
	buf := make([]byte, n*cols*4)
	for i, r := range rows {
		copy(buf[i*cols*4:], float32sToBytes(r))
	}
	m, err := gocv.NewMatFromBytes(n, cols, gocv.MatTypeCV32F, buf)
	if err != nil {
		return gocv.NewMat()
	}
	return m
}

// usacMagsac is cv::USAC_MAGSAC (calib3d.hpp:560). gocv's HomographyMethod enum does
// not name it, but the C++ wrapper (calib3d.cpp:248-254) passes `method int` straight
// through to cv::findHomography with no range check, so the raw value works exactly
// as cv2.findHomography(..., cv2.USAC_MAGSAC, ...) does. Verified by reading the
// installed OpenCV 4.13 header, not by a prior working call - see the port's task log
// for the single-image check this was confirmed against.
const usacMagsac = gocv.HomographyMethod(38)

// FindHomographyMagsac estimates the src->dst homography with cv::USAC_MAGSAC.
// Mirrors PageRegistrar._match's call: `cv2.findHomography(src, dst, cv2.USAC_MAGSAC,
// reproj, maxIters=10000, confidence=0.999)`. ok is false when OpenCV found none (too
// few points, degenerate configuration).
func FindHomographyMagsac(src, dst []Point, reprojThresh float64, maxIters int,
	confidence float64) (H [3][3]float64, inlierMask []bool, ok bool) {

	if len(src) != len(dst) || len(src) < 4 {
		return H, nil, false
	}
	srcMat := matFromPoints(src)
	defer srcMat.Close()
	dstMat := matFromPoints(dst)
	defer dstMat.Close()
	maskMat := gocv.NewMat()
	defer maskMat.Close()

	hMat := gocv.FindHomography(srcMat, dstMat, usacMagsac, reprojThresh, &maskMat, maxIters, confidence)
	defer hMat.Close()
	if hMat.Empty() || hMat.Rows() != 3 || hMat.Cols() != 3 {
		return H, nil, false
	}
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			H[r][c] = hMat.GetDoubleAt(r, c)
		}
	}
	n := maskMat.Rows() * maskMat.Cols()
	mb := maskMat.ToBytes()
	inlierMask = make([]bool, n)
	for i := range inlierMask {
		inlierMask[i] = i < len(mb) && mb[i] != 0
	}
	return H, inlierMask, true
}

func matFromPoints(pts []Point) gocv.Mat {
	pv := toPoint2fVector(pts)
	defer pv.Close()
	return gocv.NewMatFromPoint2fVector(pv, true)
}

// WarpByHomography warps src through the 3x3 matrix H (forward mapping: a source
// pixel at p lands at H(p) in the destination), matching
// cv2.warpPerspective(src, H, (width, height), flags=INTER_LINEAR,
// borderMode=BORDER_REPLICATE when replicate else BORDER_CONSTANT).
func WarpByHomography(src Image, H [3][3]float64, width, height int, replicate bool) Image {
	m := gocv.NewMatWithSize(3, 3, gocv.MatTypeCV64F)
	defer m.Close()
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			m.SetDoubleAt(r, c, H[r][c])
		}
	}
	border := gocv.BorderConstant
	if replicate {
		border = gocv.BorderReplicate
	}
	dst := gocv.NewMat()
	gocv.WarpPerspectiveWithParams(src.mat, &dst, m, image.Pt(width, height),
		gocv.InterpolationLinear, border, blackScalar())
	return Image{mat: dst}
}
