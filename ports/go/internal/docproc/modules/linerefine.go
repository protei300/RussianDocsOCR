package modules

import (
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// Port of pipeline_modules/page_registration/line_refine.py: straightens a rectified
// page by its own straight structures (LSD segments + projection-profile band tilts),
// fitting a 4-parameter model (rotation, shear, two perspective terms) with a robust
// (Cauchy-weighted IRLS) loss.
//
// Two substitutions from the reference, both forced by the Go OpenCV binding rather
// than chosen for convenience - see their definitions for the measured consequence:
//   - imaging.DetectSegments (Canny+HoughLinesP) stands in for cv2.createLineSegmentDetector,
//     which gocv v0.43.0 does not bind at all.
//   - the 4-parameter fit uses LevenbergMarquardt (this port's own, finite-difference
//     Jacobian) rather than scipy.optimize.least_squares's trust-region algorithm.

const (
	lsdMinLenFrac    = 0.04
	lsdMaxWeight     = 3.0
	angleTolDeg      = 12.0
	profileScale     = 0.5
	profileFineStep  = 0.25
	profileMinPeak   = 1.15
	profileWeight    = 3.0
	minHorizWeight   = 4.0
	maxRotDeg        = 5.0
	maxCornerShiftFr = 0.08
	refineMinGain    = 0.25
	minResidualDeg   = 0.3
	irlsIters        = 3
	irlsScaleDeg     = 1.0
	vertMinLenFrac   = 0.15
	blobMaxFrac      = 0.2
	nBands           = 5
)

var priorScale = [3]float64{0.03, 0.05, 0.03}
var profileCoarse = coarseAngles()

func coarseAngles() []float64 {
	var out []float64
	for a := -6.0; a <= 6.0001; a += 1.0 {
		out = append(out, a)
	}
	return out
}

// measureRow is one row of line_refine.measure()'s (x1,y1,x2,y2,kind,weight,source).
// kind 0 = should be horizontal, 1 = should be vertical.
type measureRow struct {
	X1, Y1, X2, Y2 float64
	Kind           int
	Weight         float64
	Source         int
}

func pymod(a, b float64) float64 {
	m := math.Mod(a, b)
	if m < 0 {
		m += b
	}
	return m
}

func segAngle(x1, y1, x2, y2 float64) float64 {
	ang := math.Atan2(y2-y1, x2-x1) * 180 / math.Pi
	return pymod(ang+90, 180) - 90
}

// otsuInvDropTallBlobs binarises (Otsu, inverted - ink=255) and drops connected
// components taller than blobMaxHeightPx, matching _profile_tilt's/‌_cell_tilts' blob
// filter. Returns the raw row-major bytes (0/255) alongside w, h.
func otsuInvDropTallBlobs(gray imaging.Image, blobMaxHeightPx float64) ([]byte, int, int) {
	binary, _ := imaging.ThresholdOtsu(gray, true)
	defer binary.Close()
	w, h := binary.Width(), binary.Height()
	labels, lw, lh, blobs := imaging.ConnectedComponents8(binary)
	buf, err := binary.Bytes()
	if err != nil || lw != w || lh != h {
		out := make([]byte, w*h)
		copy(out, buf)
		return out, w, h
	}
	out := append([]byte(nil), buf...)
	big := make([]bool, len(blobs))
	for i := 1; i < len(blobs); i++ { // index 0 is background, never dropped
		if float64(blobs[i].H) > blobMaxHeightPx {
			big[i] = true
		}
	}
	anyBig := false
	for _, b := range big {
		anyBig = anyBig || b
	}
	if anyBig {
		for i, lab := range labels {
			if lab >= 0 && int(lab) < len(big) && big[lab] {
				out[i] = 0
			}
		}
	}
	return out, w, h
}

// profileTilt is _profile_tilt (axis=1 only - the only axis line_refine/line_dewarp
// ever use). region is a grayscale crop; returns (tilt degrees, peak ratio); ratio 0
// means "no evidence".
func profileTilt(region imaging.Image) (float64, float64) {
	small := imaging.ResizeArea(region, maxInt(1, int(float64(region.Width())*profileScale)),
		maxInt(1, int(float64(region.Height())*profileScale)))
	defer small.Close()
	w, h := small.Width(), small.Height()
	ink, w2, h2 := otsuInvDropTallBlobs(small, blobMaxFrac*float64(h))
	if w2 != w || h2 != h {
		return 0, 0
	}
	inkImg, err := imaging.NewGrayFromBytes(ink, w, h)
	if err != nil {
		return 0, 0
	}
	defer inkImg.Close()
	validImg := imaging.NewGrayFilled(h, w, 255)
	defer validImg.Close()

	score := func(angles []float64) []float64 {
		out := make([]float64, len(angles))
		for i, a := range angles {
			rot := imaging.RotateNearestZero(inkImg, a)
			cnt := imaging.RotateNearestZero(validImg, a)
			inkSum, errI := imaging.RowSums(rot)
			cntSum, errC := imaging.RowSums(cnt)
			rot.Close()
			cnt.Close()
			if errI != nil || errC != nil {
				out[i] = 0
				continue
			}
			var cmax int64
			for _, c := range cntSum {
				if c > cmax {
					cmax = c
				}
			}
			var prof []float64
			for r := range cntSum {
				if cntSum[r] >= int64(0.6*float64(cmax)) {
					denom := cntSum[r]
					if denom < 1 {
						denom = 1
					}
					prof = append(prof, float64(inkSum[r])/float64(denom))
				}
			}
			if len(prof) > 2 {
				out[i] = varianceF64(prof)
			}
		}
		return out
	}

	coarse := score(profileCoarse)
	ib := argmaxFloat(coarse)
	if ib == 0 || ib == len(profileCoarse)-1 || coarse[ib] <= 0 {
		return 0, 0
	}
	var fine []float64
	for a := profileCoarse[ib] - 1.0; a <= profileCoarse[ib]+1.0001; a += profileFineStep {
		fine = append(fine, a)
	}
	fs := score(fine)
	jb := argmaxFloat(fs)
	best := fine[jb]
	if jb > 0 && jb < len(fine)-1 {
		y0, y1, y2 := fs[jb-1], fs[jb], fs[jb+1]
		den := y0 - 2*y1 + y2
		if den < 0 {
			best += profileFineStep * 0.5 * (y0 - y2) / den
		}
	}
	med := medianOf(coarse)
	ratio := 0.0
	if med > 0 {
		ratio = fs[jb] / med
	}
	return best, ratio
}

func varianceF64(v []float64) float64 {
	var mean float64
	for _, x := range v {
		mean += x
	}
	mean /= float64(len(v))
	var acc float64
	for _, x := range v {
		d := x - mean
		acc += d * d
	}
	return acc / float64(len(v))
}

func medianOf(v []float64) float64 {
	s := append([]float64(nil), v...)
	for i := 1; i < len(s); i++ {
		j := i
		for j > 0 && s[j-1] > s[j] {
			s[j-1], s[j] = s[j], s[j-1]
			j--
		}
	}
	n := len(s)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return s[n/2]
	}
	return 0.5 * (s[n/2-1] + s[n/2])
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// measure is line_refine.measure.
func measure(gray imaging.Image, inset int) []measureRow {
	H, W := float64(gray.Height()), float64(gray.Width())
	var rows []measureRow

	segs := imaging.DetectSegments(gray, lsdMinLenFrac*W)
	for _, s := range segs {
		l := math.Hypot(s.X2-s.X1, s.Y2-s.Y1)
		a := segAngle(s.X1, s.Y1, s.X2, s.Y2)
		w := math.Min(l/(0.1*W), lsdMaxWeight)
		if math.Abs(a) < angleTolDeg {
			rows = append(rows, measureRow{s.X1, s.Y1, s.X2, s.Y2, 0, w, 0})
		} else if math.Abs(math.Abs(a)-90) < angleTolDeg && l >= vertMinLenFrac*H {
			rows = append(rows, measureRow{s.X1, s.Y1, s.X2, s.Y2, 1, w, 0})
		}
	}

	x0, y0 := float64(inset), float64(inset)
	x1, y1 := W-float64(inset), H-float64(inset)
	if x1-x0 < 60 || y1-y0 < 60 {
		return rows
	}
	bh := (y1 - y0) / 3.0
	for _, yb := range linspace(y0, y1-bh, nBands) {
		top, bottom := int(yb), int(yb+bh)
		left, right := int(x0), int(x1)
		if bottom <= top || right <= left {
			continue
		}
		band, err := imaging.ClampedCrop(gray, left, top, right, bottom)
		if err != nil {
			continue
		}
		tilt, ratio := profileTilt(band)
		band.Close()
		if ratio >= profileMinPeak {
			yc := yb + 0.5*bh
			half := 0.5 * (x1 - x0)
			dy := math.Tan(tilt*math.Pi/180) * half
			w := profileWeight * math.Min(1.0, ratio-1.0)
			rows = append(rows, measureRow{x0, yc - dy, x1, yc + dy, 0, w, 1})
		}
	}
	return rows
}

// refineModel is line_refine._model.
func refineModel(params [4]float64, W, H float64) [3][3]float64 {
	th, s, p1, p2 := params[0], params[1], params[2], params[3]
	k := 2.0 / W
	T := [3][3]float64{{k, 0, -1}, {0, k, -H / W}, {0, 0, 1}}
	R := [3][3]float64{{math.Cos(th), -math.Sin(th), 0}, {math.Sin(th), math.Cos(th), 0}, {0, 0, 1}}
	S := [3][3]float64{{1, s, 0}, {0, 1, 0}, {0, 0, 1}}
	P := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {p1, p2, 1}}
	invT, ok := imaging.InvertH(T)
	if !ok {
		return imaging.Identity3()
	}
	return imaging.MulH(invT, imaging.MulH(P, imaging.MulH(R, imaging.MulH(S, T))))
}

// anglesAfter is line_refine._angles_after.
func anglesAfter(Hm [3][3]float64, meas []measureRow) []float64 {
	out := make([]float64, len(meas))
	for i, m := range meas {
		p0 := imaging.TransformPoint(Hm, imaging.Point{X: m.X1, Y: m.Y1})
		p1 := imaging.TransformPoint(Hm, imaging.Point{X: m.X2, Y: m.Y2})
		ang := math.Atan2(p1.Y-p0.Y, p1.X-p0.X) * 180 / math.Pi
		ang = pymod(ang+90, 180) - 90
		if m.Kind == 0 {
			out[i] = ang
		} else {
			out[i] = math.Abs(ang) - 90
		}
	}
	return out
}

func refineResiduals(params []float64, meas []measureRow, W, H float64, wgt []float64) []float64 {
	var p4 [4]float64
	copy(p4[:], params)
	dev := anglesAfter(refineModel(p4, W, H), meas)
	out := make([]float64, 0, len(meas)+3)
	for i, d := range dev {
		out = append(out, d*math.Sqrt(wgt[i]))
	}
	out = append(out, params[1]/priorScale[0], params[2]/priorScale[1], params[3]/priorScale[2])
	return out
}

func fitRefineParams(meas []measureRow, W, H float64) [4]float64 {
	x := []float64{0, 0, 0, 0}
	wgt := make([]float64, len(meas))
	for i, m := range meas {
		wgt[i] = m.Weight
	}
	for iter := 0; iter < irlsIters; iter++ {
		wgtCopy := append([]float64(nil), wgt...)
		x = LevenbergMarquardt(x, func(p []float64) []float64 {
			return refineResiduals(p, meas, W, H, wgtCopy)
		}, 100)
		var p4 [4]float64
		copy(p4[:], x)
		r := anglesAfter(refineModel(p4, W, H), meas)
		for i, m := range meas {
			wgt[i] = m.Weight / (1.0 + (r[i]/irlsScaleDeg)*(r[i]/irlsScaleDeg))
		}
	}
	var out [4]float64
	copy(out[:], x)
	return out
}

// StraightenInfo is diagnostic only (not part of the conformance contract - no stage
// reads it); kept small on purpose.
type StraightenInfo struct {
	Applied bool
	Reason  string
}

// RefineByLines is line_refine.refine_by_lines: the homography that straightens a
// rectified page by its own lines, or nil when there is not enough evidence or the
// correction is not warranted.
func RefineByLines(gray imaging.Image, inset int) (*[3][3]float64, StraightenInfo) {
	H, W := float64(gray.Height()), float64(gray.Width())
	meas := measure(gray, inset)
	if len(meas) == 0 {
		return nil, StraightenInfo{Reason: "no evidence"}
	}
	horizW := 0.0
	for _, m := range meas {
		if m.Kind == 0 {
			horizW += m.Weight
		}
	}
	if horizW < minHorizWeight {
		return nil, StraightenInfo{Reason: "too little horizontal evidence"}
	}
	weights := make([]float64, len(meas))
	for i, m := range meas {
		weights[i] = m.Weight
	}
	before0 := anglesAfter(imaging.Identity3(), meas)
	beforeAbs := make([]float64, len(before0))
	for i, v := range before0 {
		beforeAbs[i] = math.Abs(v)
	}
	before := wmedian(beforeAbs, weights)

	x := fitRefineParams(meas, W, H)
	Hm := refineModel(x, W, H)
	after0 := anglesAfter(Hm, meas)
	afterAbs := make([]float64, len(after0))
	for i, v := range after0 {
		afterAbs[i] = math.Abs(v)
	}
	after := wmedian(afterAbs, weights)

	if before < minResidualDeg {
		return nil, StraightenInfo{Reason: "already straight"}
	}
	if after > (1.0-refineMinGain)*before {
		return nil, StraightenInfo{Reason: "no gain"}
	}
	rotDeg := x[0] * 180 / math.Pi
	if math.Abs(rotDeg) > maxRotDeg {
		return nil, StraightenInfo{Reason: "rotation too large"}
	}
	corners := []imaging.Point{
		{X: float64(inset), Y: float64(inset)}, {X: W - float64(inset), Y: float64(inset)},
		{X: W - float64(inset), Y: H - float64(inset)}, {X: float64(inset), Y: H - float64(inset)},
	}
	moved := imaging.TransformPoints(Hm, corners)
	shift := 0.0
	for i := range corners {
		d := dist(corners[i], moved[i])
		if d > shift {
			shift = d
		}
	}
	if shift > maxCornerShiftFr*W {
		return nil, StraightenInfo{Reason: "correction too large"}
	}
	return &Hm, StraightenInfo{Applied: true}
}

// ApplyRefinement warps a page through the straightening homography, matching
// line_refine.apply_refinement (linear interpolation, replicate border).
func ApplyRefinement(page imaging.Image, Hm [3][3]float64) imaging.Image {
	return imaging.WarpByHomography(page, Hm, page.Width(), page.Height(), true)
}
