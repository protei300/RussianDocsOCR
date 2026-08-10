package modules

import (
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// DocDeskewer removes residual tilt after the perspective fix, so text lines end up
// horizontal. Port of pipeline_modules/deskewer/deskewer.py.
//
// Needs no model: it is a projection-profile angle search.
type DocDeskewer struct {
	angleRange float64
	angleSteps int
	minAngle   float64
	scale      float64

	coarseAngles  []float64
	fineHalfRange float64
	fineCount     int
}

// NewPipelineDeskewer builds the deskewer with the parameters the PIPELINE passes, which
// differ from the class's own defaults — `angle_range` is 10.0 here versus 2.0 there.
// Using the class defaults would search a fifth of the range and miss most real tilt.
func NewPipelineDeskewer() *DocDeskewer {
	return NewDocDeskewer(10.0, 101, 2.0, 0.4, 21)
}

func NewDocDeskewer(angleRange float64, angleSteps int, minAngle, scale float64,
	coarseSteps int) *DocDeskewer {

	d := &DocDeskewer{angleRange: angleRange, angleSteps: angleSteps,
		minAngle: minAngle, scale: scale}

	cs := coarseSteps
	if cs < 3 {
		cs = 3
	}
	if cs > angleSteps {
		cs = angleSteps
	}
	d.coarseAngles = linspace(-angleRange, angleRange, cs)

	coarseStep := (2 * angleRange) / float64(cs-1)
	fullRes := (2 * angleRange) / float64(angleSteps-1)
	d.fineHalfRange = coarseStep
	d.fineCount = int(math.Round(2*coarseStep/fullRes)) + 1
	if d.fineCount < 3 {
		d.fineCount = 3
	}
	return d
}

// Deskew rotates the tilt out, or returns a clone unchanged when there is nothing worth
// correcting.
//
// The two-page branch of the reference (`n_segments=2`) is not ported: the pipeline
// always calls `deskew(img)` with the default, so that branch is unreachable from
// production. Noted rather than silently omitted.
func (d *DocDeskewer) Deskew(img imaging.Image) (imaging.Image, float64, error) {
	angle, err := d.findAngle(img)
	if err != nil {
		return imaging.Image{}, 0, err
	}
	// Below min_angle the estimate is noise, not tilt — handwriting and textured
	// backgrounds routinely produce a spurious degree or two, and rotating on that
	// resamples the whole canvas for nothing.
	if math.Abs(angle) < d.minAngle {
		return img.Clone(), angle, nil
	}
	m := imaging.RotationMatrix2D(float64(img.Width())/2.0, float64(img.Height())/2.0, angle, 1.0)
	defer m.Close()
	// Linear interpolation and edge REPLICATION for the final rotation: constant-zero
	// borders would introduce black wedges that field detection then sees as content.
	return imaging.WarpAffine(img, m, img.Width(), img.Height(), false, imaging.BorderReplicate),
		angle, nil
}

// findAngle scores candidate angles by the variance of the horizontal projection
// profile, coarse then fine.
//
// Three things are load-bearing:
//
//   - **The boundary early-out.** If the coarse maximum lands on either end of the
//     range, the function returns 0.0 without refining: no clear peak means a probable
//     false detection from a stamp, a hologram or a textured background. Behaviour, not
//     an optimisation.
//   - **Two-pass variance.** Row sums of a uint8 image are exact integers; the only
//     float-sensitive step is the variance, and numpy computes it two-pass. A one-pass
//     E[x^2]-E[x]^2 loses about seven significant digits at magnitudes of 255*W, which
//     is enough to flip the argmax between adjacent angles and hence change the chosen
//     rotation.
//   - **A fractional rotation centre**, which gocv's own GetRotationMatrix2D cannot
//     express — hence imaging.RotationMatrix2D (DEVIATIONS D-08).
func (d *DocDeskewer) findAngle(img imaging.Image) (float64, error) {
	gray := imaging.ToGray(img)
	defer gray.Close()

	sh := int(float64(gray.Height()) * d.scale)
	sw := int(float64(gray.Width()) * d.scale)
	if sh < 1 {
		sh = 1
	}
	if sw < 1 {
		sw = 1
	}
	// INTER_AREA for the downscale: it averages, which preserves the density of text
	// strokes that the projection profile measures.
	small := imaging.Resize(gray, sw, sh, imaging.InterArea)
	defer small.Close()

	// Inverted Otsu, so TEXT pixels become 255 and the row sums measure ink.
	binary, _ := imaging.ThresholdOtsu(small, true)
	defer binary.Close()

	cx, cy := float64(sw)/2.0, float64(sh)/2.0

	coarse, err := d.scoreAngles(binary, sw, sh, cx, cy, d.coarseAngles)
	if err != nil {
		return 0, err
	}
	ci := argmaxFloat(coarse)
	if ci == 0 || ci == len(d.coarseAngles)-1 {
		return 0.0, nil
	}

	best := d.coarseAngles[ci]
	lo := math.Max(-d.angleRange, best-d.fineHalfRange)
	hi := math.Min(d.angleRange, best+d.fineHalfRange)
	fineAngles := linspace(lo, hi, d.fineCount)
	fine, err := d.scoreAngles(binary, sw, sh, cx, cy, fineAngles)
	if err != nil {
		return 0, err
	}
	return fineAngles[argmaxFloat(fine)], nil
}

func (d *DocDeskewer) scoreAngles(binary imaging.Image, sw, sh int, cx, cy float64,
	angles []float64) ([]float64, error) {

	out := make([]float64, len(angles))
	for i, a := range angles {
		m := imaging.RotationMatrix2D(cx, cy, a, 1.0)
		// Nearest-neighbour and zero borders: the input is a binary mask, so
		// interpolation would invent grey values, and rotated-in area must contribute
		// no ink.
		rot := imaging.WarpAffine(binary, m, sw, sh, true, imaging.BorderConstantZero)
		rows, err := imaging.RowSums(rot)
		_ = rot.Close()
		_ = m.Close()
		if err != nil {
			return nil, err
		}
		out[i] = variance(rows)
	}
	return out, nil
}

// variance is numpy's two-pass computation: subtract the mean, then average the squares.
func variance(v []int64) float64 {
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

// argmaxFloat returns the FIRST maximum, like numpy.argmax. Strict `>` only: on a tie
// between two adjacent angles, `>=` picks the later one and rotates the canvas
// differently.
func argmaxFloat(v []float64) int {
	best, bestV := 0, math.Inf(-1)
	for i, x := range v {
		if x > bestV {
			best, bestV = i, x
		}
	}
	return best
}

// linspace matches numpy.linspace: endpoints inclusive, and the final element assigned
// exactly rather than accumulated, so the range end is not drifted by rounding.
func linspace(lo, hi float64, n int) []float64 {
	if n <= 1 {
		return []float64{lo}
	}
	out := make([]float64, n)
	step := (hi - lo) / float64(n-1)
	for i := range out {
		out[i] = lo + step*float64(i)
	}
	out[n-1] = hi
	return out
}
