package modules

import (
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// Port of pipeline_modules/page_registration/line_dewarp.py: bend correction from the
// LOCAL tilt of a page's text lines (a homography, see line_refine, only makes a FLAT
// page straight - a booklet page bent near the spine needs a per-pixel vertical
// displacement instead).
//
// Simplified relative to the reference in one place, deliberately: Python's
// `_cell_tilts` rotates the WHOLE region once per angle and slices every cell's sum out
// of a cumulative sum (a performance optimisation - 25 cells x 22 angles would
// otherwise be 550 small warps). This port instead calls profileTilt (line_refine.go)
// PER CELL directly, which is simpler, reuses already-verified code, and is if
// anything MORE faithful to the per-cell blob-height threshold (BLOB_MAX_FRAC * the
// CELL's own height) than trying to thread that through a shared whole-region
// rotation. Slower, not wrong; page counts here are always 1-2 per document.

const (
	dewarpGrid     = 5
	minCells       = 8
	minDispPx      = 3.0
	maxDispFrac    = 0.04
	dewarpMinGain  = 0.25
	ridge          = 1e-3
)

// dewarpRow is measure_cells's (x, y, tilt_deg, weight, source) - source 0 = LSD
// segment centre, 1 = profile cell centre.
type dewarpRow struct {
	X, Y, Tilt, Weight float64
	Source             int
}

// measureCells is line_dewarp.measure_cells.
func measureCells(gray imaging.Image, inset int) []dewarpRow {
	H, W := float64(gray.Height()), float64(gray.Width())
	var rows []dewarpRow

	half := imaging.ResizeArea(gray, maxInt(1, gray.Width()/2), maxInt(1, gray.Height()/2))
	segs := imaging.DetectSegments(half, lsdMinLenFrac*W*0.5)
	half.Close()
	for _, s := range segs {
		x1, y1, x2, y2 := s.X1*2, s.Y1*2, s.X2*2, s.Y2*2
		l := math.Hypot(x2-x1, y2-y1)
		a := segAngle(x1, y1, x2, y2)
		w := math.Min(l/(0.1*W), lsdMaxWeight)
		if math.Abs(a) < angleTolDeg {
			rows = append(rows, dewarpRow{0.5 * (x1 + x2), 0.5 * (y1 + y2), a, w, 0})
		}
	}

	x0, y0 := float64(inset), float64(inset)
	x1, y1 := W-float64(inset), H-float64(inset)
	if x1-x0 < 90 || y1-y0 < 90 {
		return rows
	}
	cw, ch := (x1-x0)/3.0, (y1-y0)/3.0
	for _, yb := range linspace(y0, y1-ch, dewarpGrid) {
		for _, xb := range linspace(x0, x1-cw, dewarpGrid) {
			crop, err := imaging.ClampedCrop(gray, int(xb), int(yb), int(xb+cw), int(yb+ch))
			if err != nil {
				continue
			}
			tilt, ratio := profileTilt(crop)
			crop.Close()
			if ratio >= profileMinPeak {
				w := profileWeight * math.Min(1.0, ratio-1.0)
				rows = append(rows, dewarpRow{xb + 0.5*cw, yb + 0.5*ch, tilt, w, 1})
			}
		}
	}
	return rows
}

func basis5(xn, yn float64) [5]float64 {
	return [5]float64{1, xn, yn, xn * xn, xn * yn}
}

// fitTilt is line_dewarp._fit_tilt: robust (Cauchy IRLS) weighted ridge regression of
// the tilt polynomial.
func fitTilt(meas []dewarpRow, W, H float64) [5]float64 {
	n := len(meas)
	A := make([][5]float64, n)
	t := make([]float64, n)
	w := make([]float64, n)
	for i, m := range meas {
		xn := (m.X - W/2) / (W / 2)
		yn := (m.Y - H/2) / (W / 2)
		A[i] = basis5(xn, yn)
		t[i] = m.Tilt
		w[i] = m.Weight
	}
	var c [5]float64
	for iter := 0; iter < irlsIters; iter++ {
		var AtWA [5][5]float64
		var AtWt [5]float64
		for i := 0; i < n; i++ {
			for a := 0; a < 5; a++ {
				AtWt[a] += A[i][a] * w[i] * t[i]
				for b := 0; b < 5; b++ {
					AtWA[a][b] += A[i][a] * w[i] * A[i][b]
				}
			}
		}
		for d := 0; d < 5; d++ {
			AtWA[d][d] += ridge
		}
		AA := make([][]float64, 5)
		for i := range AA {
			AA[i] = AtWA[i][:]
		}
		sol, ok := solveLinearN(AA, AtWt[:])
		if !ok {
			break
		}
		copy(c[:], sol)
		for i := 0; i < n; i++ {
			r := t[i]
			for a := 0; a < 5; a++ {
				r -= A[i][a] * c[a]
			}
			w[i] = meas[i].Weight / (1.0 + (r/irlsScaleDeg)*(r/irlsScaleDeg))
		}
	}
	return c
}

// displacement is line_dewarp.displacement: the x-integral of the tilt polynomial,
// evaluated on a coarse grid (W/8 x H/8) and resized up to the full page.
func displacement(c [5]float64, w, h int) []float32 {
	W, H := float64(w), float64(h)
	gw := maxInt(2, w/8)
	gh := maxInt(2, h/8)
	grid := make([]float32, gw*gh)
	c0 := c[0] * math.Pi / 180
	c1 := c[1] * math.Pi / 180
	c2 := c[2] * math.Pi / 180
	c3 := c[3] * math.Pi / 180
	c4 := c[4] * math.Pi / 180
	for gy := 0; gy < gh; gy++ {
		yRaw := float64(gy) * (H - 1) / float64(gh-1)
		yn := (yRaw - H/2) / (W / 2)
		for gx := 0; gx < gw; gx++ {
			xRaw := float64(gx) * (W - 1) / float64(gw-1)
			xn := (xRaw - W/2) / (W / 2)
			v := c0*xn + c1*xn*xn/2 + c2*xn*yn + c3*xn*xn*xn/3 + c4*xn*xn*yn/2
			grid[gy*gw+gx] = float32(v * (W / 2))
		}
	}
	gridImg, err := imaging.NewFloat32FromBytes(grid, gw, gh)
	if err != nil {
		return make([]float32, w*h)
	}
	defer gridImg.Close()
	full := imaging.Resize(gridImg, w, h, imaging.InterLinear)
	defer full.Close()
	out, err := imaging.Float32Data(full)
	if err != nil {
		return make([]float32, w*h)
	}
	return out
}

// DewarpByLines is line_dewarp.dewarp_by_lines: the displacement map that unbends a
// page, or nil when there is not enough evidence or no worthwhile bend.
func DewarpByLines(gray imaging.Image, inset int) ([]float32, StraightenInfo) {
	H, W := gray.Height(), gray.Width()
	meas := measureCells(gray, inset)
	var cells []dewarpRow
	for _, m := range meas {
		if m.Source == 1 {
			cells = append(cells, m)
		}
	}
	if len(cells) < minCells {
		return nil, StraightenInfo{Reason: "too few cells"}
	}
	cellAbs := make([]float64, len(cells))
	cellW := make([]float64, len(cells))
	for i, c := range cells {
		cellAbs[i], cellW[i] = math.Abs(c.Tilt), c.Weight
	}
	before := wmedian(cellAbs, cellW)

	c := fitTilt(meas, float64(W), float64(H))
	v := displacement(c, W, H)
	vmax := 0.0
	for _, x := range v {
		if a := math.Abs(float64(x)); a > vmax {
			vmax = a
		}
	}
	if vmax < minDispPx {
		return nil, StraightenInfo{Reason: "flat enough"}
	}
	if vmax > maxDispFrac*float64(H) {
		return nil, StraightenInfo{Reason: "bend too large"}
	}

	afterAll := make([]float64, len(meas))
	beforeAll := make([]float64, len(meas))
	weights := make([]float64, len(meas))
	var afterCellsVals, afterCellsW []float64
	for i, m := range meas {
		xn := (m.X - float64(W)/2) / (float64(W) / 2)
		yn := (m.Y - float64(H)/2) / (float64(W) / 2)
		basis := basis5(xn, yn)
		var fit float64
		for a := 0; a < 5; a++ {
			fit += basis[a] * c[a]
		}
		afterAll[i] = math.Abs(m.Tilt - fit)
		beforeAll[i] = math.Abs(m.Tilt)
		weights[i] = m.Weight
		if m.Source == 1 {
			afterCellsVals = append(afterCellsVals, afterAll[i])
			afterCellsW = append(afterCellsW, m.Weight)
		}
	}
	after := wmedian(afterCellsVals, afterCellsW)
	beforeAllM := wmedian(beforeAll, weights)
	afterAllM := wmedian(afterAll, weights)
	if after > (1.0-dewarpMinGain)*before || afterAllM > beforeAllM {
		return nil, StraightenInfo{Reason: "no gain"}
	}
	return v, StraightenInfo{Applied: true}
}

// ApplyDewarp remaps a page by the vertical displacement map, matching
// line_dewarp.apply_dewarp.
func ApplyDewarp(page imaging.Image, v []float32) imaging.Image {
	return imaging.RemapVerticalDisplacement(page, v, page.Width(), page.Height())
}
