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

	half := imaging.ResizeAreaBy(gray, 0.5)
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
	type cellBox struct{ x, y, w, h float64 }
	var cells []cellBox
	var centres [][2]float64
	for _, yb := range linspace(y0, y1-ch, dewarpGrid) {
		for _, xb := range linspace(x0, x1-cw, dewarpGrid) {
			cells = append(cells, cellBox{xb - x0, yb - y0, cw, ch})
			centres = append(centres, [2]float64{xb + 0.5*cw, yb + 0.5*ch})
		}
	}
	region, err := imaging.ClampedCrop(gray, int(x0), int(y0), int(x1), int(y1))
	if err != nil {
		return rows
	}
	defer region.Close()
	boxes := make([][4]float64, len(cells))
	for i, c := range cells {
		boxes[i] = [4]float64{c.x, c.y, c.w, c.h}
	}
	for i, tr := range cellTilts(region, boxes) {
		tilt, ratio := tr[0], tr[1]
		if ratio >= profileMinPeak {
			w := profileWeight * math.Min(1.0, ratio-1.0)
			rows = append(rows, dewarpRow{centres[i][0], centres[i][1], tilt, w, 1})
		}
	}
	return rows
}

// cellTilts is line_dewarp._cell_tilts: the tilt and peak ratio of the row profile in every
// cell (x, y, w, h of region), like profileTilt but rotating the WHOLE region once per angle
// and slicing the cells out of it. Rotation about the region centre instead of the cell
// centre only shifts a cell's content by a few pixels at these angles, which the profile does
// not mind - but it is not the same number, and the cell count that decides whether the page
// is bent at all (MIN_CELLS) sits on it: a port that rotated each cell about its own centre
// found 16 cells where the reference found 5 and bent a page the reference left flat.
func cellTilts(region imaging.Image, cells [][4]float64) [][2]float64 {
	results := make([][2]float64, len(cells))
	small := imaging.ResizeAreaBy(region, profileScale)
	defer small.Close()
	w, h := small.Width(), small.Height()
	ink, w2, h2 := otsuInvDropTallBlobs(small, blobMaxFrac*(cells[0][3]*profileScale))
	if w2 != w || h2 != h {
		return results
	}
	inkImg, err := imaging.NewGrayFromBytes(ink, w, h)
	if err != nil {
		return results
	}
	defer inkImg.Close()
	validImg := imaging.NewGrayFilled(h, w, 255)
	defer validImg.Close()

	type box struct{ x, y, w, h int }
	boxes := make([]box, len(cells))
	for i, c := range cells {
		boxes[i] = box{int(c[0] * profileScale), int(c[1] * profileScale),
			maxInt(2, int(c[2]*profileScale)), maxInt(2, int(c[3]*profileScale))}
	}

	score := func(angles []float64) [][]float64 {
		out := make([][]float64, len(angles))
		for i, a := range angles {
			out[i] = make([]float64, len(boxes))
			rot := imaging.RotateNearestZero(inkImg, a)
			cnt := imaging.RotateNearestZero(validImg, a)
			rb, errR := rot.Bytes()
			cb, errC := cnt.Bytes()
			if errR != nil || errC != nil {
				rot.Close()
				cnt.Close()
				continue
			}
			for j, b := range boxes {
				x2 := minInt(b.x+b.w, w) // exclusive
				y2 := minInt(b.y+b.h, h)
				if b.y >= y2 || b.x >= x2 {
					continue
				}
				inkRow := make([]int64, y2-b.y)
				cntRow := make([]int64, y2-b.y)
				var cmax int64
				for y := b.y; y < y2; y++ {
					var si, sc int64
					for x := b.x; x < x2; x++ {
						si += int64(rb[y*w+x])
						sc += int64(cb[y*w+x])
					}
					// the images hold 0 / 255: counts of pixels, as the reference's 0 / 1 floats
					inkRow[y-b.y], cntRow[y-b.y] = si/255, sc/255
					if sc/255 > cmax {
						cmax = sc / 255
					}
				}
				if cmax <= 0 {
					continue
				}
				var prof []float64
				for r := range cntRow {
					if enoughValid(cntRow[r], cmax) {
						prof = append(prof, float64(inkRow[r])/float64(maxInt64(cntRow[r], 1)))
					}
				}
				if len(prof) > 2 {
					out[i][j] = varianceF64(prof)
				}
			}
			rot.Close()
			cnt.Close()
		}
		return out
	}

	coarse := score(profileCoarse)
	column := func(m [][]float64, j int) []float64 {
		c := make([]float64, len(m))
		for i := range m {
			c[i] = m[i][j]
		}
		return c
	}
	peaks := make([]int, len(boxes))
	fineCache := map[int][][]float64{}
	for j := range boxes {
		col := column(coarse, j)
		ib := argmaxFloat(col)
		if ib == 0 || ib == len(profileCoarse)-1 || col[ib] <= 0 {
			peaks[j] = -1
			continue
		}
		peaks[j] = ib
	}
	fineAngles := []float64{}
	for a := -1.0; a <= 1.001; a += profileFineStep {
		fineAngles = append(fineAngles, a)
	}
	for _, ib := range peaks {
		if ib < 0 {
			continue
		}
		if _, seen := fineCache[ib]; seen {
			continue
		}
		angles := make([]float64, len(fineAngles))
		for k, fa := range fineAngles {
			angles[k] = profileCoarse[ib] + fa
		}
		fineCache[ib] = score(angles)
	}
	for j := range boxes {
		ib := peaks[j]
		if ib < 0 {
			continue
		}
		fs := column(fineCache[ib], j)
		jb := argmaxFloat(fs)
		best := profileCoarse[ib] + fineAngles[jb]
		if jb > 0 && jb < len(fs)-1 {
			y0v, y1v, y2v := fs[jb-1], fs[jb], fs[jb+1]
			den := y0v - 2*y1v + y2v
			if den < 0 {
				best += profileFineStep * 0.5 * (y0v - y2v) / den
			}
		}
		med := medianOf(column(coarse, j))
		ratio := 0.0
		if med > 0 {
			ratio = fs[jb] / med
		}
		results[j] = [2]float64{best, ratio}
	}
	return results
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

// enoughValid is `c >= 0.6 * c.max()` on the reference's float32 count arrays: NumPy keeps a
// float32 array float32 against a Python number, so the limit is the float32 product of
// float32(0.6) and the maximum - not a float64 0.6 * max, and not an integer truncation of it
// (a row whose count is 4 does not pass a limit of 4.2).
func enoughValid(c, cmax int64) bool {
	return float32(c) >= float32(float32(0.6)*float32(cmax))
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
