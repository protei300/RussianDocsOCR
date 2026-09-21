package modules

import (
	"math"
	"math/rand"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// Port of pipeline_modules/page_registration/quad_fit.py: a page quad from
// straight-line fits to the segmentation contour, robust to a thumb merged into the
// mask or a bent/blurred corner that throws off the plain polygon-simplification quad
// (imaging.ExtractQuad).
//
// Pure Go, no OpenCV: contour resampling, RANSAC line fitting and line intersection
// are all plain point arithmetic. The one place this CANNOT be bit-exact with the
// reference is the RANSAC sample draw itself - `math/rand`'s PRNG is not
// `numpy.random.default_rng(0)`'s PCG64, so the same seed does not reproduce the same
// sample pairs. The fit this converges to is still a legitimate least-squares line
// through the same inlier band (RANSAC is used here to reject a handful of outlier
// contour points, not to pick between structurally different candidates), so the
// practical effect is a few-pixel difference in the fitted quad, not a different
// answer; see the task log for the measured canvas divergence this and the LSD
// substitution (imaging.DetectSegments) together cause.

const (
	resampleStepPx     = 3.0
	quadFitIterations  = 3
	ransacSamples      = 200
	inlierFrac         = 0.012
	minSidePoints      = 12
	minSupportFrac     = 0.30
	assignOverhang     = 0.05
	minIouWithInit     = 0.6
	maxOutsideFrac     = 0.35
	frameTolPx         = 3.0
	extrapolateMinVis  = 0.5
	extrapolateMaxVis  = 0.97
)

// QuadFitInfo mirrors quad_fit.fit_quad_lines's info dict, trimmed to what the port's
// callers (page_registration, and the conformance boundary if ever needed) read.
type QuadFitInfo struct {
	Method       string // "lines" | "polygon" | "none"
	Reason       string
	Clipped      []int
	Extrapolated *int
	Visible      *float64
}

// FitQuadLines is quad_fit.fit_quad_lines. imageShape is (h, w); pass (0, 0) for "no
// frame", matching image_shape=None (disables the frame-clipping/extrapolation logic).
// aspectHOverW enables extrapolation of a side that lies on the photo frame - pass 0 to
// disable, matching aspect_h_over_w=None.
func FitQuadLines(contour []imaging.Point, imgH, imgW int, aspectHOverW float64,
	extrapolate bool) ([]imaging.Point, QuadFitInfo) {

	init := imaging.ExtractQuad(contour)
	if init == nil {
		return nil, QuadFitInfo{Method: "none"}
	}
	init = imaging.OrderPoints(init)
	info := QuadFitInfo{Method: "polygon"}
	hasFrame := imgH > 0 && imgW > 0

	pts := resampleContour(contour, resampleStepPx)
	if len(pts) < 4*minSidePoints {
		info.Reason = "few points"
		return init, info
	}
	rng := rand.New(rand.NewSource(0))

	usable := make([]bool, len(pts))
	for i, p := range pts {
		if hasFrame {
			usable[i] = !(p.X <= frameTolPx || p.X >= float64(imgW)-1-frameTolPx ||
				p.Y <= frameTolPx || p.Y >= float64(imgH)-1-frameTolPx)
		} else {
			usable[i] = true
		}
	}

	quad := append([]imaging.Point(nil), init...)
	refit := make([]bool, 4)
	for iter := 0; iter < quadFitIterations; iter++ {
		A := quad
		B := []imaging.Point{quad[1], quad[2], quad[3], quad[0]}
		d := make([]imaging.Point, 4)
		L := make([]float64, 4)
		u := make([]imaging.Point, 4)
		degenerate := false
		for k := 0; k < 4; k++ {
			d[k] = imaging.Point{X: B[k].X - A[k].X, Y: B[k].Y - A[k].Y}
			L[k] = math.Hypot(d[k].X, d[k].Y)
			if L[k] < 2 {
				degenerate = true
			}
			if L[k] > 0 {
				u[k] = imaging.Point{X: d[k].X / L[k], Y: d[k].Y / L[k]}
			}
		}
		if degenerate {
			info.Reason = "degenerate side"
			return init, info
		}

		side := make([]int, len(pts))
		assigned := make([]bool, len(pts))
		for i, p := range pts {
			bestK, bestDist := -1, math.Inf(1)
			for k := 0; k < 4; k++ {
				rx, ry := p.X-A[k].X, p.Y-A[k].Y
				proj := (rx*u[k].X + ry*u[k].Y) / L[k]
				if proj < -assignOverhang || proj > 1+assignOverhang {
					continue
				}
				dist := math.Abs(rx*(-u[k].Y) + ry*u[k].X)
				if dist < bestDist {
					bestDist, bestK = dist, k
				}
			}
			side[i] = bestK
			assigned[i] = bestK >= 0
		}

		type line struct {
			p, dir imaging.Point
		}
		lines := make([]line, 4)
		for k := 0; k < 4; k++ {
			var sel []imaging.Point
			for i, p := range pts {
				if side[i] == k && assigned[i] && usable[i] {
					sel = append(sel, p)
				}
			}
			refit[k] = false
			if len(sel) >= minSidePoints {
				if p0, dv, inl, ok := fitLineRansac(sel, math.Max(2.0, inlierFrac*L[k]), rng); ok {
					// span of the inliers along the fitted direction
					var lo, hi float64
					first := true
					for i, keep := range inl {
						if !keep {
							continue
						}
						t := (sel[i].X-p0.X)*dv.X + (sel[i].Y-p0.Y)*dv.Y
						if first {
							lo, hi, first = t, t, false
						} else {
							if t < lo {
								lo = t
							}
							if t > hi {
								hi = t
							}
						}
					}
					if !first && hi-lo >= minSupportFrac*L[k] {
						lines[k] = line{p0, dv}
						refit[k] = true
					}
				}
			}
			if !refit[k] {
				lines[k] = line{A[k], u[k]}
			}
		}

		corners := make([]imaging.Point, 4)
		for k := 0; k < 4; k++ {
			prev := lines[(k+3)%4]
			cur := lines[k]
			c, ok := intersectLines(prev.p, prev.dir, cur.p, cur.dir)
			if !ok {
				info.Reason = "parallel sides"
				return init, info
			}
			corners[k] = c
		}
		quad = imaging.OrderPoints(corners)
	}
	info.Clipped = nil
	anyRefit := false
	for _, r := range refit {
		anyRefit = anyRefit || r
	}
	if !anyRefit {
		info.Reason = "no side fitted"
		return init, info
	}
	if !isConvex(quad) {
		info.Reason = "not convex"
		return init, info
	}
	iou := quadIoU(quad, init)
	if iou < minIouWithInit {
		info.Reason = "disagrees with polygon"
		return init, info
	}
	if hasFrame {
		w, h := float64(imgW), float64(imgH)
		slack := maxOutsideFrac * dist(quad[1], quad[0])
		for _, p := range quad {
			if p.X < -slack || p.X > w+slack || p.Y < -slack || p.Y > h+slack {
				info.Reason = "corner outside frame"
				return init, info
			}
		}
		info.Clipped = clippedSides(quad, w, h)
		for _, k := range info.Clipped {
			for _, idx := range [2]int{k, (k + 1) % 4} {
				quad[idx].X = clampF(quad[idx].X, 0, w-1)
				quad[idx].Y = clampF(quad[idx].Y, 0, h-1)
			}
		}
		if extrapolate && aspectHOverW > 0 && len(info.Clipped) == 1 {
			newQuad, visible := extrapolateSide(quad, info.Clipped[0], aspectHOverW)
			v := visible
			info.Visible = &v
			if newQuad != nil {
				quad = newQuad
				side := info.Clipped[0]
				info.Extrapolated = &side
			}
		}
	}
	info.Method = "lines"
	return quad, info
}

func resampleContour(contour []imaging.Point, step float64) []imaging.Point {
	if len(contour) < 3 {
		return append([]imaging.Point(nil), contour...)
	}
	closed := append(append([]imaging.Point(nil), contour...), contour[0])
	cum := make([]float64, len(closed))
	for i := 1; i < len(closed); i++ {
		cum[i] = cum[i-1] + dist(closed[i-1], closed[i])
	}
	total := cum[len(cum)-1]
	if total <= 0 {
		return append([]imaging.Point(nil), contour...)
	}
	n := int(total / step)
	if n < len(contour) {
		n = len(contour)
	}
	out := make([]imaging.Point, n)
	for i := 0; i < n; i++ {
		t := total * float64(i) / float64(n)
		out[i] = interpAlong(closed, cum, t)
	}
	return out
}

// interpAlong linearly interpolates the polyline `closed` (cumulative arc lengths
// `cum`) at arc length t, matching np.interp per axis.
func interpAlong(closed []imaging.Point, cum []float64, t float64) imaging.Point {
	if t <= cum[0] {
		return closed[0]
	}
	if t >= cum[len(cum)-1] {
		return closed[len(closed)-1]
	}
	// binary search for the segment
	lo, hi := 0, len(cum)-1
	for hi-lo > 1 {
		mid := (lo + hi) / 2
		if cum[mid] <= t {
			lo = mid
		} else {
			hi = mid
		}
	}
	seg := cum[hi] - cum[lo]
	if seg <= 0 {
		return closed[lo]
	}
	f := (t - cum[lo]) / seg
	return imaging.Point{
		X: closed[lo].X + f*(closed[hi].X-closed[lo].X),
		Y: closed[lo].Y + f*(closed[hi].Y-closed[lo].Y),
	}
}

// fitLineRansac is quad_fit._fit_line: RANSAC line through pts, then two rounds of
// least-squares refit (cv2.fitLine DIST_L2 == ordinary total-least-squares through the
// inliers, i.e. the centroid and the leading eigenvector of the centred covariance).
func fitLineRansac(pts []imaging.Point, thr float64, rng *rand.Rand) (
	p0, dir imaging.Point, inliers []bool, ok bool) {

	n := len(pts)
	if n < 2 {
		return p0, dir, nil, false
	}
	bestCount := -1
	var bestI, bestJ int
	for s := 0; s < ransacSamples; s++ {
		i, j := rng.Intn(n), rng.Intn(n)
		dx, dy := pts[j].X-pts[i].X, pts[j].Y-pts[i].Y
		l := math.Hypot(dx, dy)
		if l <= 1e-6 {
			continue
		}
		ux, uy := dx/l, dy/l
		nx, ny := -uy, ux
		count := 0
		for _, p := range pts {
			rx, ry := p.X-pts[i].X, p.Y-pts[i].Y
			if math.Abs(rx*nx+ry*ny) < thr {
				count++
			}
		}
		if count > bestCount {
			bestCount, bestI, bestJ = count, i, j
		}
	}
	if bestCount < 0 {
		return p0, dir, nil, false
	}
	dx, dy := pts[bestJ].X-pts[bestI].X, pts[bestJ].Y-pts[bestI].Y
	l := math.Hypot(dx, dy)
	ux, uy := dx/l, dy/l
	inl := make([]bool, n)
	for i, p := range pts {
		rx, ry := p.X-pts[bestI].X, p.Y-pts[bestI].Y
		inl[i] = math.Abs(rx*(-uy)+ry*ux) < thr
	}
	p0, dir = pts[bestI], imaging.Point{X: ux, Y: uy}
	for iter := 0; iter < 2; iter++ {
		var sel []imaging.Point
		for i, keep := range inl {
			if keep {
				sel = append(sel, pts[i])
			}
		}
		if len(sel) < 2 {
			return p0, dir, nil, false
		}
		p0, dir = fitLineLS(sel)
		nx, ny := -dir.Y, dir.X
		inl = make([]bool, n)
		for i, p := range pts {
			rx, ry := p.X-p0.X, p.Y-p0.Y
			inl[i] = math.Abs(rx*nx+ry*ny) < thr
		}
	}
	return p0, dir, inl, true
}

// fitLineLS is the DIST_L2 case of cv2.fitLine: total least squares through the point
// set (centroid + leading eigenvector of the centred 2x2 covariance).
func fitLineLS(pts []imaging.Point) (p0, dir imaging.Point) {
	var mx, my float64
	for _, p := range pts {
		mx += p.X
		my += p.Y
	}
	n := float64(len(pts))
	mx /= n
	my /= n
	var sxx, sxy, syy float64
	for _, p := range pts {
		dx, dy := p.X-mx, p.Y-my
		sxx += dx * dx
		sxy += dx * dy
		syy += dy * dy
	}
	// Leading eigenvector of [[sxx, sxy], [sxy, syy]] via the closed-form 2x2 formula.
	trace := sxx + syy
	diff := sxx - syy
	disc := math.Sqrt(diff*diff + 4*sxy*sxy)
	lambda1 := (trace + disc) / 2
	var vx, vy float64
	if math.Abs(sxy) > 1e-12 {
		vx, vy = lambda1-syy, sxy
	} else if sxx >= syy {
		vx, vy = 1, 0
	} else {
		vx, vy = 0, 1
	}
	l := math.Hypot(vx, vy)
	if l < 1e-12 {
		vx, vy, l = 1, 0, 1
	}
	return imaging.Point{X: mx, Y: my}, imaging.Point{X: vx / l, Y: vy / l}
}

func intersectLines(p, d, q, e imaging.Point) (imaging.Point, bool) {
	den := d.X*e.Y - d.Y*e.X
	if math.Abs(den) < 1e-9 {
		return imaging.Point{}, false
	}
	t := ((q.X-p.X)*e.Y - (q.Y-p.Y)*e.X) / den
	return imaging.Point{X: p.X + t*d.X, Y: p.Y + t*d.Y}, true
}

func isConvex(quad []imaging.Point) bool {
	n := len(quad)
	sign := 0.0
	for i := 0; i < n; i++ {
		a, b, c := quad[i], quad[(i+1)%n], quad[(i+2)%n]
		cross := (b.X-a.X)*(c.Y-b.Y) - (b.Y-a.Y)*(c.X-b.X)
		if cross == 0 {
			continue
		}
		s := 1.0
		if cross < 0 {
			s = -1.0
		}
		if sign == 0 {
			sign = s
		} else if s != sign {
			return false
		}
	}
	return true
}

// quadIoU is quad_fit._quad_iou: intersection over union of two ordered convex quads.
func quadIoU(a, b []imaging.Point) float64 {
	inter := convexIntersectionArea(a, b)
	union := polygonArea(a) + polygonArea(b) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func polygonArea(pts []imaging.Point) float64 {
	var a float64
	n := len(pts)
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += pts[i].X*pts[j].Y - pts[j].X*pts[i].Y
	}
	return math.Abs(a) / 2
}

// convexIntersectionArea is the Sutherland-Hodgman clip of convex polygon a by convex
// polygon b, area of what remains - equivalent to cv2.intersectConvexConvex for two
// already-convex inputs.
func convexIntersectionArea(a, b []imaging.Point) float64 {
	out := append([]imaging.Point(nil), a...)
	n := len(b)
	for i := 0; i < n && len(out) > 0; i++ {
		p1, p2 := b[i], b[(i+1)%n]
		out = clipPolygon(out, p1, p2)
	}
	if len(out) < 3 {
		return 0
	}
	return polygonArea(out)
}

func clipPolygon(poly []imaging.Point, a, b imaging.Point) []imaging.Point {
	var out []imaging.Point
	n := len(poly)
	edgeX, edgeY := b.X-a.X, b.Y-a.Y
	// OrderPoints (and quadInImage, which follows the same TL,TR,BR,BL convention)
	// winds a quad counter-clockwise treating X,Y as plain Cartesian numbers (TL(0,0)
	// -> TR(1,0) -> BR(1,1) -> BL(0,1) is the textbook CCW unit square) - Y being
	// "down" in image terms never enters this arithmetic. For a CCW polygon the
	// Sutherland-Hodgman "inside" half-plane is cross(edge, p-a) >= 0. Getting this
	// backwards (<= 0) makes every clip empty and QuadIoU return 0 for every pair,
	// however much they visually overlap - caught the hard way: it silently broke
	// every quad<->registration IoU decision in _register_pages (see the task log).
	inside := func(p imaging.Point) bool {
		return edgeX*(p.Y-a.Y)-edgeY*(p.X-a.X) >= 0
	}
	for i := 0; i < n; i++ {
		cur, prev := poly[i], poly[(i+n-1)%n]
		curIn, prevIn := inside(cur), inside(prev)
		if curIn {
			if !prevIn {
				out = append(out, segIntersect(prev, cur, a, b))
			}
			out = append(out, cur)
		} else if prevIn {
			out = append(out, segIntersect(prev, cur, a, b))
		}
	}
	return out
}

func segIntersect(p1, p2, p3, p4 imaging.Point) imaging.Point {
	x1, y1, x2, y2 := p1.X, p1.Y, p2.X, p2.Y
	x3, y3, x4, y4 := p3.X, p3.Y, p4.X, p4.Y
	den := (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
	if math.Abs(den) < 1e-12 {
		return p1
	}
	tA := (x1*y2 - y1*x2)
	tB := (x3*y4 - y3*x4)
	x := (tA*(x3-x4) - (x1-x2)*tB) / den
	y := (tA*(y3-y4) - (y1-y2)*tB) / den
	return imaging.Point{X: x, Y: y}
}

func clippedSides(quad []imaging.Point, w, h float64) []int {
	var out []int
	for k := 0; k < 4; k++ {
		a, b := quad[k], quad[(k+1)%4]
		onFrame := func(coord func(imaging.Point) float64, val float64) bool {
			return math.Abs(coord(a)-val) <= frameTolPx && math.Abs(coord(b)-val) <= frameTolPx
		}
		xOf := func(p imaging.Point) float64 { return p.X }
		yOf := func(p imaging.Point) float64 { return p.Y }
		if onFrame(xOf, 0) || onFrame(xOf, w-1) || onFrame(yOf, 0) || onFrame(yOf, h-1) {
			out = append(out, k)
		}
	}
	return out
}

func extrapolateSide(quad []imaging.Point, side int, aspectHOverW float64) ([]imaging.Point, float64) {
	k0, k1 := side, (side+1)%4
	o0, o1 := (side+3)%4, (side+2)%4
	oppLen := dist(quad[o1], quad[o0])
	if oppLen < 1 {
		return nil, 0
	}
	expect := oppLen * aspectHOverW
	if side == 1 || side == 3 {
		expect = oppLen / aspectHOverW
	}
	u0 := imaging.Point{X: quad[k0].X - quad[o0].X, Y: quad[k0].Y - quad[o0].Y}
	u1 := imaging.Point{X: quad[k1].X - quad[o1].X, Y: quad[k1].Y - quad[o1].Y}
	l0, l1 := math.Hypot(u0.X, u0.Y), math.Hypot(u1.X, u1.Y)
	if l0 < 1 || l1 < 1 {
		return nil, 0
	}
	visible := 0.5 * (l0 + l1) / expect
	if visible < extrapolateMinVis || visible > extrapolateMaxVis {
		return nil, visible
	}
	newQuad := append([]imaging.Point(nil), quad...)
	newQuad[k0] = imaging.Point{X: quad[o0].X + u0.X/l0*expect, Y: quad[o0].Y + u0.Y/l0*expect}
	newQuad[k1] = imaging.Point{X: quad[o1].X + u1.X/l1*expect, Y: quad[o1].Y + u1.Y/l1*expect}
	return newQuad, visible
}

func dist(a, b imaging.Point) float64 {
	dx, dy := a.X-b.X, a.Y-b.Y
	return math.Sqrt(dx*dx + dy*dy)
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
