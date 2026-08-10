package imaging

import (
	"math"
	"sort"
)

// DocMarginFrac is the outward cushion applied to a detected document quad.
//
// Not a fudge factor: the reference documents that most of an earlier crop regression
// was a mis-calibrated mask threshold (fixed at source), and this constant now only
// restores the slight over-coverage the pre-retrain model happened to have — which is
// what kept edge-printed content such as the internal passport's series and number
// inside the crop. Applied BEFORE the corners are clipped to the image bounds, so it can
// never pull in area from outside the frame.
const DocMarginFrac = 0.01

// OrderPoints canonicalises four corners to [TL, TR, BR, BL].
//
// Uses the sum and difference trick: the top-left has the smallest x+y and the
// bottom-right the largest; the top-right has the smallest y-x and the bottom-left the
// largest. Note the difference is **y - x**, matching `np.diff(pts, axis=1)` on
// (x, y) pairs — reversing it silently swaps TR and BL, which then mirrors the warp.
func OrderPoints(pts []Point) []Point {
	if len(pts) != 4 {
		return nil
	}
	minSum, maxSum := 0, 0
	minDiff, maxDiff := 0, 0
	for i, p := range pts {
		s, d := p.X+p.Y, p.Y-p.X
		if s < pts[minSum].X+pts[minSum].Y {
			minSum = i
		}
		if s > pts[maxSum].X+pts[maxSum].Y {
			maxSum = i
		}
		if d < pts[minDiff].Y-pts[minDiff].X {
			minDiff = i
		}
		if d > pts[maxDiff].Y-pts[maxDiff].X {
			maxDiff = i
		}
	}
	return []Point{pts[minSum], pts[minDiff], pts[maxSum], pts[maxDiff]}
}

// ExtractQuad reduces a segmentation contour to four corners.
//
// Convex hull first, then Douglas-Peucker at a SWEEP of epsilon fractions, stopping at
// the first that yields exactly four points. The sweep exists because a single epsilon
// either over-simplifies a clean rectangle or under-simplifies a ragged mask; the
// fractions and their order are the reference's and must not be reordered, because the
// first match wins.
//
// The fallback is the minimum-area rotated rectangle, which always yields four corners.
// Returns nil for a degenerate contour of fewer than four points.
func ExtractQuad(contour []Point) []Point {
	if len(contour) < 4 {
		return nil
	}
	hull := ConvexHull(contour)
	if len(hull) == 0 {
		return nil
	}
	peri := ArcLength(hull)
	for _, frac := range []float64{0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15} {
		approx := ApproxPolyDP(hull, frac*peri)
		if len(approx) == 4 {
			return approx
		}
	}
	return MinAreaRectPoints(contour)
}

// ExpandQuad pushes corners outward from the centroid.
//
// The applied scale is 1 + 2*margin, because `margin` names how far each EDGE moves out
// as a fraction of the document's own size — both edges of an axis together account for
// twice that.
func ExpandQuad(quad []Point, margin float64) []Point {
	if margin <= 0 {
		return quad
	}
	var cx, cy float64
	for _, p := range quad {
		cx += p.X
		cy += p.Y
	}
	cx /= float64(len(quad))
	cy /= float64(len(quad))

	scale := 1.0 + 2.0*margin
	out := make([]Point, len(quad))
	for i, p := range quad {
		out[i] = Point{X: cx + (p.X-cx)*scale, Y: cy + (p.Y-cy)*scale}
	}
	return out
}

// FourPointTransform warps a quad to an axis-aligned rectangle.
//
// The output size comes from the quad's REAL side lengths, so a tilted document is
// rectified without being stretched. Width is the longer of the two horizontal edges and
// height the longer of the two vertical ones — taking the max rather than an average
// avoids cropping content on the longer side.
//
// Returns a zero Image and false when the target is degenerate (under 2 px), matching
// the reference's `return None`, which its caller then skips.
func FourPointTransform(img Image, quad []Point) (Image, bool) {
	rect := OrderPoints(quad)
	if rect == nil {
		return Image{}, false
	}
	tl, tr, br, bl := rect[0], rect[1], rect[2], rect[3]

	// RoundToEven, not math.Round: Python's round() breaks ties to EVEN, and a
	// half-away-from-zero rounding here changes the output size by one pixel — which
	// is a shape mismatch, not a tolerance question (CONVENTIONS §6.5).
	width := int(math.RoundToEven(math.Max(dist(br, bl), dist(tr, tl))))
	height := int(math.RoundToEven(math.Max(dist(tr, br), dist(tl, bl))))
	if width < 2 || height < 2 {
		return Image{}, false
	}
	out, err := WarpPerspectiveQuad(img, rect, width, height)
	if err != nil {
		return Image{}, false
	}
	return out, true
}

func dist(a, b Point) float64 {
	dx, dy := a.X-b.X, a.Y-b.Y
	return math.Sqrt(dx*dx + dy*dy)
}

// StackDirection selects how two rectified pages are joined.
type StackDirection string

const (
	// StackAuto picks the direction from the pages' actual layout.
	StackAuto StackDirection = "auto"
	// StackHorizontal joins left to right with a common height.
	StackHorizontal StackDirection = "horizontal"
	// StackVertical joins top to bottom with a common width.
	StackVertical StackDirection = "vertical"
)

// FixPerspective rectifies each detected segment and merges the results.
//
// Returns the rectified image and whether anything was rectified at all. When no valid
// quad is found the caller keeps the ORIGINAL image — the reference returns `img`
// unchanged, and a port that returned an error instead would break every document the
// border detector cannot read.
//
// The two-page branch exists for an open passport spread. 'auto' compares the pages'
// centroid separation: further apart horizontally means side by side, so stitch left to
// right; otherwise one above the other. The debug overlay the reference also produces
// (`cnt_img`, the original with quads drawn) is not built here — nothing in the
// conformance contract consumes it, and it would cost a full image copy per document.
func FixPerspective(img Image, segments [][]Point, stack StackDirection,
	margin float64) (Image, bool) {

	type page struct {
		quad  []Point
		image Image
	}
	var pages []page

	for _, cnt := range segments {
		quad := ExtractQuad(cnt)
		if quad == nil {
			continue
		}
		rect := OrderPoints(quad)
		rect = ExpandQuad(rect, margin)
		// Clip to the frame AFTER expanding, so the cushion can never reach outside
		// the image. Note the bounds are width and height, not width-1/height-1 —
		// matching np.clip(..., 0, img.shape[1]).
		for i := range rect {
			rect[i].X = clampF(rect[i].X, 0, float64(img.Width()))
			rect[i].Y = clampF(rect[i].Y, 0, float64(img.Height()))
		}
		warped, ok := FourPointTransform(img, rect)
		if !ok {
			continue
		}
		pages = append(pages, page{quad: rect, image: warped})
	}

	if len(pages) == 0 {
		return Image{}, false
	}
	if len(pages) == 1 {
		return pages[0].image, true
	}

	direction := stack
	if stack == StackAuto {
		c0x, c0y := centroid(pages[0].quad)
		c1x, c1y := centroid(pages[1].quad)
		if math.Abs(c0x-c1x) >= math.Abs(c0y-c1y) {
			direction = StackHorizontal
		} else {
			direction = StackVertical
		}
	}

	// Only the first two pages are merged, as in the reference: `stack` inspects
	// quads[0] and quads[1], and max_pages is 2 for the one doc type that spreads.
	if direction == StackHorizontal {
		sort.SliceStable(pages, func(a, b int) bool { return minX(pages[a].quad) < minX(pages[b].quad) })
		commonH := pages[0].image.Height()
		for _, p := range pages[1:] {
			if p.image.Height() < commonH {
				commonH = p.image.Height()
			}
		}
		scaled := make([]Image, len(pages))
		for i, p := range pages {
			// RoundToEven again: this is `int(round(...))` in the reference, and the
			// two-page spread is where the one-pixel difference shows up as a changed
			// canvas width.
			w := int(math.RoundToEven(float64(p.image.Width()) * float64(commonH) / float64(p.image.Height())))
			if w < 1 {
				w = 1
			}
			scaled[i] = Resize(p.image, w, commonH, InterLinear)
			_ = p.image.Close()
		}
		return joinAll(scaled, HStack)
	}

	sort.SliceStable(pages, func(a, b int) bool { return minY(pages[a].quad) < minY(pages[b].quad) })
	commonW := pages[0].image.Width()
	for _, p := range pages[1:] {
		if p.image.Width() < commonW {
			commonW = p.image.Width()
		}
	}
	scaled := make([]Image, len(pages))
	for i, p := range pages {
		h := int(math.RoundToEven(float64(p.image.Height()) * float64(commonW) / float64(p.image.Width())))
		if h < 1 {
			h = 1
		}
		scaled[i] = Resize(p.image, commonW, h, InterLinear)
		_ = p.image.Close()
	}
	return joinAll(scaled, VStack)
}

func joinAll(parts []Image, join func(a, b Image) (Image, error)) (Image, bool) {
	acc := parts[0]
	for _, p := range parts[1:] {
		next, err := join(acc, p)
		_ = acc.Close()
		_ = p.Close()
		if err != nil {
			return Image{}, false
		}
		acc = next
	}
	return acc, true
}

func centroid(q []Point) (float64, float64) {
	var x, y float64
	for _, p := range q {
		x += p.X
		y += p.Y
	}
	return x / float64(len(q)), y / float64(len(q))
}

func minX(q []Point) float64 {
	m := q[0].X
	for _, p := range q[1:] {
		if p.X < m {
			m = p.X
		}
	}
	return m
}

func minY(q []Point) float64 {
	m := q[0].Y
	for _, p := range q[1:] {
		if p.Y < m {
			m = p.Y
		}
	}
	return m
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
