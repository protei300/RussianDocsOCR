// Package geometry answers where a point of a pipeline canvas lies on the input image.
// Port of document_processing/geometry.py (PR #19, issue #18).
//
// A caller that shows a document to a person needs the field on the PHOTO, not on the canvas
// the pipeline built: the canvas is resized, turned upright, warped page by page, stitched and
// deskewed. Every stage that changes the geometry of the image it passes on describes that
// change as a map from its OUTPUT back to its INPUT, and the pipeline chains the maps of the
// stages that actually ran, in the order they ran.
//
// COORDINATES ARE CONTINUOUS PIXELS. An image spans [0, width] x [0, height], and the pixel
// with index (i, j) is the unit square centred at (i + 0.5, j + 0.5); box edges from the
// detectors are read the same way. OpenCV's warps put pixel centres on integer coordinates
// instead, so Homography shifts by half a pixel on the way in and out and the matrices OpenCV
// computed are used unchanged. cv2.resize and cv2.rotate need no shift in these coordinates.
//
// A MAP RECEIVES THE POINTS OF ONE SHAPE. A stitched canvas picks the piece a shape lies on by
// the shape's centroid, so the corners of one box are never sent to different pages.
//
// TWO TRAPS, both paid for by the first port of PR #19:
//   - a chain is read OUTPUT -> INPUT: Chain.Maps lists the stages in the order they RAN, and
//     ToInput walks it backwards. A map built "forward" (page -> canvas) sends a box off the photo.
//   - "not known" is said ONCE for the whole run (ToInput returns ok == false), never per box.
//
// Pure Go, no OpenCV: the package is testable without the native libraries.
package geometry

import "math"

// Point is a point of an image in continuous pixels.
type Point struct{ X, Y float64 }

// Quad is a quadrilateral, four corners in order.
type Quad [4]Point

// Corners are the corners of an axis-aligned box, clockwise from the top-left.
func Corners(x0, y0, x1, y1 float64) []Point {
	return []Point{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}
}

// Geometry is a map from the output of a stage back to its input.
//
// ToInput answers ok == false when the way back is not known. NOT KNOWN IS NOT THE SAME AS
// UNCHANGED: a stage that passes its image on untouched contributes no map at all
// (Chain.Then(nil)), while a stage whose effect no point map can express contributes Unknown
// and makes every answer downstream not known. A box that quietly lands somewhere else is worse
// than no box: the caller draws it over a photo and trusts it.
type Geometry interface {
	// ToInput maps points of the output to the input.
	ToInput(points []Point) (mapped []Point, ok bool)
}

// Scale is the input resized: output = input * (SX, SY).
type Scale struct{ SX, SY float64 }

func (s Scale) ToInput(points []Point) ([]Point, bool) {
	out := make([]Point, len(points))
	for i, p := range points {
		out[i] = Point{p.X / s.SX, p.Y / s.SY}
	}
	return out, true
}

// Offset is the input shifted: output = input + (DX, DY); a crop is a negative shift.
type Offset struct{ DX, DY float64 }

func (o Offset) ToInput(points []Point) ([]Point, bool) {
	out := make([]Point, len(points))
	for i, p := range points {
		out[i] = Point{p.X - o.DX, p.Y - o.DY}
	}
	return out, true
}

// QuarterTurns is cv2.ROTATE_90_COUNTERCLOCKWISE applied Turns times to a Width x Height input.
type QuarterTurns struct{ Width, Height, Turns int }

func (q QuarterTurns) ToInput(points []Point) ([]Point, bool) {
	var widths []int
	w, h := q.Width, q.Height
	n := ((q.Turns % 4) + 4) % 4 // Python's % is never negative
	for i := 0; i < n; i++ {
		widths = append(widths, w)
		w, h = h, w
	}
	out := append([]Point(nil), points...)
	// One turn sends (x, y) of a W-wide image to (y, W - x); the last turn is undone first.
	for i := len(widths) - 1; i >= 0; i-- {
		w := float64(widths[i])
		for j, p := range out {
			out[j] = Point{w - p.Y, p.X}
		}
	}
	return out, true
}

// Homography is the output of cv2.warpPerspective or cv2.warpAffine with a matrix (input ->
// output).
type Homography struct {
	inv [3][3]float64
}

// NewHomography keeps the inverse of m, the matrix the warp was called with.
func NewHomography(m [3][3]float64) Homography { return Homography{inv: invert3(m)} }

// NewAffine is NewHomography of a 2x3 matrix completed with (0, 0, 1).
func NewAffine(m [2][3]float64) Homography {
	return NewHomography([3][3]float64{m[0], m[1], {0, 0, 1}})
}

func (h Homography) ToInput(points []Point) ([]Point, bool) {
	out := make([]Point, len(points))
	for i, p := range points {
		x, y := p.X-0.5, p.Y-0.5
		mx := x*h.inv[0][0] + y*h.inv[0][1] + h.inv[0][2]
		my := x*h.inv[1][0] + y*h.inv[1][1] + h.inv[1][2]
		mw := x*h.inv[2][0] + y*h.inv[2][1] + h.inv[2][2]
		out[i] = Point{mx/mw + 0.5, my/mw + 0.5}
	}
	return out, true
}

// invert3 inverts a 3x3 matrix by Gauss-Jordan elimination with partial pivoting (np.linalg.inv
// is an LU solve; the two agree to rounding).
func invert3(m [3][3]float64) [3][3]float64 {
	var a [3][6]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			a[i][j] = m[i][j]
		}
		a[i][3+i] = 1
	}
	for c := 0; c < 3; c++ {
		piv := c
		for r := c + 1; r < 3; r++ {
			if math.Abs(a[r][c]) > math.Abs(a[piv][c]) {
				piv = r
			}
		}
		a[c], a[piv] = a[piv], a[c]
		d := a[c][c]
		for j := 0; j < 6; j++ {
			a[c][j] /= d
		}
		for r := 0; r < 3; r++ {
			if r == c {
				continue
			}
			f := a[r][c]
			for j := 0; j < 6; j++ {
				a[r][j] -= f * a[c][j]
			}
		}
	}
	var out [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = a[i][3+j]
		}
	}
	return out
}

// VerticalRemap is the output of cv2.remap(img, xs, ys + v): a page unbent by a vertical
// displacement map.
//
// The bend map of page_registration.line_dewarp is exactly this call, so the output pixel (i, j)
// SAMPLED the input at (i, j + v[j, i]) - the map is already the way back, no matrix and no
// inversion needed. v is read bilinearly between pixel centres and held at the edge outside the
// map, the way the remap replicated its border. A point of the output moves only along y.
type VerticalRemap struct {
	V    []float32 // row-major, H x W
	W, H int
}

func (r VerticalRemap) ToInput(points []Point) ([]Point, bool) {
	out := make([]Point, len(points))
	at := func(y, x int) float64 { return float64(r.V[y*r.W+x]) }
	for i, p := range points {
		x := math.Min(math.Max(p.X-0.5, 0), float64(r.W-1))
		y := math.Min(math.Max(p.Y-0.5, 0), float64(r.H-1))
		x0, y0 := int(math.Floor(x)), int(math.Floor(y))
		x1, y1 := min(x0+1, r.W-1), min(y0+1, r.H-1)
		fx, fy := x-float64(x0), y-float64(y0)
		shift := at(y0, x0)*(1-fx)*(1-fy) + at(y0, x1)*fx*(1-fy) +
			at(y1, x0)*(1-fx)*fy + at(y1, x1)*fx*fy
		out[i] = Point{p.X, p.Y + shift}
	}
	return out, true
}

// Unknown is a stage that changed the image in a way no point map expresses. The canvas is still
// correct and recognition is unaffected - only the way back is gone, and it stays gone for every
// stage after this one.
type Unknown struct{}

func (Unknown) ToInput([]Point) ([]Point, bool) { return nil, false }

// Chain is the maps of stages in the order the stages ran: the first one reads the input.
type Chain struct{ Maps []Geometry }

// Then is this chain followed by one more stage; nil - the stage passed its input on unchanged.
func (c Chain) Then(later Geometry) Chain {
	if later == nil {
		return c
	}
	maps := make([]Geometry, len(c.Maps), len(c.Maps)+1)
	copy(maps, c.Maps)
	return Chain{append(maps, later)}
}

// ToInput walks the chain from its last stage to its first.
func (c Chain) ToInput(points []Point) ([]Point, bool) {
	cur := append([]Point(nil), points...)
	for i := len(c.Maps) - 1; i >= 0; i-- {
		var ok bool
		cur, ok = c.Maps[i].ToInput(cur)
		// One stage that cannot answer ends the walk: the stages before it are fine, but their
		// input is no longer known.
		if !ok {
			return nil, false
		}
	}
	return cur, true
}

// Piece is a rectangle (x0, y0, x1, y1) of a canvas and the map of what fills it.
type Piece struct {
	Rect [4]float64
	Geo  Geometry
}

// Pieces is a canvas made of pieces.
type Pieces struct{ Pieces []Piece }

func (p Pieces) ToInput(points []Point) ([]Point, bool) {
	var cx, cy float64
	for _, q := range points {
		cx += q.X
		cy += q.Y
	}
	cx /= float64(len(points))
	cy /= float64(len(points))
	distance := func(pc Piece) float64 {
		dx := math.Max(math.Max(pc.Rect[0]-cx, 0), cx-pc.Rect[2])
		dy := math.Max(math.Max(pc.Rect[1]-cy, 0), cy-pc.Rect[3])
		return dx*dx + dy*dy
	}
	best := 0
	for i := 1; i < len(p.Pieces); i++ {
		if distance(p.Pieces[i]) < distance(p.Pieces[best]) { // min() keeps the first of equals
			best = i
		}
	}
	return p.Pieces[best].Geo.ToInput(points) // not ok if that piece cannot answer
}

// PlacedPage is one page of a stitched canvas: its size, where stitch_pages put it
// (scale, dx, dy) and the map from the page back to the image it was cut from (nil: handed on
// unchanged).
type PlacedPage struct {
	W, H   int
	Scale  float64
	DX, DY float64
	Geo    Geometry
}

// PlacedSize is image_transformation._placed: the size of a page on the canvas after the resize
// to the common side, with stitch_pages' rounding - a map built from it puts a point where the
// page's pixels went, not where the scale alone would put it.
func PlacedSize(w, h int, scale float64) (nw, nh int) {
	nw, nh = w, h
	if scale != 1.0 {
		nw = max(1, int(math.RoundToEven(float64(w)*scale)))
		nh = max(1, int(math.RoundToEven(float64(h)*scale)))
	}
	return nw, nh
}

// StitchedGeometry is image_transformation.stitched_geometry: the map from a canvas built by
// stitch_pages back to the image the pages came from - one rectangle of the canvas per page,
// with the page's own resize and offset composed after the page's own map.
func StitchedGeometry(pages []PlacedPage) Pieces {
	pieces := make([]Piece, 0, len(pages))
	for _, pg := range pages {
		nw, nh := PlacedSize(pg.W, pg.H, pg.Scale)
		placed := Chain{[]Geometry{
			Scale{float64(nw) / float64(pg.W), float64(nh) / float64(pg.H)},
			Offset{pg.DX, pg.DY},
		}}
		var maps []Geometry
		if pg.Geo != nil {
			maps = append(maps, pg.Geo)
		}
		maps = append(maps, placed)
		pieces = append(pieces, Piece{
			Rect: [4]float64{pg.DX, pg.DY, pg.DX + float64(nw), pg.DY + float64(nh)},
			Geo:  Chain{maps},
		})
	}
	return Pieces{pieces}
}
