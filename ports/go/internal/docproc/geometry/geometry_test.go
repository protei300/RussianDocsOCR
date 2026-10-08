package geometry

import (
	"math"
	"testing"
)

// Mirrors the point-map part of tests/test_geometry.py of the reference. The tests that need
// OpenCV (a warped page, a deskew, a remap against the real call) are in modules/geometry_test.go.
//
// COORDINATES ARE CONTINUOUS: the pixel with index (i, j) is the unit square centred at
// (i + 0.5, j + 0.5). The expected points below state that half, which is also what keeps them
// honest about the half-pixel shift OpenCV's warps need.

func near(t *testing.T, got, want Point, atol float64) {
	t.Helper()
	if math.Abs(got.X-want.X) > atol || math.Abs(got.Y-want.Y) > atol {
		t.Errorf("got (%.4f, %.4f), want (%.4f, %.4f)", got.X, got.Y, want.X, want.Y)
	}
}

func one(t *testing.T, g Geometry, p Point) Point {
	t.Helper()
	out, ok := g.ToInput([]Point{p})
	if !ok {
		t.Fatal("the way back is not known")
	}
	return out[0]
}

// rotateCCW is cv2.rotate(img, ROTATE_90_COUNTERCLOCKWISE) on the index of one marked pixel of a
// width x height image: (col, row) -> (row, width-1-col), the new image being height wide.
func rotateCCW(col, row, width int) (int, int) { return row, width - 1 - col }

// Against the rotation itself: a turn count of its own is the easiest thing to get wrong by one.
func TestQuarterTurnsPutThePixelBack(t *testing.T) {
	for turns := 0; turns < 4; turns++ {
		w, h := 7, 4
		col, row := 5, 1
		cw, ch := w, h
		for i := 0; i < turns; i++ {
			col, row = rotateCCW(col, row, cw)
			cw, ch = ch, cw
		}
		got := one(t, QuarterTurns{Width: 7, Height: 4, Turns: turns}, Point{float64(col) + 0.5, float64(row) + 0.5})
		near(t, got, Point{5.5, 1.5}, 1e-9)
	}
}

// The chain is READ output -> input: the last stage is undone first. Built forward, it would send
// a box off the photo (the first port of PR #19 did exactly that).
func TestChainUndoesTheLastStageFirst(t *testing.T) {
	chain := Chain{[]Geometry{Scale{0.5, 0.25}, Offset{10, 20}}}
	near(t, one(t, chain, Point{15, 45}), Point{10, 100}, 1e-9)
	reversed := Chain{[]Geometry{Offset{10, 20}, Scale{0.5, 0.25}}}
	if got := one(t, reversed, Point{15, 45}); math.Abs(got.X-10) < 1e-9 && math.Abs(got.Y-100) < 1e-9 {
		t.Error("the same answer for the stages in the other order: the test cannot tell them apart")
	}
	// A stage that passed its image on unchanged adds nothing to the chain.
	if got := chain.Then(nil); len(got.Maps) != len(chain.Maps) {
		t.Error("Then(nil) added a map")
	}
	// Then does not touch the chain it was called on.
	longer := chain.Then(Offset{1, 1})
	if len(chain.Maps) != 2 || len(longer.Maps) != 3 {
		t.Errorf("chain %d, longer %d", len(chain.Maps), len(longer.Maps))
	}
}

// A stage with no way back must not be silently dropped: the stages around it are still correct, so
// the chain would keep answering - with a point that quietly belongs to another place on the photo.
func TestAStageWithNoWayBackMakesTheWholeChainNotKnown(t *testing.T) {
	chain := Chain{[]Geometry{Scale{0.5, 0.5}, Unknown{}, Offset{3, 4}}}
	if _, ok := chain.ToInput([]Point{{10, 10}}); ok {
		t.Error("a chain with Unknown answered")
	}
	// Not identity: the stage did change the image, it is the way back that is gone.
	if _, ok := chain.Then(Unknown{}).ToInput([]Point{{10, 10}}); ok {
		t.Error("Then(Unknown) answered")
	}
	if _, ok := (Pieces{[]Piece{{Rect: [4]float64{0, 0, 10, 10}, Geo: Unknown{}}}}).ToInput([]Point{{1, 1}}); ok {
		t.Error("a piece with Unknown answered")
	}
	// And a chain without it still answers.
	near(t, one(t, Chain{[]Geometry{Scale{0.5, 0.5}}}, Point{10, 10}), Point{20, 20}, 1e-9)
}

// OpenCV's warps put pixel centres on integer coordinates: dst(x, y) = src(M^-1 (x, y)) in INDEX
// coordinates, so a continuous point is shifted by half a pixel in and out.
func TestHomographyKeepsTheHalfPixelConvention(t *testing.T) {
	shift := NewAffine([2][3]float64{{1, 0, 10}, {0, 1, 5}})
	near(t, one(t, shift, Point{20.5, 15.5}), Point{10.5, 10.5}, 1e-9)
	double := NewHomography([3][3]float64{{2, 0, 0}, {0, 2, 0}, {0, 0, 1}})
	// the output pixel (4, 4) sampled the input pixel (2, 2): centres 4.5 and 2.5
	near(t, one(t, double, Point{4.5, 4.5}), Point{2.5, 2.5}, 1e-9)
	// a real perspective: the inverse really inverts
	m := [3][3]float64{{1.1, 0.05, 12}, {-0.03, 0.95, -7}, {1e-4, -2e-4, 1}}
	h := NewHomography(m)
	src := Point{300.5, 210.5}
	x, y := src.X-0.5, src.Y-0.5
	w := m[2][0]*x + m[2][1]*y + m[2][2]
	out := Point{(m[0][0]*x+m[0][1]*y+m[0][2])/w + 0.5, (m[1][0]*x+m[1][1]*y+m[1][2])/w + 0.5}
	near(t, one(t, h, out), src, 1e-9)
}

// The bend map is the way back, read at the right pixel: bilinear between pixel centres, held at
// the edge, moving a point along y only.
func TestVerticalRemapReadsTheMapBilinearly(t *testing.T) {
	w, h := 6, 4
	v := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v[y*w+x] = float32(2*x + 10*y) // not uniform: a pixel read wrongly is plain to see
		}
	}
	r := VerticalRemap{V: v, W: w, H: h}
	// exactly on the centre of pixel (3, 2): v = 6 + 20
	near(t, one(t, r, Point{3.5, 2.5}), Point{3.5, 2.5 + 26}, 1e-6)
	// between four pixel centres: the mean of v(1,1)=12, v(2,1)=14, v(1,2)=22, v(2,2)=24
	near(t, one(t, r, Point{2.0, 2.0}), Point{2.0, 2.0 + 18}, 1e-6)
	// outside the map the edge is held
	near(t, one(t, r, Point{-5, -5}), Point{-5, -5 + 0}, 1e-6)
	near(t, one(t, r, Point{99, 99}), Point{99, 99 + float64(2*5+10*3)}, 1e-6)
}

// A stitched canvas picks the piece by the centroid of the shape, so the corners of one box are
// never sent to different pages.
func TestPiecesChooseByTheCentroid(t *testing.T) {
	pieces := Pieces{[]Piece{
		{Rect: [4]float64{0, 0, 100, 100}, Geo: Offset{1000, 0}},
		{Rect: [4]float64{100, 0, 200, 100}, Geo: Offset{0, 1000}},
	}}
	// three corners on the left piece, one beyond the border: the centroid decides
	box := []Point{{90, 10}, {95, 10}, {95, 20}, {120, 20}}
	got, ok := pieces.ToInput(box)
	if !ok {
		t.Fatal("not known")
	}
	near(t, got[3], Point{120 - 1000, 20}, 1e-9)
	// the centroid on the right piece
	got, _ = pieces.ToInput([]Point{{150, 50}})
	near(t, got[0], Point{150, 50 - 1000}, 1e-9)
}

// stitched_geometry: each page's own map, then its resize and its place on the canvas.
func TestStitchedGeometryPutsEachPageBack(t *testing.T) {
	left := Offset{DX: -10, DY: -20} // page = input shifted
	right := Scale{SX: 2, SY: 2}
	canvas := StitchedGeometry([]PlacedPage{
		{W: 100, H: 80, Scale: 1.0, DX: 0, DY: 0, Geo: left},
		{W: 50, H: 40, Scale: 2.0, DX: 100, DY: 0, Geo: right}, // resized to 100 x 80, placed at x = 100
	})
	// a point of the left piece: page (30, 40) -> input (30 - (-10)... Offset subtracts its shift
	near(t, one(t, canvas, Point{30, 40}), Point{40, 60}, 1e-9)
	// the right piece: canvas (150, 40) -> page (25, 20) -> input (12.5, 10)
	near(t, one(t, canvas, Point{150, 40}), Point{12.5, 10}, 1e-9)
	// the rounding of stitch_pages: int(round(w * scale)), at least 1
	if w, h := PlacedSize(33, 21, 1.5); w != 50 || h != 32 { // 49.5 -> 50 (even), 31.5 -> 32
		t.Errorf("placed %d x %d", w, h)
	}
}
