package modules

import (
	"image"
	"image/color"
	"math"
	"testing"

	"gocv.io/x/gocv"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// Mirrors the OpenCV part of tests/test_geometry.py of the reference: every stage that changes
// the image knows the way back to its input. Checked with the port's own functions and OpenCV
// on drawings with a mark - the mark found on the output of a stage must land where it was
// drawn after the map is applied. A synthetic mark is the only ground truth that does not come
// from the maps themselves.
//
// COORDINATES ARE CONTINUOUS: the pixel (i, j) is the unit square centred at (i + 0.5, j + 0.5).
// Test code, so it draws with gocv directly.

var (
	background = color.RGBA{R: 110, G: 70, B: 40} // as the pixels are laid out: B, G, R in the mat
	paper      = color.RGBA{R: 235, G: 235, B: 235}
	markColour = color.RGBA{R: 230, G: 20, B: 20}
)

func newCanvas(w, h int, c color.RGBA) gocv.Mat {
	return gocv.NewMatWithSizeFromScalar(gocv.NewScalar(float64(c.B), float64(c.G), float64(c.R), 0), h, w, gocv.MatTypeCV8UC3)
}

func fillPage(m *gocv.Mat, quad []image.Point) []imaging.Point {
	pv := gocv.NewPointVectorFromPoints(quad)
	defer pv.Close()
	pvs := gocv.NewPointsVector()
	defer pvs.Close()
	pvs.Append(pv)
	_ = gocv.FillPoly(m, pvs, paper)
	out := make([]imaging.Point, len(quad))
	for i, p := range quad {
		out[i] = imaging.Point{X: float64(p.X), Y: float64(p.Y)}
	}
	return out
}

func mark(m *gocv.Mat, x, y, r int) { _ = gocv.Circle(m, image.Pt(x, y), r, markColour, -1) }

// markCentres are the centres of the mark-coloured pixels, one per cell of a coarse grid of
// `cells` columns x rows (the marks of a test are far apart), in continuous coordinates.
func markCentres(t *testing.T, img imaging.Image, cols, rows int) []geometry.Point {
	t.Helper()
	b, err := img.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	w, h := img.Width(), img.Height()
	type acc struct{ sx, sy, n float64 }
	cells := make([]acc, cols*rows)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (y*w + x) * 3
			if b[i+2] > 180 && b[i+1] < 90 && b[i] < 90 {
				c := (y*rows/h)*cols + x*cols/w
				cells[c].sx += float64(x)
				cells[c].sy += float64(y)
				cells[c].n++
			}
		}
	}
	var out []geometry.Point
	for _, c := range cells {
		if c.n >= 5 {
			out = append(out, geometry.Point{X: c.sx/c.n + 0.5, Y: c.sy/c.n + 0.5})
		}
	}
	if len(out) == 0 {
		t.Fatal("no mark on the image")
	}
	return out
}

func assertLands(t *testing.T, g geometry.Geometry, found []geometry.Point, want []geometry.Point, atol float64) {
	t.Helper()
	for _, w := range want {
		best := math.Inf(1)
		for _, f := range found {
			back, ok := g.ToInput([]geometry.Point{f})
			if !ok {
				t.Fatal("the way back is not known")
			}
			best = math.Min(best, math.Hypot(back[0].X-w.X, back[0].Y-w.Y))
		}
		if best > atol {
			t.Errorf("the mark that belongs at (%.1f, %.1f) lands %.2f px away (tolerance %.1f)", w.X, w.Y, best, atol)
		}
	}
}

func TestAStraightenedPageRoundTripsTheMark(t *testing.T) {
	m := newCanvas(900, 700, background)
	defer m.Close()
	contour := fillPage(&m, []image.Point{{120, 90}, {760, 140}, {700, 610}, {90, 560}})
	mark(&m, 400, 300, 5)
	img := imaging.Wrap(m.Clone())
	defer img.Close()

	pages, _, matrices := imaging.RectifyPagesMatrices(img, [][]imaging.Point{contour}, imaging.DocMarginFrac)
	if len(pages) != 1 {
		t.Fatalf("%d pages", len(pages))
	}
	defer pages[0].Close()
	assertLands(t, geometry.NewHomography(matrices[0]), markCentres(t, pages[0], 1, 1),
		[]geometry.Point{{X: 400.5, Y: 300.5}}, 1.0)
}

// Both pages, because a stitched canvas picks the piece by the centroid of the shape. Two pages of
// different size are resized to a common side before the stitch, so a box that is mapped through
// the wrong piece lands plausibly - on the other page.
func TestAStitchedSpreadRoundTripsTheMarkOfEachPage(t *testing.T) {
	cases := []struct {
		name   string
		w, h   int
		quads  [][]image.Point
		points [][2]int
		cols   int
		rows   int
	}{
		{"side by side", 1024, 760,
			[][]image.Point{{{40, 60}, {430, 80}, {420, 640}, {50, 620}}, {{480, 90}, {960, 70}, {980, 700}, {470, 660}}},
			[][2]int{{200, 300}, {700, 400}}, 2, 1},
		{"one above the other", 800, 1040,
			[][]image.Point{{{60, 40}, {700, 60}, {690, 420}, {70, 400}}, {{80, 470}, {720, 460}, {740, 1000}, {60, 980}}},
			[][2]int{{300, 200}, {400, 700}}, 1, 2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := newCanvas(c.w, c.h, background)
			defer m.Close()
			var contours [][]imaging.Point
			for _, q := range c.quads {
				contours = append(contours, fillPage(&m, q))
			}
			for _, p := range c.points {
				mark(&m, p[0], p[1], 5)
			}
			img := imaging.Wrap(m.Clone())
			defer img.Close()

			pages, quads, matrices := imaging.RectifyPagesMatrices(img, contours, imaging.DocMarginFrac)
			if len(pages) != 2 {
				t.Fatalf("%d pages", len(pages))
			}
			placed := make([]geometry.PlacedPage, 2)
			sizes := [][2]int{{pages[0].Width(), pages[0].Height()}, {pages[1].Width(), pages[1].Height()}}
			canvas, placements, ok := imaging.StitchPages(pages, quads, imaging.StackAuto)
			if !ok {
				t.Fatal("no stitch")
			}
			defer canvas.Close()
			for i := range placed {
				placed[i] = geometry.PlacedPage{W: sizes[i][0], H: sizes[i][1], Scale: placements[i].Scale,
					DX: placements[i].DX, DY: placements[i].DY, Geo: geometry.NewHomography(matrices[i])}
			}
			var want []geometry.Point
			for _, p := range c.points {
				want = append(want, geometry.Point{X: float64(p[0]) + 0.5, Y: float64(p[1]) + 0.5})
			}
			assertLands(t, geometry.StitchedGeometry(placed), markCentres(t, canvas, c.cols, c.rows), want, 1.5)
		})
	}
}

func tiltedLines(m *gocv.Mat, top, bottom int, degrees float64) {
	for row := top; row < bottom; row += 28 {
		rise := int(math.Round(640 * math.Tan(degrees*math.Pi/180)))
		_ = gocv.Line(m, image.Pt(80, row), image.Pt(720, row+rise), color.RGBA{R: 20, G: 20, B: 20}, 5)
	}
}

func TestDeskewRoundTripsTheMark(t *testing.T) {
	m := newCanvas(800, 600, color.RGBA{R: 245, G: 245, B: 245})
	defer m.Close()
	tiltedLines(&m, 40, 540, 5.0)
	mark(&m, 433, 287, 5)
	img := imaging.Wrap(m.Clone())
	defer img.Close()
	deskewer := NewPipelineDeskewer()

	straight, _, g, err := deskewer.DeskewWithGeometry(img)
	if err != nil {
		t.Fatal(err)
	}
	defer straight.Close()
	if g == nil {
		t.Fatal("5 degrees is above min_angle, so the image is turned")
	}
	assertLands(t, g, markCentres(t, straight, 1, 1), []geometry.Point{{X: 433.5, Y: 287.5}}, 1.0)

	flat := imaging.Wrap(newCanvas(800, 600, color.RGBA{R: 245, G: 245, B: 245}))
	defer flat.Close()
	same, _, none, err := deskewer.DeskewWithGeometry(flat)
	if err != nil {
		t.Fatal(err)
	}
	defer same.Close()
	// Nothing to turn: the image is passed on as it is, and the stage adds no map.
	if none != nil {
		t.Error("an untouched image contributed a map")
	}
}

// bend is the shape line_dewarp fits: a bow across the page, row-major H x W.
func bend(w, h int, amplitude float32) []float32 {
	v := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			xs := (float32(x) - float32(w)/2) / (float32(w) / 2)
			ys := (float32(y) - float32(h)/2) / (float32(w) / 2)
			v[y*w+x] = amplitude * (1 - xs*xs) * (0.5 + ys)
		}
	}
	return v
}

// Against the remap itself: the bend map is the way back, read at the right pixel. The marks are
// off-centre on purpose: the map is not uniform, and a mark read at a wrong pixel of it lands a
// pixel or two away.
func TestAnUnbentPageRoundTripsTheMarkThroughTheBendMap(t *testing.T) {
	for _, amplitude := range []float32{4, 12} {
		m := newCanvas(600, 400, paper)
		mark(&m, 170, 290, 4)
		mark(&m, 450, 110, 4)
		img := imaging.Wrap(m)
		v := bend(600, 400, amplitude)

		unbent := ApplyDewarp(img, v)
		assertLands(t, geometry.VerticalRemap{V: v, W: 600, H: 400}, markCentres(t, unbent, 2, 2),
			[]geometry.Point{{X: 170.5, Y: 290.5}, {X: 450.5, Y: 110.5}}, 1.0)
		unbent.Close()
		img.Close()
	}
}

// The registration's straightening: a homography, then the bend map on its result. The chain is
// read output -> input, so the order the stages ran is the order of Maps.
func TestAStraightenedAndUnbentPageChainsBothMaps(t *testing.T) {
	m := newCanvas(800, 500, paper)
	mark(&m, 300, 260, 4)
	img := imaging.Wrap(m)
	defer img.Close()
	Hm, ok := imaging.SolvePerspective4(
		[4]imaging.Point{{X: 0, Y: 0}, {X: 800, Y: 0}, {X: 800, Y: 500}, {X: 0, Y: 500}},
		[4]imaging.Point{{X: 30, Y: 15}, {X: 780, Y: -20}, {X: 815, Y: 520}, {X: -15, Y: 480}})
	if !ok {
		t.Fatal("no perspective")
	}
	v := bend(800, 500, 16)

	refined := ApplyRefinement(img, Hm)
	defer refined.Close()
	page := ApplyDewarp(refined, v)
	defer page.Close()

	forward := geometry.Chain{Maps: []geometry.Geometry{geometry.NewHomography(Hm), geometry.VerticalRemap{V: v, W: 800, H: 500}}}
	assertLands(t, forward, markCentres(t, page, 1, 1), []geometry.Point{{X: 300.5, Y: 260.5}}, 1.0)
}
