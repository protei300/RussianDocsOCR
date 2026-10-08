package imaging

import (
	"math"
	"testing"
)

// Mirrors test_the_fill_paints_past_the_photo_instead_of_smearing_its_edge of
// tests/test_card_registration.py: the default warp repeats the edge pixels, a fill paints.
func TestFillPaintsPastThePhotoInsteadOfSmearingItsEdge(t *testing.T) {
	photo := NewFilled(200, 100, 255, 255, 255) // the whole of it is "card"
	defer photo.Close()
	identity := Identity3()
	smeared := WarpByHomography(photo, identity, 200, 200, true)
	defer smeared.Close()
	painted := WarpByHomographyFill(photo, identity, 200, 200, [3]uint8{10, 20, 30})
	defer painted.Close()
	sb, _ := smeared.Bytes()
	pb, _ := painted.Bytes()
	at := func(b []byte, x, y int) [3]byte { i := (y*200 + x) * 3; return [3]byte{b[i], b[i+1], b[i+2]} }
	if at(sb, 150, 100) != [3]byte{255, 255, 255} {
		t.Errorf("the default must repeat the edge, got %v", at(sb, 150, 100))
	}
	if at(pb, 150, 100) != [3]byte{10, 20, 30} {
		t.Errorf("the fill must paint, got %v", at(pb, 150, 100))
	}
	if at(pb, 50, 50) != [3]byte{255, 255, 255} {
		t.Errorf("inside the photo nothing changes, got %v", at(pb, 50, 50))
	}
}

// np.median of an even count is the mean of the two middle values and int() truncates it.
func TestPolygonMedianColourTruncatesTheHalf(t *testing.T) {
	img := NewFilled(10, 10, 100, 7, 255)
	defer img.Close()
	fill, ok := PolygonMedianColour(img, RoundPolygon([]Point{{X: 1, Y: 1}, {X: 8, Y: 1}, {X: 8, Y: 8}, {X: 1, Y: 8}}))
	if !ok || fill != [3]uint8{100, 7, 255} {
		t.Errorf("got %v %v", fill, ok)
	}
	if _, ok := PolygonMedianColour(img, nil); ok {
		t.Error("an empty polygon covers nothing")
	}
}

func TestRoundPolygonIsHalfToEven(t *testing.T) {
	got := RoundPolygon([]Point{{X: 0.5, Y: 1.5}, {X: 2.5, Y: -0.5}})
	if got[0].X != 0 || got[0].Y != 2 || got[1].X != 2 || got[1].Y != 0 {
		t.Errorf("got %v", got)
	}
}

// A similarity fitted to points that ARE a similarity of them leaves no residual.
func TestEstimateAffinePartialFindsTheSimilarity(t *testing.T) {
	from := []Point{{X: 0, Y: 0}, {X: 1000, Y: 0}, {X: 1000, Y: 1400}, {X: 0, Y: 1400}}
	a, b := 0.99*math.Cos(0.02), 0.99*math.Sin(0.02)
	to := make([]Point, len(from))
	for i, p := range from {
		to[i] = Point{X: a*p.X - b*p.Y + 30, Y: b*p.X + a*p.Y - 12}
	}
	A, ok := EstimateAffinePartialLMEDS(from, to)
	if !ok {
		t.Fatal("no model")
	}
	for i, p := range from {
		x := A[0][0]*p.X + A[0][1]*p.Y + A[0][2]
		y := A[1][0]*p.X + A[1][1]*p.Y + A[1][2]
		if math.Hypot(x-to[i].X, y-to[i].Y) > 1e-2 {
			t.Errorf("point %d off by %v", i, math.Hypot(x-to[i].X, y-to[i].Y))
		}
	}
}

// cv2.perspectiveTransform of float32 points answers in float32.
func TestPerspectiveTransformIsFloat32(t *testing.T) {
	H := [3][3]float64{{1.0000001, 0, 0.1}, {0, 1, 0.2}, {0, 0, 1}}
	got := PerspectiveTransformF32(H, []Point{{X: 3, Y: 4}})
	if len(got) != 1 || got[0].X != float64(float32(got[0].X)) || got[0].Y != float64(float32(got[0].Y)) {
		t.Errorf("got %v", got)
	}
}
