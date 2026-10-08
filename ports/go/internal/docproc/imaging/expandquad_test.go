package imaging

import (
	"math"
	"testing"
)

// ExpandQuadF32 must land on float32 values (the reference expands a float32 array) and stay
// within float32 noise of the float64 version.
func TestExpandQuadF32IsFloat32AndCloseToFloat64(t *testing.T) {
	quad := []Point{{X: 103, Y: 211}, {X: 1017, Y: 188}, {X: 1041, Y: 900}, {X: 77, Y: 925}}
	got := ExpandQuadF32(quad, 0.01)
	want := ExpandQuad(quad, 0.01)
	for i := range got {
		if got[i].X != float64(float32(got[i].X)) || got[i].Y != float64(float32(got[i].Y)) {
			t.Errorf("corner %d is not float32-representable: %v", i, got[i])
		}
		if math.Abs(got[i].X-want[i].X) > 1e-3 || math.Abs(got[i].Y-want[i].Y) > 1e-3 {
			t.Errorf("corner %d: %v vs float64 %v", i, got[i], want[i])
		}
	}
	if same := ExpandQuadF32(quad, 0); &same[0] != &quad[0] {
		t.Errorf("margin 0 must return the input unchanged")
	}
}
