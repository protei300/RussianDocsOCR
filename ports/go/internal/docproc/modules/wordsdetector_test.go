package modules

import (
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// boxes builds detections from x1,y1,x2,y2 quads, in the order the detector emits them
// (already lexsorted y1 then x1 by the YOLO postprocessing).
func boxes(quads ...[4]float64) []postprocess.Box {
	out := make([]postprocess.Box, len(quads))
	for i, q := range quads {
		out[i] = postprocess.Box{X1: q[0], Y1: q[1], X2: q[2], Y2: q[3], Label: "Word"}
	}
	return out
}

func order(bs []postprocess.Box) [][4]float64 {
	out := make([][4]float64, len(bs))
	for i, b := range bs {
		out[i] = [4]float64{b.X1, b.Y1, b.X2, b.Y2}
	}
	return out
}

// A two-line field must be read line by line. A plain x-sort interleaves the lines, which
// is what the reference measured as word salad on the birth certificates' Birth_place and
// ZAGS fields. Expected order taken from WordsDetector._reading_order on these boxes.
func TestReadingOrderKeepsLinesTogether(t *testing.T) {
	in := boxes(
		[4]float64{10, 0, 60, 18},   // line 1, word 1
		[4]float64{70, 1, 130, 19},  // line 1, word 2
		[4]float64{140, 0, 200, 18}, // line 1, word 3
		[4]float64{5, 22, 55, 40},   // line 2, word 1
		[4]float64{65, 23, 190, 41}, // line 2, word 2
	)
	want := [][4]float64{{10, 0, 60, 18}, {70, 1, 130, 19}, {140, 0, 200, 18},
		{5, 22, 55, 40}, {65, 23, 190, 41}}

	got := order(readingOrder(in))
	if len(got) != len(want) {
		t.Fatalf("got %d boxes, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("box %d: got %v, want %v (full: %v)", i, got[i], want[i], got)
		}
	}

	// And the naive sort really does disagree — a test that cannot fail proves nothing.
	// x1-sorted, the second line's first word would come SECOND, between two words of
	// the first line.
	if got[1] == ([4]float64{5, 22, 55, 40}) {
		t.Fatal("this fixture no longer distinguishes reading order from an x1 sort")
	}
}

// A single-line field must come out exactly as the old x1 sort produced it.
func TestReadingOrderIsAnX1SortOnOneLine(t *testing.T) {
	in := boxes(
		[4]float64{140, 0, 200, 18},
		[4]float64{10, 2, 60, 20},
		[4]float64{70, 1, 130, 19},
	)
	want := [][4]float64{{10, 2, 60, 20}, {70, 1, 130, 19}, {140, 0, 200, 18}}
	got := order(readingOrder(in))
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("box %d: got %v, want %v", i, got[i], want[i])
		}
	}
}
