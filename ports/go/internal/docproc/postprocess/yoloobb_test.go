package postprocess

import (
	"math"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// obbHead builds the raw head of an OBB export, [1, 4 + nc + 1, anchors], from rows
// (cx, cy, w, h, score per class..., angle) - one row per anchor.
func obbHead(t *testing.T, nc int, rows [][]float32) *tensor.Array {
	t.Helper()
	feats := 4 + nc + 1
	// A real head has thousands of anchors and the decoder takes the longer axis for them
	// (np: "ensure [n_anchors, features]"); pad with silent anchors so that holds here too.
	for len(rows) < 4*feats {
		rows = append(rows, make([]float32, feats))
	}
	flat := make([]float32, feats*len(rows))
	for a, row := range rows {
		if len(row) != feats {
			t.Fatalf("row %d has %d columns, want %d", a, len(row), feats)
		}
		for f, v := range row {
			flat[f*len(rows)+a] = v // features first, as the export lays it out
		}
	}
	arr, err := tensor.Float32Of([]int{1, feats, len(rows)}, flat)
	if err != nil {
		t.Fatal(err)
	}
	return arr
}

func newObb(t *testing.T, labels ...string) *YoloObb {
	t.Helper()
	y, err := NewYoloObb(labels, 0.45, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	return y
}

func decodeObb(t *testing.T, y *YoloObb, head *tensor.Array, ctx Context) []ObbBox {
	t.Helper()
	res, err := y.Apply(head, ctx)
	if err != nil {
		t.Fatal(err)
	}
	return res.(ObbResult).Boxes
}

func TestAnObbHeadIsFilteredSuppressedMappedAndOrdered(t *testing.T) {
	y := newObb(t, "A", "B")
	// the letterbox: ratio 0.5, padded 20 px down, none across
	ctx := Context{Ratio: 0.5, PadLetter: [2]float64{0, 20}, PadExtra: [2]int{0, 0}}
	head := obbHead(t, 2, [][]float32{
		{200, 120, 100, 20, 0.90, 0.10, 0.0}, // kept (class A)
		{202, 121, 100, 20, 0.80, 0.10, 0.0}, // the same box again: suppressed
		{200, 80, 100, 20, 0.10, 0.70, 0.0},  // kept (class B), ABOVE the first
		{50, 50, 10, 10, 0.20, 0.20, 0.0},    // below the confidence threshold
	})
	got := decodeObb(t, y, head, ctx)
	if len(got) != 2 {
		t.Fatalf("%d boxes, want 2: %+v", len(got), got)
	}
	// ordered top to bottom: the class-B box (cy 80 -> (80-20)/0.5 = 120) comes first
	if got[0].Label != "B" || got[1].Label != "A" {
		t.Fatalf("order %s, %s", got[0].Label, got[1].Label)
	}
	if got[1].Cx != 400 || got[1].Cy != 200 || got[1].W != 200 || got[1].H != 40 {
		t.Errorf("the box is not mapped back out of the letterbox: %+v", got[1])
	}
	if got[1].Cls != 0 || got[0].Cls != 1 {
		t.Errorf("classes %d, %d", got[1].Cls, got[0].Cls)
	}
	if math.Abs(got[0].Conf-0.70) > 1e-6 {
		t.Errorf("conf %v", got[0].Conf)
	}
}

// The suppression ignores the class (the stamps of one canvas overlap across classes only when
// one of them is wrong), and a box turned a quarter of a turn does not overlap an upright one
// of the same centre by enough to be suppressed.
func TestObbSuppressionIsByRotatedOverlapNotByClass(t *testing.T) {
	y := newObb(t, "A", "B")
	ctx := Context{Ratio: 1}
	head := obbHead(t, 2, [][]float32{
		{100, 100, 80, 20, 0.9, 0.0, 0.0},
		{100, 100, 80, 20, 0.0, 0.8, 0.0},                  // same box, other class: suppressed
		{100, 100, 80, 20, 0.7, 0.0, float32(math.Pi / 2)}, // the same rectangle turned: kept
	})
	got := decodeObb(t, y, head, ctx)
	if len(got) != 2 {
		t.Fatalf("%d boxes, want 2: %+v", len(got), got)
	}
}

func TestAnEmptyObbHeadGivesNoBoxes(t *testing.T) {
	y := newObb(t, "A")
	got := decodeObb(t, y, obbHead(t, 1, [][]float32{{10, 10, 5, 5, 0.1, 0}}), Context{Ratio: 1})
	if len(got) != 0 {
		t.Fatalf("%+v", got)
	}
}

// Rows of 10 px, then left to right: two boxes 4 px apart vertically are one row and are told
// apart by x; the ratio of the rounding is half-to-even.
func TestObbReadingOrderUsesRowsOfTenPixels(t *testing.T) {
	y := newObb(t, "A")
	head := obbHead(t, 1, [][]float32{
		{300, 102, 40, 10, 0.9, 0},
		{100, 98, 40, 10, 0.8, 0},
		{200, 140, 40, 10, 0.7, 0},
	})
	got := decodeObb(t, y, head, Context{Ratio: 1})
	if len(got) != 3 || got[0].Cx != 100 || got[1].Cx != 300 || got[2].Cx != 200 {
		t.Fatalf("%+v", got)
	}
}
