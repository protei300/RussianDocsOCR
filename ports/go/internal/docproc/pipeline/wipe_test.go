package pipeline

import (
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// Results.Wipe zeroes every image the run kept — the intermediates and the canvas — before
// releasing them (ports/MEMORY.md §5). Looked at through shared handles, because the Results'
// own handles are released by then.
func TestResultsWipeZeroesEveryImage(t *testing.T) {
	prepared := imaging.NewFilled(20, 30, 7, 7, 7)
	upright := imaging.NewFilled(30, 20, 9, 9, 9)
	canvas := imaging.NewFilled(40, 50, 5, 5, 5)
	seen := []imaging.Image{prepared.Share(), upright.Share(), canvas.Share()}
	defer func() {
		for _, s := range seen {
			_ = s.Close()
		}
	}()

	res := &Results{Canvas: canvas, HasCanvas: true, owned: []imaging.Image{prepared, upright}}
	res.Wipe()
	for i, s := range seen {
		if !s.IsAllZero() {
			t.Errorf("image %d was released unzeroed", i)
		}
	}
	if res.HasCanvas || res.owned != nil {
		t.Error("Wipe left the Results holding images")
	}
	res.Close() // a no-op now, and safe
}
