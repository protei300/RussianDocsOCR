package pipeline

import (
	"math"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// The way a field read on the canvas is put back on the photo (Pipeline._field_quads), without
// any model: the frames, the canvas chain and the word boxes are given.

func approx(t *testing.T, got, want geometry.Point) {
	t.Helper()
	if math.Abs(got.X-want.X) > 1e-9 || math.Abs(got.Y-want.Y) > 1e-9 {
		t.Errorf("got (%v, %v), want (%v, %v)", got.X, got.Y, want.X, want.Y)
	}
}

// a canvas that is the photo cut at (100, 50) and halved: canvas = (photo - (100, 50)) / 2
func canvasChain() geometry.Chain {
	return geometry.Chain{}.Then(geometry.Offset{DX: -100, DY: -50}).Then(geometry.Scale{SX: 0.5, SY: 0.5})
}

func TestAFieldQuadStartsAtTheBoxTopLeftAndLandsOnThePhoto(t *testing.T) {
	canvas := canvasChain()
	// a box (20, 10) - (60, 30) of the canvas, cut as it is
	frames := []fieldFrame{{Map: geometry.Offset{DX: -20, DY: -10}, W: 40, H: 20}}
	records := []SplitRecord{{Index: 0, Label: "Reg_number", PatchW: 40, PatchH: 20}}

	q := buildQuads(records, frames, canvas.ToInput, false)

	if !q.Known() {
		t.Fatal("not known")
	}
	got := q.Fields["Reg_number"][0]
	// canvas (20,10) -> photo (20/0.5 + 100, 10/0.5 + 50) = (140, 70); the box is 80 x 40 there
	approx(t, got[0], geometry.Point{X: 140, Y: 70})
	approx(t, got[1], geometry.Point{X: 220, Y: 70})
	approx(t, got[2], geometry.Point{X: 220, Y: 110})
	approx(t, got[3], geometry.Point{X: 140, Y: 110})
	// a field that needs no splitting: its whole patch is the single word, the same quad
	if w := q.Words["Reg_number"]; len(w) != 1 || w[0] != got {
		t.Errorf("words = %v", w)
	}
}

func TestEachWordBoxOfASplitFieldGetsItsOwnQuad(t *testing.T) {
	canvas := canvasChain()
	frames := []fieldFrame{{Map: geometry.Offset{DX: -20, DY: -10}, W: 40, H: 20}}
	records := []SplitRecord{{Index: 0, Label: "Last_name_ru", PatchW: 40, PatchH: 20, Split: true,
		WordBoxes: []postprocess.Box{{X1: 2.9, Y1: 1, X2: 10, Y2: 19}, {X1: 12, Y1: -3, X2: 99, Y2: 25}}}}

	q := buildQuads(records, frames, canvas.ToInput, false)

	words := q.Words["Last_name_ru"]
	if len(words) != 2 {
		t.Fatalf("%d word quads", len(words))
	}
	// the first word box (truncated to 2..10 x 1..19) inside the patch at (20, 10)
	approx(t, words[0][0], geometry.Point{X: (20+2)/0.5 + 100, Y: (10+1)/0.5 + 50})
	approx(t, words[0][2], geometry.Point{X: (20+10)/0.5 + 100, Y: (10+19)/0.5 + 50})
	// the second is clipped to the patch like the reference: x 12..40, y 0..20
	approx(t, words[1][0], geometry.Point{X: (20+12)/0.5 + 100, Y: (10+0)/0.5 + 50})
	approx(t, words[1][2], geometry.Point{X: (20+40)/0.5 + 100, Y: (10+20)/0.5 + 50})
	if len(q.Fields["Last_name_ru"]) != 1 {
		t.Error("one quad per detection for the field")
	}
}

// The passport series is turned a quarter after it is cut: the words are found on the turned
// patch, the field is not.
func TestTheTurnedSeriesPutsItsWordsBackThroughTheTurn(t *testing.T) {
	canvas := geometry.Chain{} // the canvas is the photo
	// cut as 30 wide, 100 tall at (50, 60); turned once counter-clockwise it is 100 wide, 30 tall
	frames := []fieldFrame{{Map: geometry.Offset{DX: -50, DY: -60}, W: 30, H: 100}}
	records := []SplitRecord{{Index: 0, Label: "Licence_number", PatchW: 100, PatchH: 30, Split: true,
		WordBoxes: []postprocess.Box{{X1: 10, Y1: 5, X2: 40, Y2: 25}}}}

	q := buildQuads(records, frames, canvas.ToInput, true)

	word := q.Words["Licence_number"][0]
	// One quarter turn sends (x, y) of a W-wide image to (y, W - x); undone for the 30-wide cut
	// patch it is (x, y) -> (30 - y, x): corner (10, 5) of the turned patch is (25, 10) of the cut
	// patch, (75, 70) on the photo; corner (40, 25) is (5, 40) of the cut patch, (55, 100).
	approx(t, word[0], geometry.Point{X: 75, Y: 70})
	approx(t, word[2], geometry.Point{X: 55, Y: 100})
}

// "Not known" is said once, for the whole run.
func TestAnUnknownWayBackIsSaidOnceForTheRun(t *testing.T) {
	canvas := geometry.Chain{}.Then(geometry.Unknown{})
	frames := []fieldFrame{{Map: geometry.Offset{}, W: 10, H: 10}, {Map: geometry.Offset{}, W: 10, H: 10}}
	records := []SplitRecord{{Index: 0, Label: "A"}, {Index: 1, Label: "B"}}

	q := buildQuads(records, frames, canvas.ToInput, false)

	if q.Known() {
		t.Error("an unknown way back answered")
	}
	payload := QuadsPayload(q).(map[string]any)
	if payload["fields"] != nil || payload["words"] != nil {
		t.Errorf("payload %v: both must be null", payload)
	}
}

func TestThePayloadHasTheStageShape(t *testing.T) {
	canvas := geometry.Chain{}
	frames := []fieldFrame{{Map: geometry.Offset{DX: -5, DY: -6}, W: 10, H: 20}}
	q := buildQuads([]SplitRecord{{Index: 0, Label: "X"}}, frames, canvas.ToInput, false)

	payload := QuadsPayload(q).(map[string]any)
	fields := payload["fields"].(map[string][][4][2]float64)["X"]
	if len(fields) != 1 || fields[0][0] != [2]float64{5, 6} || fields[0][2] != [2]float64{15, 26} {
		t.Errorf("fields = %v", fields)
	}
	if _, ok := payload["words"].(map[string][][4][2]float64); !ok {
		t.Error("words is not an object of quads")
	}
}
