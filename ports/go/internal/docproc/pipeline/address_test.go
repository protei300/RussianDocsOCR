package pipeline

import (
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/geometry"
)

func TestANumberTokenIsDigitDominated(t *testing.T) {
	cases := map[string]bool{
		"16":       true,
		"76":       true,
		"1К":       true, // a digit and a letter: not outnumbered
		"КВ76":     true, // two letters, two digits: digits are not outnumbered
		"ПСКОВ":    false,
		"":         false,
		"...":      false,
		"12А":      true,
		"1АБВ":     false,
		"КВ.":      false,
		"7":        true,
		"УЛ.1":     false,
		"ДОМ 12 К": false,
	}
	for text, want := range cases {
		if got := IsNumberToken(text); got != want {
			t.Errorf("IsNumberToken(%q) = %v, want %v", text, got, want)
		}
	}
}

func quadAt(cx, cy float64) []geometry.Point {
	return []geometry.Point{{X: cx - 10, Y: cy - 5}, {X: cx + 10, Y: cy - 5}, {X: cx + 10, Y: cy + 5}, {X: cx - 10, Y: cy + 5}}
}

// On one page the page is filled top to bottom, so the freshest stamp is the lowest.
func TestStampsAreOrderedFreshestFirstOnAPage(t *testing.T) {
	quads := [][]geometry.Point{quadAt(100, 50), quadAt(100, 300), quadAt(100, 180)}
	got := orderStamps(quads, 400, 500)
	want := []int{1, 2, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order %v, want %v", got, want)
		}
	}
}

// A canvas wider than 1.2 heights is a spread: the left page is filled before the right one.
func TestOnASpreadTheRightPageIsFresher(t *testing.T) {
	quads := [][]geometry.Point{quadAt(700, 50), quadAt(100, 400), quadAt(120, 100)}
	got := orderStamps(quads, 1000, 500)
	want := []int{0, 1, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order %v, want %v", got, want)
		}
	}
	// the same positions on a narrow canvas are one page, ordered by height alone
	single := orderStamps(quads, 600, 500)
	if single[0] != 1 {
		t.Fatalf("on one page the lowest stamp is the freshest, got %v", single)
	}
}

func TestADeregisteredStampCancelsTheNearestEarlierRegistration(t *testing.T) {
	stamps := []Stamp{
		{Kind: "registered"},   // 0: freshest, nothing cancels it
		{Kind: "deregistered"}, // 1: cancels 2
		{Kind: "registered"},   // 2
		{Kind: "registered"},   // 3
	}
	tieCancellations(stamps)
	if stamps[1].Cancels == nil || *stamps[1].Cancels != 2 {
		t.Fatalf("cancels %v", stamps[1].Cancels)
	}
	if !stamps[0].Current || stamps[2].Current || stamps[3].Current {
		t.Errorf("current: %v %v %v", stamps[0].Current, stamps[2].Current, stamps[3].Current)
	}
}

func TestADeregisteredStampThatOpensTheCanvasCancelsNothing(t *testing.T) {
	stamps := []Stamp{{Kind: "deregistered"}, {Kind: "unclear"}}
	tieCancellations(stamps)
	if stamps[0].Cancels != nil {
		t.Fatalf("cancels %v", *stamps[0].Cancels)
	}
	if stamps[0].Current || stamps[1].Current {
		t.Error("no registration, so none is current")
	}
}

func TestACancelledRegistrationIsNotCurrent(t *testing.T) {
	stamps := []Stamp{{Kind: "deregistered"}, {Kind: "registered"}, {Kind: "registered"}}
	tieCancellations(stamps)
	if stamps[1].Current || !stamps[2].Current {
		t.Fatalf("current: %v %v", stamps[1].Current, stamps[2].Current)
	}
}

func TestALineIsInTheStampItsCentreLiesIn(t *testing.T) {
	quads := [][]geometry.Point{quadAt(100, 100), quadAt(300, 100)}
	if got := containingStamp(geometry.Point{X: 305, Y: 102}, quads); got != 1 {
		t.Errorf("got %d", got)
	}
	if got := containingStamp(geometry.Point{X: 200, Y: 100}, quads); got != -1 {
		t.Errorf("a point between the stamps belongs to none, got %d", got)
	}
	// overlapping stamps: the nearer centre wins
	overlap := [][]geometry.Point{quadAt(100, 100), quadAt(112, 100)}
	if got := containingStamp(geometry.Point{X: 108, Y: 100}, overlap); got != 1 {
		t.Errorf("got %d", got)
	}
}

func TestTheStampStageIsNullWithoutADetector(t *testing.T) {
	if got := addressStampsPayload(&AddressResult{}); got != nil {
		t.Fatalf("%v", got)
	}
	empty := addressStampsPayload(&AddressResult{StampsRan: true})
	if list, ok := empty.([]any); !ok || len(list) != 0 {
		t.Fatalf("a detector that found nothing says [], got %#v", empty)
	}
}

func TestTheLinesCarryAStampKeyOnlyWhenTheDetectorRan(t *testing.T) {
	text := "Г. ПСКОВ"
	a := &AddressResult{Lines: []AddressLine{{Kind: "printed", Text: &text}}}
	if _, has := addressLinesPayload(a)[0].(map[string]any)["stamp"]; has {
		t.Error("stamp key without a stamp detector")
	}
	a.StampsRan = true
	if v, has := addressLinesPayload(a)[0].(map[string]any)["stamp"]; !has || v != nil {
		t.Errorf("stamp %v %v", v, has)
	}
}

func TestTheQuadsStageCarriesTheAddressLinesOnlyWhenTheAddressPathRan(t *testing.T) {
	plain := QuadsPayload(&Quads{Fields: map[string][]Quad{}, Words: map[string][]Quad{}}).(map[string]any)
	if _, has := plain["address_lines"]; has {
		t.Error("address_lines without the address path")
	}
	q := &Quads{Fields: map[string][]Quad{}, Words: map[string][]Quad{}, HasAddressLines: true}
	if v, has := QuadsPayload(q).(map[string]any)["address_lines"]; !has || v != nil {
		t.Errorf("an unknown way back is null, got %v", v)
	}
	q.AddressLines = []Quad{{{X: 1, Y: 2}, {X: 3, Y: 2}, {X: 3, Y: 4}, {X: 1, Y: 4}}}
	got := QuadsPayload(q).(map[string]any)["address_lines"].([][4][2]float64)
	if len(got) != 1 || got[0][2] != [2]float64{3, 4} {
		t.Errorf("%v", got)
	}
}
