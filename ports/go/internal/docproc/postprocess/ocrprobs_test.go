package postprocess

import (
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// probs builds a [1,T,C] tensor from per-step rows.
func probs(t *testing.T, rows [][]float32) *tensor.Array {
	t.Helper()
	if len(rows) == 0 {
		t.Fatal("need at least one step")
	}
	classes := len(rows[0])
	flat := make([]float32, 0, len(rows)*classes)
	for _, r := range rows {
		if len(r) != classes {
			t.Fatalf("ragged rows: %d vs %d", len(r), classes)
		}
		flat = append(flat, r...)
	}
	a, err := tensor.Float32Of([]int{1, len(rows), classes}, flat)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func allowedOf(s string) map[rune]bool {
	set := map[rune]bool{}
	for _, r := range s {
		set[r] = true
	}
	return set
}

// The alphabet is indexed by RUNE. Cyrillic is multi-byte UTF-8, so a port that indexed
// the string by byte would emit fragments of characters -- mojibake with certainty, not
// by chance. This is the trap the whole spike was built around.
func TestDecodeIndexesAlphabetByRune(t *testing.T) {
	// classes: 0=blank, 1='А', 2='Б', 3='В'
	p, err := NewOcrProbs("АБВ", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.Decode(probs(t, [][]float32{
		{0.1, 0.9, 0.0, 0.0},
		{0.1, 0.0, 0.9, 0.0},
		{0.1, 0.0, 0.0, 0.9},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got != "АБВ" {
		t.Fatalf("got %q (% x), want %q", got, got, "АБВ")
	}
}

// Greedy CTC: consecutive equal indices collapse to one character, and a blank BETWEEN
// them preserves the doubling. Getting this wrong turns "АА" into "А" or vice versa.
func TestDecodeCollapsesRepeatsButNotAcrossBlank(t *testing.T) {
	p, _ := NewOcrProbs("АБ", nil, 0)

	same, err := p.Decode(probs(t, [][]float32{
		{0.0, 0.9, 0.1},
		{0.0, 0.9, 0.1},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if same != "А" {
		t.Fatalf("repeats must collapse: got %q", same)
	}

	split, err := p.Decode(probs(t, [][]float32{
		{0.0, 0.9, 0.1},
		{0.9, 0.0, 0.1},
		{0.0, 0.9, 0.1},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if split != "АА" {
		t.Fatalf("a blank between repeats must preserve both: got %q", split)
	}
}

// np.argmax returns the FIRST maximum. On a tie a >= comparison returns the LAST, which
// selects a different class and changes a character.
func TestArgmaxFirstOnTies(t *testing.T) {
	if got := argmaxFirst([]float32{0.5, 0.5, 0.5}); got != 0 {
		t.Fatalf("a three-way tie must pick index 0, got %d", got)
	}
	if got := argmaxFirst([]float32{0.1, 0.7, 0.7}); got != 1 {
		t.Fatalf("a tie must pick the EARLIER index, got %d", got)
	}
}

// A disallowed character is SUBSTITUTED with the best allowed non-blank, not dropped.
// Here 'Ѣ' (disallowed) wins outright and must become 'И', the best allowed letter.
func TestDecodeSubstitutesDisallowedRatherThanDropping(t *testing.T) {
	// classes: 0=blank, 1='И', 2='Ѣ'
	p, err := NewOcrProbs("ИѢ", allowedOf("И"), 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.Decode(probs(t, [][]float32{
		{0.05, 0.15, 0.80},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got != "И" {
		t.Fatalf("a disallowed win must substitute the best allowed letter, got %q", got)
	}
}

// THE reason masking is written as -inf rather than as zeroing the disallowed columns:
// when the model is confident about the diacritic, BLANK is the runner-up, and zeroing
// lets blank win, silently DELETING the character.
//
// Blank here scores 0.30, higher than the allowed letter's 0.10, so a blank-eligible
// fallback yields "" while the correct behaviour yields "И".
func TestDecodeExcludesBlankFromSubstitution(t *testing.T) {
	p, _ := NewOcrProbs("ИѢ", allowedOf("И"), 0)
	got, err := p.Decode(probs(t, [][]float32{
		{0.30, 0.10, 0.60},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got == "" {
		t.Fatal("blank won the substitution: the character was deleted, which is the " +
			"exact bug -inf masking exists to prevent")
	}
	if got != "И" {
		t.Fatalf("want И, got %q", got)
	}
}

// An allowed character that wins is kept untouched, and blank winning still means blank
// -- the substitution path must not fire for either.
func TestDecodeLeavesAllowedAndBlankAlone(t *testing.T) {
	p, _ := NewOcrProbs("ИѢ", allowedOf("И"), 0)
	got, err := p.Decode(probs(t, [][]float32{
		{0.1, 0.8, 0.1},   // allowed letter wins
		{0.9, 0.05, 0.05}, // blank wins
		{0.1, 0.8, 0.1},   // same letter again, separated by the blank
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got != "ИИ" {
		t.Fatalf("want ИИ, got %q", got)
	}
}

// With no mask at all the decode is a plain argmax and every alphabet character is
// emittable, including ones a document would never be allowed to produce.
func TestDecodeWithoutMaskEmitsAnything(t *testing.T) {
	p, _ := NewOcrProbs("ИѢ", nil, 0)
	got, err := p.Decode(probs(t, [][]float32{{0.05, 0.15, 0.80}}))
	if err != nil {
		t.Fatal(err)
	}
	if got != "Ѣ" {
		t.Fatalf("an unmasked decode must emit the raw argmax, got %q", got)
	}
}

// A [T,C] tensor is accepted as well as [1,T,C], matching the reference's ndim check.
func TestDecodeAcceptsUnbatchedShape(t *testing.T) {
	p, _ := NewOcrProbs("АБ", nil, 0)
	a, err := tensor.Float32Of([]int{2, 3}, []float32{
		0.0, 0.9, 0.1,
		0.0, 0.1, 0.9,
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.Decode(a)
	if err != nil {
		t.Fatal(err)
	}
	if got != "АБ" {
		t.Fatalf("want АБ, got %q", got)
	}
}
