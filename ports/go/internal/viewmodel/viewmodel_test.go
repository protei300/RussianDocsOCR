package viewmodel

import (
	"encoding/json"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

func box(label string, conf float64) postprocess.Box {
	return postprocess.Box{Label: label, Conf: conf, X1: 1, Y1: 2, X2: 3, Y2: 4}
}

// The highest-confidence box of a label owns the text; the others are flagged ambiguous.
// This exists because the correspondence is NOT recoverable from the library's output --
// Licence_number is detected twice on internal passports and the pipeline deduplicates the
// field, not the boxes.
func TestAmbiguousMarksTheNonOwningDuplicate(t *testing.T) {
	boxes := buildBoxes(
		[]postprocess.Box{box("Licence_number", 0.71), box("Licence_number", 0.88)},
		map[string]string{"Licence_number": "69 18 812211"},
	)
	if boxes[0].Text != nil {
		t.Errorf("the lower-confidence box must not carry the text, got %q", *boxes[0].Text)
	}
	if !boxes[0].Ambiguous {
		t.Error("the lower-confidence box must be flagged ambiguous")
	}
	if boxes[1].Text == nil || *boxes[1].Text != "69 18 812211" {
		t.Error("the higher-confidence box must own the text")
	}
	if boxes[1].Ambiguous {
		t.Error("the owning box must not be flagged ambiguous")
	}
}

// A label with no recognised text is not ambiguous -- there is no text to be ambiguous
// ABOUT. Face is the everyday case: two boxes, no OCR, neither flagged.
func TestAmbiguousIsFalseWithoutOcrText(t *testing.T) {
	boxes := buildBoxes([]postprocess.Box{box("Face", 0.9), box("Face", 0.8)}, map[string]string{})
	for i, b := range boxes {
		if b.Ambiguous {
			t.Errorf("box %d: a label absent from the OCR dict cannot be ambiguous", i)
		}
		if b.Text != nil {
			t.Errorf("box %d: expected no text", i)
		}
	}
}

// Face and Signature are detected but never OCR'd, so they are visual and carry no text.
// A UI expecting every box to have a value renders them as broken rows otherwise.
func TestKindIsVisualForNonTextLabels(t *testing.T) {
	boxes := buildBoxes(
		[]postprocess.Box{box("Face", 0.9), box("Signature", 0.9), box("Last_name_ru", 0.9)},
		map[string]string{"Last_name_ru": "ИВАНОВ"},
	)
	want := []string{"visual", "visual", "text"}
	for i, b := range boxes {
		if b.Kind != want[i] {
			t.Errorf("box %d (%s): kind %q, want %q", i, b.Label, b.Kind, want[i])
		}
	}
}

// Fields come out in document READING order, which is neither alphabetical nor the
// library's insertion order.
func TestFieldsAreInReadingOrder(t *testing.T) {
	ocr := map[string]string{
		"Birth_date":     "06.01.1985",
		"Last_name_ru":   "ИВАНОВ",
		"Licence_number": "1234",
		"First_name_ru":  "ПЕТР",
	}
	fields := buildFields("SNILS_1996", ocr, nil)
	got := make([]string, len(fields))
	for i, f := range fields {
		got[i] = f.Name
	}
	want := []string{"Last_name_ru", "First_name_ru", "Birth_date", "Licence_number"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order %v, want %v", got, want)
		}
	}
}

// A field the labels table does not know about is APPENDED alphabetically rather than
// dropped, so the API degrades gracefully when the library gains a field first.
func TestUnknownFieldsAreAppendedAlphabetically(t *testing.T) {
	fields := buildFields("SNILS_1996", map[string]string{
		"Zzz_new_field": "x",
		"Last_name_ru":  "ИВАНОВ",
		"Aaa_new_field": "y",
	}, nil)
	got := make([]string, len(fields))
	for i, f := range fields {
		got[i] = f.Name
	}
	want := []string{"Last_name_ru", "Aaa_new_field", "Zzz_new_field"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order %v, want %v", got, want)
		}
	}
}

// A field's conf is the confidence of the box OWNING its text, not of any box sharing the
// label -- which is why the duplicate above must not overwrite it.
func TestFieldConfComesFromTheOwningBox(t *testing.T) {
	ocr := map[string]string{"Licence_number": "1234"}
	boxes := buildBoxes([]postprocess.Box{box("Licence_number", 0.71), box("Licence_number", 0.88)}, ocr)
	fields := buildFields("SNILS_1996", ocr, boxes)
	if len(fields) != 1 {
		t.Fatalf("expected one field, got %d", len(fields))
	}
	if fields[0].Conf == nil || *fields[0].Conf != 0.88 {
		t.Fatalf("conf = %v, want 0.88 (the owning box)", fields[0].Conf)
	}
	// Both boxes are still linked: one field can legitimately own several.
	if len(fields[0].BoxIds) != 2 {
		t.Fatalf("box_ids = %v, want both boxes", fields[0].BoxIds)
	}
}

// The monospace check precedes the suffix check: Licence_number has no script suffix, and
// Birth_date would otherwise fall through to the "ru" default.
func TestFieldScript(t *testing.T) {
	cases := map[string]string{
		"Last_name_ru":            "ru",
		"Last_name_en":            "en",
		"Licence_number":          "num",
		"Birth_date":              "num",
		"Issue_organisation_code": "num",
		"Driver_class":            "ru",
		"Address":                 "ru",
	}
	for name, want := range cases {
		if got := FieldScript(name); got != want {
			t.Errorf("FieldScript(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestDocTypeSplitting(t *testing.T) {
	if got := BaseDocType("INTPASSPORTADDR_ALL"); got != "INTPASSPORTADDR" {
		t.Errorf("base = %q", got)
	}
	if era := DocTypeEra("INTPASSPORTADDR_ALL"); era == nil || *era != "ALL" {
		t.Errorf("era = %v, want ALL (the UI renders it as a chip like any other)", era)
	}
	// No underscore: survivable, unlike the pipeline's own split.
	if got := BaseDocType("NONE"); got != "NONE" {
		t.Errorf("base = %q", got)
	}
	if era := DocTypeEra("NONE"); era != nil {
		t.Errorf("era = %v, want null", *era)
	}
}

// Requirement 4 of the spec: emit null, never omit. A key that vanishes fails the
// checker's key-set comparison for a reason nobody intended.
func TestEveryContractKeyIsPresentEvenWhenEmpty(t *testing.T) {
	raw, err := json.Marshal(Build(Input{DocType: "NONE", CanvasMissing: true}, false))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"doc_type", "doc_type_base", "doc_type_era", "recognised",
		"device", "canvas", "coord_space", "coord_space_note", "boxes", "fields",
		"ocr", "quality", "timings", "address"} {
		if _, ok := got[key]; !ok {
			t.Errorf("key %q is absent; it must be present and null", key)
		}
	}
	if len(got) != 14 {
		t.Errorf("got %d top-level keys, want exactly 14: %v", len(got), keysOf(got))
	}
	// debug is the one key that may be absent, and only without --include-debug.
	if _, ok := got["debug"]; ok {
		t.Error("debug must be omitted unless include_debug is set")
	}
}

// An unrecognised document is a normal outcome the SPA renders as a state, not an error.
func TestNoneIsNotRecognisedButStillWellFormed(t *testing.T) {
	p := Build(Input{DocType: "NONE", CanvasMissing: true}, false)
	if p.Recognised {
		t.Error("NONE must not be reported as recognised")
	}
	if !p.Canvas.IsFallback {
		t.Error("a run with no canvas must set is_fallback")
	}
	if p.Canvas.Width != nil || p.Canvas.Height != nil {
		t.Error("a fallback canvas has null dimensions, not zeros")
	}
	// Empty collections, not null: the keys are non-nullable in the contract.
	if p.Boxes == nil || p.Fields == nil || p.Ocr == nil {
		t.Error("boxes, fields and ocr must be empty containers rather than null")
	}
}

// An empty doc_type yields a null base rather than an empty string, matching the
// reference's `base_doc_type(...) or None`.
func TestEmptyDocTypeYieldsNullBase(t *testing.T) {
	p := Build(Input{DocType: "", CanvasMissing: true}, false)
	if p.DocTypeBase != nil {
		t.Errorf("doc_type_base = %q, want null", *p.DocTypeBase)
	}
	if p.Recognised {
		t.Error("an empty doc_type is not recognised")
	}
}

func TestNumRoundsHalfToEvenAndRejectsNaN(t *testing.T) {
	// Half-to-even, matching Python's round(): 0.00005 -> 0.0000, 0.00015 -> 0.0002.
	if v := num(0.123456); v == nil || *v != 0.1235 {
		t.Errorf("num(0.123456) = %v, want 0.1235", v)
	}
	if v := num(float64(0) / 1); v == nil || *v != 0 {
		t.Errorf("num(0) = %v", v)
	}
	nan := func() float64 { var z float64; return z / z }()
	if v := num(nan); v != nil {
		t.Errorf("NaN must become null, got %v", *v)
	}
}

func keysOf(m map[string]json.RawMessage) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
