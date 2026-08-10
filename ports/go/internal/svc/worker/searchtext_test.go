package worker

import (
	"strings"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/viewmodel"
)

// The haystack must be STABLE across runs over the same document. Go randomises map
// iteration, so without the sort inside buildSearchText the stored record would differ
// between two identical runs — which makes the store non-reproducible and any diff of it
// noise.
func TestSearchTextIsStable(t *testing.T) {
	docType := "INTPASSPORT_2011"
	vm := viewmodel.Payload{
		DocType: &docType,
		Ocr: map[string]string{
			"Last_name_ru":   "ИВАНОВ",
			"First_name_ru":  "ПЕТР",
			"Birth_date":     "01.02.1990",
			"Licence_number": "69 18 812211",
		},
	}
	first := buildSearchText("Паспорт.jpg", vm)
	for i := 0; i < 50; i++ {
		if got := buildSearchText("Паспорт.jpg", vm); got != first {
			t.Fatalf("unstable haystack:\n  %q\n  %q", first, got)
		}
	}

	// Lowercased, and containing everything a user might search for.
	if first != strings.ToLower(first) {
		t.Error("the haystack is not lowercased")
	}
	for _, needle := range []string{"паспорт.jpg", "intpassport_2011", "иванов", "69 18 812211"} {
		if !strings.Contains(first, needle) {
			t.Errorf("haystack is missing %q: %q", needle, first)
		}
	}
}

func TestSearchTextHandlesAnUnrecognisedDocument(t *testing.T) {
	got := buildSearchText("broken.jpg", viewmodel.Payload{})
	if got != "broken.jpg" {
		t.Fatalf("got %q, want just the filename", got)
	}
}

// toMap must go through JSON, so the stored blob is EXACTLY what the API serves. A hand-written
// projection would be a second definition of the wire format.
func TestToMapPreservesTheWireShape(t *testing.T) {
	docType := "DL_2011"
	vm := viewmodel.Build(viewmodel.Input{DocType: docType, CanvasMissing: true}, false)
	got, err := toMap(vm)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"doc_type", "recognised", "canvas", "coord_space",
		"coord_space_note", "boxes", "fields", "ocr", "quality", "timings", "address"} {
		if _, ok := got[key]; !ok {
			t.Errorf("key %q was lost converting to a map", key)
		}
	}
	if got["doc_type"] != docType {
		t.Errorf("doc_type = %v", got["doc_type"])
	}
}
