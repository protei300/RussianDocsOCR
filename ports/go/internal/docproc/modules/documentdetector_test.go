package modules

import (
	"reflect"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/postprocess"
)

// Mirrors tests/test_document_detector.py of the reference (the grouping part; pairing the
// two sides of an STS is process_frame's business, see pipeline/frame.go and sts_test.go).

func box(x1, y1, x2, y2, conf float64, cls int, label string) postprocess.Box {
	return postprocess.Box{X1: x1, Y1: y1, X2: x2, Y2: y2, Conf: conf, Cls: cls, Label: label}
}

func TestPagesBelongToTheirDocumentAndDocumentsGoLargestFirst(t *testing.T) {
	docs := GroupDocuments([]postprocess.Box{
		box(500, 50, 700, 200, 0.9, 0, "document"), // a small card
		box(0, 0, 400, 600, 0.95, 0, "document"),   // a passport spread
		box(10, 310, 390, 590, 0.9, 1, "page"),     // its lower page
		box(10, 10, 390, 290, 0.9, 1, "page"),      // its upper page
		box(900, 900, 950, 950, 0.5, 1, "page"),    // a page in no document
	})
	if len(docs) != 2 {
		t.Fatalf("got %d documents, want 2", len(docs))
	}
	if docs[0].X2 != 400 || docs[0].Y2 != 600 || docs[1].X1 != 500 {
		t.Errorf("documents not largest first: %+v", docs)
	}
	var pages [][4]float64
	for _, p := range docs[0].Pages {
		pages = append(pages, [4]float64{p.X1, p.Y1, p.X2, p.Y2})
	}
	want := [][4]float64{{10, 10, 390, 290}, {10, 310, 390, 590}}
	if !reflect.DeepEqual(pages, want) {
		t.Errorf("pages = %v, want %v (top to bottom)", pages, want)
	}
	if len(docs[1].Pages) != 0 {
		t.Errorf("the card has pages: %v", docs[1].Pages)
	}
}

func TestNothingFoundGroupsToNothing(t *testing.T) {
	if docs := GroupDocuments(nil); len(docs) != 0 {
		t.Errorf("got %v", docs)
	}
}
