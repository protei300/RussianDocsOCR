package pipeline

import (
	"reflect"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/imaging"
)

// Mirrors tests/test_dates_read_whole.py of the reference. The values are made up
// («14» МАРТА 2019).

// stubEngine returns a fixed text for any crop and counts the calls.
type stubEngine struct {
	text  string
	calls int
}

func (s *stubEngine) Predict(imaging.Image) (string, error) { s.calls++; return s.text, nil }
func (s *stubEngine) FixErrors(_, text string) string      { return text }

func runReread(ocr map[string]string, lines map[string]int, cyrText, latText string,
	ruFields []string) (map[string]string, []DateReread, *stubEngine, *stubEngine) {

	cyr, lat := &stubEngine{text: cyrText}, &stubEngine{text: latText}
	var fields []FieldWords
	for label, n := range lines {
		fields = append(fields, FieldWords{Label: label, DateLines: make([]imaging.Image, n)})
	}
	got := map[string]string{}
	for k, v := range ocr {
		got[k] = v
	}
	done, err := rereadDatesWhole(fields, got, OcrOptions{RuFields: ruFields}, cyr, lat)
	if err != nil {
		panic(err)
	}
	return got, done, cyr, lat
}

var issueDateRu = []string{"Issue_date"}

func TestRereadDateThatLostItsDayIsReadWhole(t *testing.T) {
	ocr, done, _, _ := runReread(map[string]string{"Issue_date": "МАРТА 2019"},
		map[string]int{"Issue_date": 1}, "14МАРТА2019", "14.03.2019", issueDateRu)
	if ocr["Issue_date"] != "14МАРТА2019" {
		t.Errorf("Issue_date = %q, want the whole reading", ocr["Issue_date"])
	}
	want := []DateReread{{Field: "Issue_date", Split: "МАРТА 2019", Whole: "14МАРТА2019"}}
	if !reflect.DeepEqual(done, want) {
		t.Errorf("done = %v, want %v", done, want)
	}
}

func TestRereadDateThatAlreadyConvertsIsNeverReRead(t *testing.T) {
	ocr, done, cyr, _ := runReread(map[string]string{"Issue_date": "28 ИЮЛЯ 2010"},
		map[string]int{"Issue_date": 1}, "14МАРТА2019", "14.03.2019", issueDateRu)
	if ocr["Issue_date"] != "28 ИЮЛЯ 2010" || cyr.calls != 0 || len(done) != 0 {
		t.Errorf("got %q, %d engine calls, %v", ocr["Issue_date"], cyr.calls, done)
	}
}

func TestRereadWholeReadingThatIsNoDateEitherChangesNothing(t *testing.T) {
	ocr, done, _, _ := runReread(map[string]string{"Issue_date": "МАРТА 2019"},
		map[string]int{"Issue_date": 1}, "МАРТА2019", "14.03.2019", issueDateRu)
	if ocr["Issue_date"] != "МАРТА 2019" || len(done) != 0 {
		t.Errorf("got %q, %v", ocr["Issue_date"], done)
	}
}

// A date routed to Latin (not in ru_fields) is re-read by the Latin engine.
func TestRereadFieldKeepsItsEngine(t *testing.T) {
	ocr, _, cyr, lat := runReread(map[string]string{"Issue_date": "01.2011"},
		map[string]int{"Issue_date": 1}, "14МАРТА2019", "14.03.2019", nil)
	if lat.calls != 1 || cyr.calls != 0 || ocr["Issue_date"] != "14.03.2019" {
		t.Errorf("lat=%d cyr=%d value=%q", lat.calls, cyr.calls, ocr["Issue_date"])
	}
}

func TestRereadNothingRememberedMeansNothingDone(t *testing.T) {
	ocr, _, cyr, _ := runReread(map[string]string{"Issue_date": "МАРТА 2019"},
		map[string]int{}, "14МАРТА2019", "14.03.2019", issueDateRu)
	if ocr["Issue_date"] != "МАРТА 2019" || cyr.calls != 0 {
		t.Errorf("got %q, %d calls", ocr["Issue_date"], cyr.calls)
	}
}

// The record date is judged by ITS converter: a reading that the general one refuses but
// the record-date one accepts is a date, and is not re-read.
func TestRereadJudgesActDateByItsOwnConverter(t *testing.T) {
	ocr, done, cyr, _ := runReread(map[string]string{"Act_date": "2015ГОДАИЮНЯИЕСЯЦА16"},
		map[string]int{"Act_date": 1}, "14МАРТА2019", "", []string{"Act_date"})
	if ocr["Act_date"] != "2015ГОДАИЮНЯИЕСЯЦА16" || cyr.calls != 0 || len(done) != 0 {
		t.Errorf("got %q, %d calls, %v", ocr["Act_date"], cyr.calls, done)
	}
}
