package store

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
)

// This is the REPOSITORY CONTRACT SUITE. It is written against the DocumentStore interface,
// not against FileStore, so the same tests run unchanged against a SQL backend when one
// arrives — which is what makes the migration a configuration change rather than a rewrite.
// The Python project states the same intent in test_repository_contract.py.

func openTemp(t *testing.T) DocumentStore {
	t.Helper()
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func put(t *testing.T, s DocumentStore, id int, status string, mutate ...func(*model.Document)) *model.Document {
	t.Helper()
	rec := model.NewDocument(id, "f.jpg", "image/jpeg", 100, ".jpg")
	rec.Status = status
	for _, m := range mutate {
		m(rec)
	}
	return s.PutRecord(rec)
}

func TestPutAndGetRoundTrip(t *testing.T) {
	s := openTemp(t)
	put(t, s, 1, model.StatusQueued)

	got := s.GetRecord(1)
	if got == nil {
		t.Fatal("record not found after put")
	}
	if got.ID != 1 || got.Status != model.StatusQueued || got.Filename != "f.jpg" {
		t.Fatalf("round trip lost fields: %+v", got)
	}
	if s.GetRecord(99) != nil {
		t.Fatal("a missing record must be nil, not a zero value")
	}
}

// Reads return COPIES. A shared instance would let one request's edit leak into another's
// view, and the whole update-and-rebind idiom depends on this.
func TestGetReturnsACopy(t *testing.T) {
	s := openTemp(t)
	put(t, s, 1, model.StatusQueued)

	first := s.GetRecord(1)
	first.Status = model.StatusDone
	first.Quality["tampered"] = true

	second := s.GetRecord(1)
	if second.Status != model.StatusQueued {
		t.Error("mutating a returned record changed the store")
	}
	if _, leaked := second.Quality["tampered"]; leaked {
		t.Error("the quality map is shared between copies — a shallow copy is not enough")
	}
}

// A record survives a restart, because the index is rebuilt from disk. This is what keeps
// `uvicorn --reload` (and its Go equivalent) from losing everything mid-session.
func TestRecordsSurviveReopen(t *testing.T) {
	dir := t.TempDir()
	first, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	put(t, first, 7, model.StatusDone, func(d *model.Document) {
		dt := "SNILS_1996"
		d.DocType = &dt
	})

	second, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	got := second.GetRecord(7)
	if got == nil || got.DocType == nil || *got.DocType != "SNILS_1996" {
		t.Fatalf("record did not survive a reopen: %+v", got)
	}
	// The id counter must clear the highest recovered id, or the next upload would collide.
	if next := second.NextDocumentID(); next <= 7 {
		t.Errorf("next id %d would collide with the recovered record", next)
	}
}

// A corrupt record is skipped, not fatal. A service that refuses to start because one of two
// hundred files is truncated is worse than one that starts with 199.
func TestCorruptRecordIsSkipped(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	put(t, s, 1, model.StatusDone)

	bad := filepath.Join(dir, "documents", "2")
	if err := os.MkdirAll(bad, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bad, "record.json"), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatalf("a corrupt record must not stop the store from opening: %v", err)
	}
	if reopened.GetRecord(1) == nil {
		t.Error("the good record was lost along with the bad one")
	}
}

// FIFO by creation, with the id as a tie-break. Without the tie-break two uploads inside one
// clock tick have an unspecified order — exactly the case somebody tests by uploading twice
// quickly.
func TestQueueIsFifoWithIdTieBreak(t *testing.T) {
	s := openTemp(t)
	now := time.Now().UTC()
	for _, id := range []int{3, 1, 2} {
		put(t, s, id, model.StatusQueued, func(d *model.Document) {
			d.CreatedAt = model.At(now) // identical timestamps on purpose
		})
	}
	got, ok := s.NextQueuedID()
	if !ok || got != 1 {
		t.Fatalf("next queued = %d (ok=%v), want 1", got, ok)
	}
	if pos, ok := s.QueuePosition(2); !ok || pos != 1 {
		t.Fatalf("queue position of 2 = %d (ok=%v), want 1", pos, ok)
	}
	// A document that is not queued has no position, which is distinct from position 0.
	put(t, s, 4, model.StatusDone)
	if _, ok := s.QueuePosition(4); ok {
		t.Error("a done document must have no queue position")
	}
}

// Nulls sort LAST in both directions. A queued document has no doc_conf and must not lead an
// ascending sort, or the list page opens on rows with nothing in the column being sorted.
func TestSortPutsNullsLastInBothDirections(t *testing.T) {
	s := openTemp(t)
	conf := 0.9
	put(t, s, 1, model.StatusDone, func(d *model.Document) { d.DocConf = &conf })
	put(t, s, 2, model.StatusQueued) // DocConf nil

	for _, dir := range []string{"asc", "desc"} {
		rows, total := s.QueryDocuments(Query{SortBy: "doc_conf", SortDir: dir, Page: 1, PageSize: 10})
		if total != 2 {
			t.Fatalf("%s: total %d", dir, total)
		}
		if rows[0].ID != 1 {
			t.Errorf("%s: the null doc_conf sorted first (order %d, %d)", dir, rows[0].ID, rows[1].ID)
		}
	}
}

// An unknown sort column falls back to created_at rather than erroring or sorting by nothing.
// A whitelist, because in a SQL backend that difference is an injection vector.
func TestUnknownSortColumnFallsBack(t *testing.T) {
	s := openTemp(t)
	put(t, s, 1, model.StatusDone)
	put(t, s, 2, model.StatusDone)
	if _, total := s.QueryDocuments(Query{SortBy: "; DROP TABLE", Page: 1, PageSize: 10}); total != 2 {
		t.Fatal("an unknown sort column must not change the result set")
	}
}

func TestFilters(t *testing.T) {
	s := openTemp(t)
	dl := "DL_2011"
	put(t, s, 1, model.StatusDone, func(d *model.Document) {
		d.DocType = &dl
		d.Recognised = true
		d.SearchText = "ivanov dl_2011"
	})
	put(t, s, 2, model.StatusFailed, func(d *model.Document) { d.SearchText = "broken.jpg" })

	cases := []struct {
		name string
		q    Query
		want int
	}{
		{"status", Query{Status: model.StatusDone}, 1},
		{"doc_type prefix", Query{DocType: "DL"}, 1},
		{"unrecognised", Query{DocType: "__none__"}, 1},
		{"search hit", Query{Search: "IVANOV"}, 1},
		{"search miss", Query{Search: "petrov"}, 0},
		// A half-typed date disables the filter rather than erroring: the list page sends
		// it on every keystroke.
		{"partial date is ignored", Query{DateFrom: "2026-0"}, 2},
	}
	for _, c := range cases {
		c.q.Page, c.q.PageSize = 1, 10
		if _, total := s.QueryDocuments(c.q); total != c.want {
			t.Errorf("%s: total %d, want %d", c.name, total, c.want)
		}
	}
}

func TestPagination(t *testing.T) {
	s := openTemp(t)
	for id := 1; id <= 5; id++ {
		put(t, s, id, model.StatusDone)
	}
	rows, total := s.QueryDocuments(Query{Page: 2, PageSize: 2})
	if total != 5 || len(rows) != 2 {
		t.Fatalf("page 2 of 5 with size 2: total %d, rows %d", total, len(rows))
	}
	// Past the end is EMPTY, not an error and not a panic.
	rows, _ = s.QueryDocuments(Query{Page: 99, PageSize: 2})
	if len(rows) != 0 {
		t.Fatalf("a page past the end must be empty, got %d rows", len(rows))
	}
}

func TestAggregateStats(t *testing.T) {
	s := openTemp(t)
	ms := 500
	put(t, s, 1, model.StatusDone, func(d *model.Document) {
		d.Recognised = true
		d.ProcessingMs = &ms
	})
	ms2 := 700
	put(t, s, 2, model.StatusDone, func(d *model.Document) { d.ProcessingMs = &ms2 })
	put(t, s, 3, model.StatusQueued)

	got := s.AggregateStats()
	if got.Total != 3 || got.Done != 2 || got.Queued != 1 || got.Recognised != 1 {
		t.Fatalf("counts wrong: %+v", got)
	}
	if got.AvgProcessingMs == nil || *got.AvgProcessingMs != 600 {
		t.Fatalf("average = %v, want 600", got.AvgProcessingMs)
	}
	// With nothing timed the average is NULL, not zero: "no data" and "instant" are
	// different claims.
	empty := openTemp(t)
	if empty.AggregateStats().AvgProcessingMs != nil {
		t.Error("an empty store must report a null average, not 0")
	}
}

func TestResultPayloadIsSeparateFromTheRecord(t *testing.T) {
	s := openTemp(t)
	put(t, s, 1, model.StatusDone)
	if err := s.SaveResultPayload(1, map[string]any{"doc_type": "DL_2011"}); err != nil {
		t.Fatal(err)
	}
	// GetRecord attaches it lazily...
	if got := s.GetRecord(1); got.Result == nil || got.Result["doc_type"] != "DL_2011" {
		t.Fatalf("result not attached by GetRecord: %+v", got.Result)
	}
	// ...but a list query must NOT carry it, or every row pulls 100 KB of boxes.
	rows, _ := s.QueryDocuments(Query{Page: 1, PageSize: 10})
	if rows[0].Result != nil {
		t.Error("a list row carried the result blob; the index must not hold it")
	}
}

func TestSettingsRoundTripAndMerge(t *testing.T) {
	s := openTemp(t)
	if _, err := s.SetSettings(map[string]string{"a": "1"}); err != nil {
		t.Fatal(err)
	}
	// A second write MERGES rather than replacing: the settings page sends only what
	// changed.
	if _, err := s.SetSettings(map[string]string{"b": "2"}); err != nil {
		t.Fatal(err)
	}
	got := s.AllSettings()
	if got["a"] != "1" || got["b"] != "2" {
		t.Fatalf("settings did not merge: %v", got)
	}
}

func TestDropRemovesRecordAndArtifacts(t *testing.T) {
	s := openTemp(t)
	put(t, s, 1, model.StatusDone)
	if err := s.SaveResultPayload(1, map[string]any{"x": 1}); err != nil {
		t.Fatal(err)
	}
	dir := s.DocDir(1)
	s.DropRecord(1)

	if s.GetRecord(1) != nil {
		t.Error("record still present after drop")
	}
	if _, err := os.Stat(dir); err == nil {
		t.Error("artifact directory survived the drop")
	}
}
