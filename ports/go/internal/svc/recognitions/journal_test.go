package recognitions

import (
	"encoding/json"
	"testing"
)

func TestJournalIsBoundedNewestFirstAndCounts(t *testing.T) {
	j := New(3)
	for i, o := range []string{Recognised, Rejected, Busy, NotRecognised, Error} {
		j.Record(o, 200+i, i)
	}
	page, total := j.Recent(10, 0)
	if total != 3 || len(page) != 3 {
		t.Fatalf("total %d, page %d; want the cap, 3", total, len(page))
	}
	if page[0].ID != 5 || page[1].ID != 4 || page[2].ID != 3 {
		t.Errorf("not newest first: %+v", page)
	}
	if page, _ := j.Recent(1, 1); len(page) != 1 || page[0].ID != 4 {
		t.Errorf("offset: %+v", page)
	}
	counts := j.Counts()
	if counts["total"] != 3 || counts[Busy] != 1 || counts[Recognised] != 0 || len(counts) != 6 {
		t.Errorf("counts %v", counts)
	}

	// Exactly the six keys of ports/MEMORY.md §4, stored=false.
	raw, _ := json.Marshal(page[0])
	var line map[string]any
	_ = json.Unmarshal(raw, &line)
	if len(line) != 6 || line["stored"] != false {
		t.Errorf("journal line %s", raw)
	}
	if at, _ := line["at"].(string); len(at) != len("2026-10-19T05:02:53.239750Z") || at[len(at)-1] != 'Z' {
		t.Errorf("at %q", line["at"])
	}
}
