// Package recognitions is the recognition journal: one line per POST /recognize call, and nothing
// about the document.
//
// A line says THAT something was recognised, how it ended and how long it took: id, time,
// outcome, HTTP status, milliseconds. No filename, no document type, no size, no field, no caller,
// no client address — it is the one record of an in-memory recognition and must be safe to show
// to anyone who can see the service at all. The UI lists these lines but cannot open them: there
// is nothing behind them to open.
//
// Kept IN MEMORY ONLY, capped, lost at restart. Writing it to the store would be harmless
// content-wise, but the point of the mode it serves is that a recognition leaves nothing on disk.
//
// A value owned by the Server rather than package state (CONVENTIONS: no package-level mutable
// state outside the runtime pool). Port of service/repositories/recognitions.py.
package recognitions

import (
	"sync"
	"time"
)

// Outcomes, as the UI shows them. recognised = a document type was found; not_recognised = the
// image was read but no supported document was in it; rejected = refused before recognition (bad
// or oversized image); busy = no pipeline became free in time; error = anything else.
const (
	Recognised    = "recognised"
	NotRecognised = "not_recognised"
	Rejected      = "rejected"
	Busy          = "busy"
	Error         = "error"
)

// Outcomes is the closed set, in the order the counts list them.
var Outcomes = []string{Recognised, NotRecognised, Rejected, Busy, Error}

// Event is one journal line. The JSON is EXACTLY these six keys (ports/MEMORY.md §4).
type Event struct {
	ID           int    `json:"id"`
	At           string `json:"at"`
	Outcome      string `json:"outcome"`
	StatusCode   int    `json:"status_code"`
	ProcessingMs int    `json:"processing_ms"`
	// Stored is always false: the UI keys "no drill-down" off it.
	Stored bool `json:"stored"`
}

// Journal is a bounded ring, newest kept, oldest dropped.
type Journal struct {
	mu     sync.Mutex
	cap    int
	events []Event // oldest first
	nextID int
}

// New returns an empty journal holding at most capacity lines (at least one).
func New(capacity int) *Journal {
	if capacity < 1 {
		capacity = 1
	}
	return &Journal{cap: capacity, nextID: 1}
}

// Record appends one line and returns it.
func (j *Journal) Record(outcome string, statusCode, processingMs int) Event {
	if processingMs < 0 {
		processingMs = 0
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	ev := Event{
		ID: j.nextID,
		// Python's datetime.isoformat() with the offset written as Z: microseconds.
		At:           time.Now().UTC().Format("2006-01-02T15:04:05.000000Z"),
		Outcome:      outcome,
		StatusCode:   statusCode,
		ProcessingMs: processingMs,
	}
	j.nextID++
	j.events = append(j.events, ev)
	if over := len(j.events) - j.cap; over > 0 {
		// Copied down rather than resliced, so the dropped lines do not stay reachable from
		// the backing array.
		j.events = append(j.events[:0], j.events[over:]...)
	}
	return ev
}

// Recent returns one page, newest first, and the total held.
func (j *Journal) Recent(limit, offset int) ([]Event, int) {
	j.mu.Lock()
	defer j.mu.Unlock()
	total := len(j.events)
	page := []Event{}
	for i := total - 1 - offset; i >= 0 && len(page) < limit; i-- {
		page = append(page, j.events[i])
	}
	return page, total
}

// Counts is how many lines ended each way, plus "total".
func (j *Journal) Counts() map[string]int {
	j.mu.Lock()
	defer j.mu.Unlock()
	out := make(map[string]int, len(Outcomes)+1)
	for _, o := range Outcomes {
		out[o] = 0
	}
	for _, e := range j.events {
		out[e.Outcome]++
	}
	out["total"] = len(j.events)
	return out
}
