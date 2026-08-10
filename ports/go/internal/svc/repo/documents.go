// Package repo holds the repository functions: query, create, mutate.
//
// **These signatures ARE the migration contract.** They are copied from
// service/repositories/*, deliberately, and the whole point of the layer is that swapping
// the store implementation underneath changes nothing above it.
//
// Thin by design. Every function takes the store first and delegates the actual query to
// it, because the backends must express the same question differently — in-memory filtering
// over JSON files versus real SQL. What lives here is the genuinely shared part:
// validation, timestamp rules, and the denormalisation performed when a result is saved.
//
// Mutating functions return a NEW record; callers rebind (`rec = repo.Update(db, rec, ...)`).
// The store hands out copies, so mutating what you got back never touches storage on its
// own — which is a property to rely on, not a limitation to work around.
package repo

import (
	"fmt"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// ActiveStatuses are the statuses that mean "the list page should keep polling".
var ActiveStatuses = map[string]bool{
	model.StatusQueued: true, model.StatusProcessing: true,
}

// GetAll returns one page of matching records plus the unpaged total.
func GetAll(db store.DocumentStore, q store.Query) ([]*model.Document, int) {
	return db.QueryDocuments(q)
}

// GetByID returns the full record, including the recognition result.
func GetByID(db store.DocumentStore, id int) *model.Document {
	return db.GetRecord(id)
}

// ReserveID claims an id WITHOUT inserting a row yet.
//
// This exists so a caller can write the upload's bytes BEFORE the document becomes visible
// to the worker. Inserting first looks harmless and is a real race: the row lands in
// `queued`, the drain loop runs on its own schedule, and if it claims the document in the
// window before the file is written the job fails with "has no stored original" — a good
// upload reported as a failed document.
func ReserveID(db store.DocumentStore) int { return db.NextDocumentID() }

// Create inserts a record. Pass an id from ReserveID when the artifacts were written first.
func Create(db store.DocumentStore, rec *model.Document) *model.Document {
	return db.PutRecord(rec)
}

// Mutation is one field change, applied by Update.
//
// A function over the record rather than a map of field names: Go has no keyword arguments,
// and a `map[string]any` would move every field name and type from compile time to run
// time — in a layer whose entire job is that the field names are stable.
type Mutation func(*model.Document)

// Update applies mutations to a COPY and persists it.
//
// UpdatedAt is stamped here, once, so no caller can forget it. `Result` is carried across
// because it is stored separately: a plain field update must not look like a request to
// clear it.
func Update(db store.DocumentStore, rec *model.Document, muts ...Mutation) *model.Document {
	next := rec.Clone()
	for _, m := range muts {
		m(next)
	}
	next.UpdatedAt = model.At(model.UtcNow())
	next.Result = rec.Result
	return db.PutRecord(next)
}

// UpdateStatus moves a document between statuses and stamps the matching timestamp.
//
// The status is VALIDATED rather than trusted: it reaches the store, the wire and the SPA's
// badge classes, and an invented value would render as an unstyled row somebody then
// reports as a UI bug.
func UpdateStatus(db store.DocumentStore, rec *model.Document, status string,
	errText, errCode *string) (*model.Document, error) {

	if !model.ValidStatuses[status] {
		return nil, fmt.Errorf("repo: invalid status %q", status)
	}
	muts := []Mutation{func(d *model.Document) {
		d.Status = status
		d.Error = errText
		d.ErrorCode = errCode
	}}
	switch status {
	case model.StatusProcessing:
		muts = append(muts, func(d *model.Document) { d.StartedAt = model.At(model.UtcNow()) })
	case model.StatusDone, model.StatusFailed:
		muts = append(muts, func(d *model.Document) { d.FinishedAt = model.At(model.UtcNow()) })
	}
	return Update(db, rec, muts...), nil
}

// SaveResult stores the view model and denormalises the columns the list page needs.
//
// The denormalisation IS the point: without it, filtering or sorting the log means opening
// every result blob on every keystroke.
func SaveResult(db store.DocumentStore, rec *model.Document, payload map[string]any,
	searchText string, processingMs int) (*model.Document, error) {

	if err := db.SaveResultPayload(rec.ID, payload); err != nil {
		return nil, err
	}

	quality, _ := payload["quality"].(map[string]any)
	canvas, _ := payload["canvas"].(map[string]any)

	// DocConf is lifted OUT of the quality map into its own column, because the list page
	// sorts by it. The remaining keys stay together: they are verdict strings with no
	// single vocabulary ('good'/'bad' and 'REAL'/'FAKE'), so a column each would invite a
	// client to assume otherwise.
	var docConf *float64
	trimmedQuality := map[string]any{}
	for k, v := range quality {
		if k == "DocConf" {
			if f, ok := asFloat(v); ok {
				docConf = &f
			}
			continue
		}
		trimmedQuality[k] = v
	}

	fields, _ := payload["fields"].([]any)
	recognised, _ := payload["recognised"].(bool)
	docType := asStringPtr(payload["doc_type"])
	device := asStringPtr(payload["device"])

	var canvasW, canvasH *int
	if canvas != nil {
		if w, ok := asInt(canvas["width"]); ok {
			canvasW = &w
		}
		if h, ok := asInt(canvas["height"]); ok {
			canvasH = &h
		}
	}
	ms := processingMs

	return Update(db, rec, func(d *model.Document) {
		d.Status = model.StatusDone
		d.Error = nil
		d.ErrorCode = nil
		d.DocType = docType
		d.DocConf = docConf
		d.Quality = trimmedQuality
		d.Recognised = recognised
		d.FieldCount = len(fields)
		d.Device = device
		d.ProcessingMs = &ms
		d.CanvasW = canvasW
		d.CanvasH = canvasH
		d.HasCanvas = canvasW != nil
		d.SearchText = searchText
		d.FinishedAt = model.At(model.UtcNow())
	}), nil
}

// Requeue resets a document for another attempt, clearing the previous outcome.
//
// RetryCount goes back to zero because this is an OPERATOR action, not an automatic retry:
// a human asking for a reprocess should get the full retry budget, not whatever was left.
func Requeue(db store.DocumentStore, rec *model.Document) *model.Document {
	return Update(db, rec, func(d *model.Document) {
		d.Status = model.StatusQueued
		d.RetryCount = 0
		d.Error = nil
		d.ErrorCode = nil
		d.StartedAt = model.Never()
		d.FinishedAt = model.Never()
	})
}

func Delete(db store.DocumentStore, rec *model.Document) { db.DropRecord(rec.ID) }

func NextQueued(db store.DocumentStore) (int, bool) { return db.NextQueuedID() }

func QueuePosition(db store.DocumentStore, id int) (int, bool) { return db.QueuePosition(id) }

// ResetStaleProcessing recovers jobs interrupted mid-flight by a restart.
//
// Without it a document caught in `processing` when the process died sits there forever:
// the drain loop only ever claims `queued` rows. Called once at startup.
func ResetStaleProcessing(db store.DocumentStore) int {
	count := 0
	for _, rec := range db.AllRecords() {
		if rec.Status != model.StatusProcessing {
			continue
		}
		Update(db, rec, func(d *model.Document) {
			d.Status = model.StatusQueued
			d.StartedAt = model.Never()
		})
		count++
	}
	return count
}

func CountByStatus(db store.DocumentStore) map[string]int { return db.CountByStatus() }

func Stats(db store.DocumentStore) store.Stats { return db.AggregateStats() }

// -- JSON coercion ----------------------------------------------------------
// The view model arrives as map[string]any because it round-trips through JSON, where
// every number is a float64. These helpers are the one place that knows it, so no caller
// has to guess whether a field is int or float.

func asFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	default:
		return 0, false
	}
}

func asInt(v any) (int, bool) {
	if f, ok := asFloat(v); ok {
		return int(f), true
	}
	return 0, false
}

func asStringPtr(v any) *string {
	s, ok := v.(string)
	if !ok || s == "" {
		return nil
	}
	return &s
}
