// Package store is the storage contract and its filesystem implementation.
//
// The service has no database on purpose — the data is a scratch pad that dies with the
// container. But "no database" must not mean "no abstraction", or moving to SQL later
// becomes a rewrite. So this package presents the surface a database session would, and the
// repositories on top of it take the store as their first argument, exactly as the
// reference's repository functions take `db`.
//
// **SQL SWAP POINT.** Implementing DocumentStore over a real database, and constructing
// that instead of FileStore, is the whole migration as far as callers are concerned.
// Router and worker code does not change.
//
// On-disk layout:
//
//	$DATA_DIR/
//	  documents/42/
//	    record.json     the "row"
//	    original.jpg    exactly the bytes uploaded
//	    canvas.png      the deskewed/rectified canvas
//	    result.json     the full recognition view model
//	  api_keys.json
//	  settings.json
//
// Four design notes worth reading before changing anything here:
//
//   - **The index lives in memory; disk is scanned once at startup.** The service is
//     pinned to ONE process — the pipeline singleton and this index both are — so a
//     shared in-memory index is legitimate rather than a shortcut.
//   - **Writes are atomic** (temp file plus rename, atomic on NTFS and ext4). A
//     half-written record.json would survive a crash and poison the next boot.
//   - **Reads return COPIES.** A live shared instance would let one request's edit leak
//     into another's view, so Update returns a NEW record and callers must rebind. That
//     is already how the reference's routers are written (`m = repo.update_status(...)`),
//     so the idiom ports without changing call sites.
//   - **`result` is not held in the index** — it can be 100 KB of boxes per document.
//     GetByID loads it lazily; list queries never touch it.
//
// Port of service/core/store.py and service/core/database.py.
package store

import (
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
)

// SortColumns is the whitelist of sortable columns, shared by every backend so they cannot
// drift apart.
//
// A whitelist rather than dynamic field lookup: in a SQL backend that difference is an
// injection vector, and here it is what stops a typo in a query string from silently
// sorting by nothing.
var SortColumns = map[string]bool{
	"created_at": true, "filename": true, "status": true, "doc_type": true,
	"doc_conf": true, "processing_ms": true, "size_bytes": true,
}

// Query is the filter/sort/page request for a document listing.
//
// A struct rather than a long parameter list because it crosses the store boundary and
// grows: a positional call is where "swap date_from and date_to" hides.
type Query struct {
	Status   string
	DocType  string
	Search   string
	DateFrom string
	DateTo   string
	Page     int
	PageSize int
	SortBy   string
	SortDir  string
}

// Stats is the aggregate summary the status page shows.
type Stats struct {
	Queued          int  `json:"queued"`
	Processing      int  `json:"processing"`
	Done            int  `json:"done"`
	Failed          int  `json:"failed"`
	Total           int  `json:"total"`
	Recognised      int  `json:"recognised"`
	AvgProcessingMs *int `json:"avg_processing_ms"`
}

// DocumentStore is everything the service needs from a storage backend.
//
// Query methods live HERE rather than in the repository functions, deliberately: filtering
// a list in memory is correct for a few hundred JSON files and wrong for a table, so each
// backend has to express "the newest twenty matching rows" in its own terms. Putting the
// queries behind the interface is what lets it.
type DocumentStore interface {
	// Backend is "files" or "sql" — surfaced on the status page, because "why did my
	// data vanish" is answered by this one word.
	Backend() string
	// IsEphemeral reports whether the contents survive a restart.
	IsEphemeral() bool

	NextDocumentID() int
	GetRecord(id int) *model.Document
	PutRecord(rec *model.Document) *model.Document
	DropRecord(id int)
	QueryDocuments(q Query) ([]*model.Document, int)
	AllRecords() []*model.Document
	NextQueuedID() (int, bool)
	QueuePosition(id int) (int, bool)
	CountByStatus() map[string]int
	AggregateStats() Stats

	SaveResultPayload(id int, payload map[string]any) error
	LoadResultPayload(id int) map[string]any

	AllApiKeys() []*model.ApiKey
	NextApiKeyID() int
	PutApiKey(key *model.ApiKey) (*model.ApiKey, error)
	DropApiKey(id int) (bool, error)

	AllSettings() map[string]string
	SetSettings(values map[string]string) (map[string]string, error)

	// DocDir is a plain directory in every backend: binary artifacts stay on the
	// filesystem regardless of where the metadata lives.
	DocDir(id int) string
	DiskUsageBytes() int64
}
