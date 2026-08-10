// Package model holds the record shapes for the store.
//
// **SQL SWAP POINT.** These structs become the ORM entities when a real database arrives.
// Field names are chosen to be valid SQL column names, and the API layer depends on THESE
// NAMES ONLY — so the swap touches this package, the store and the repository bodies, and
// nothing else.
//
// Two denormalisations are deliberate and would be kept in SQL:
//
//   - DocType / DocConf / ProcessingMs / Canvas* are columns, so the list page can filter
//     and sort without parsing the stored result blob;
//   - SearchText is a precomputed lowercase haystack (filename + doc type + every
//     recognised value). Without it, "search by recognised surname" means parsing every
//     result blob on every keystroke. In SQL this becomes an indexed computed column.
//
// Port of service/core/models.py.
package model

import (
	"encoding/json"
	"time"
)

// Status values a document can hold. String constants rather than an iota enum: they go on
// the wire, the SPA's badge classes map to them one-to-one, and three languages serialise
// integer enums three different ways (CONVENTIONS §1).
const (
	StatusQueued     = "queued"
	StatusProcessing = "processing"
	StatusDone       = "done"
	StatusFailed     = "failed"
)

// ValidStatuses is the closed set, for validating a filter parameter.
var ValidStatuses = map[string]bool{
	StatusQueued: true, StatusProcessing: true, StatusDone: true, StatusFailed: true,
}

// UtcNow is timezone-aware UTC, serialised with an explicit Z on the wire.
//
// Naive timestamps are how a frontend ends up guessing the zone. Go's time.Time carries a
// location, so the discipline here is only ever to CREATE times in UTC.
func UtcNow() time.Time { return time.Now().UTC() }

// Time is a timestamp that marshals as ISO-8601 UTC with an explicit Z, or as null.
//
// A named type rather than *time.Time with a custom marshaller at each site: the wire
// format is a cross-language requirement (spec/viewmodel.md), and Go's default
// time.Time JSON encoding is RFC 3339 with a numeric offset — "+00:00", not "Z" — which
// would differ from Python's output for the same instant.
type Time struct {
	// Set distinguishes "no timestamp" from the zero instant. A document that has not
	// started has started_at == null, which is not the same as 1970.
	Set  bool
	Time time.Time
}

func At(t time.Time) Time { return Time{Set: true, Time: t.UTC()} }
func Never() Time         { return Time{} }

func (t Time) MarshalJSON() ([]byte, error) {
	if !t.Set {
		return []byte("null"), nil
	}
	return json.Marshal(t.Time.UTC().Format("2006-01-02T15:04:05.999999999Z"))
}

func (t *Time) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		*t = Time{}
		return nil
	}
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	// Accept both spellings on the way IN, because a record written by the Python
	// implementation may carry either and the two services share a data directory.
	parsed, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		return err
	}
	*t = At(parsed)
	return nil
}

// Document is one uploaded document and everything known about it.
//
// The JSON tags are the on-disk record format AND the SQL column names. They are written
// by hand for the reason every wire name in this port is: three languages have three
// default naming policies, and a record written by one implementation must be readable by
// the others.
type Document struct {
	ID          int    `json:"id"`
	Filename    string `json:"filename"` // sanitised, DISPLAY ONLY — never a path
	ContentType string `json:"content_type"`
	SizeBytes   int64  `json:"size_bytes"`
	Status      string `json:"status"`

	DocType    *string  `json:"doc_type"`
	DocConf    *float64 `json:"doc_conf"`
	Recognised bool     `json:"recognised"`
	FieldCount int      `json:"field_count"`
	// Quality holds the denormalised verdicts so the list page can show them without
	// loading each result blob. Values are whatever the library reports — currently
	// 'good'/'bad' for glare and blur but 'REAL'/'FAKE' for the spoofing checks, so
	// clients must NOT assume a single vocabulary.
	Quality map[string]any `json:"quality"`

	Device       *string `json:"device"`
	ProcessingMs *int    `json:"processing_ms"`
	Error        *string `json:"error"`
	ErrorCode    *string `json:"error_code"`
	RetryCount   int     `json:"retry_count"`

	OriginalExt string `json:"original_ext"`
	OriginalW   *int   `json:"original_w"`
	OriginalH   *int   `json:"original_h"`
	CanvasW     *int   `json:"canvas_w"`
	CanvasH     *int   `json:"canvas_h"`
	HasCanvas   bool   `json:"has_canvas"`

	SearchText string `json:"search_text"`

	CreatedAt  Time `json:"created_at"`
	StartedAt  Time `json:"started_at"`
	FinishedAt Time `json:"finished_at"`
	UpdatedAt  Time `json:"updated_at"`

	// Result is the full recognition view model. Kept OUT of the in-memory index — it can
	// be 100 KB of boxes per document — and loaded lazily by the repository's GetByID.
	// `json:"-"` because it lives in its own file, not in the record.
	Result map[string]any `json:"-"`
}

// NewDocument builds a queued record with the defaults the store expects.
func NewDocument(id int, filename, contentType string, size int64, ext string) *Document {
	now := UtcNow()
	return &Document{
		ID:          id,
		Filename:    filename,
		ContentType: contentType,
		SizeBytes:   size,
		Status:      StatusQueued,
		Quality:     map[string]any{},
		OriginalExt: ext,
		CreatedAt:   At(now),
		UpdatedAt:   At(now),
	}
}

// Clone returns a deep-enough copy.
//
// The store hands out copies rather than pointers into its index, so a caller mutating a
// record cannot corrupt the index without going through an Update. The maps are copied
// because a shallow struct copy would still share them — which is exactly the kind of
// aliasing that makes a concurrency bug reproduce only under load.
func (d *Document) Clone() *Document {
	if d == nil {
		return nil
	}
	out := *d
	out.Quality = make(map[string]any, len(d.Quality))
	for k, v := range d.Quality {
		out.Quality[k] = v
	}
	// Result is intentionally NOT deep-copied: it is loaded lazily, treated as immutable
	// once read, and copying 100 KB of boxes on every list row would be pure waste.
	return &out
}

// ApiKey is a credential for machine callers.
//
// The plaintext is shown ONCE, at creation, and only the hash is kept — the same reasoning
// as any password store: a leaked data directory must not hand over working credentials.
type ApiKey struct {
	ID      int    `json:"id"`
	Label   string `json:"label"`
	Prefix  string `json:"prefix"`   // first few chars, to identify a key in the UI
	KeyHash string `json:"key_hash"` // sha256 of the full key
	// IsDefault marks the key that came from the environment. It cannot be deleted:
	// without it a restart would leave the API with no way in at all.
	IsDefault  bool `json:"is_default"`
	CreatedAt  Time `json:"created_at"`
	LastUsedAt Time `json:"last_used_at"`
}

// Public is what the UI may see. Never the hash.
//
// A separate shape rather than `json:"-"` on KeyHash, because the same struct is persisted
// WITH the hash — one type with two audiences needs two explicit projections, not a tag
// that has to be right in both directions.
func (k *ApiKey) Public() map[string]any {
	return map[string]any{
		"id":           k.ID,
		"label":        k.Label,
		"prefix":       k.Prefix,
		"masked":       k.Prefix + "••••••••",
		"is_default":   k.IsDefault,
		"created_at":   k.CreatedAt,
		"last_used_at": k.LastUsedAt,
	}
}
