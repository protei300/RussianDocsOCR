package store

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
)

// AtomicWriteJSON writes JSON so a crash can never leave a partial file behind.
//
// Temp file plus rename, which is atomic on NTFS and ext4. Not a nicety: a truncated
// record.json survives the crash and poisons the next boot, and the failure looks like
// data corruption rather than an interrupted write.
func AtomicWriteJSON(path string, payload any) error {
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("store: encode %s: %w", path, err)
	}
	return AtomicWriteBytes(path, data)
}

// AtomicWriteBytes is the same guarantee for opaque bytes.
func AtomicWriteBytes(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("store: write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("store: rename %s: %w", tmp, err)
	}
	return nil
}

// FileStore is the filesystem backend.
//
// Concurrency: ONE mutex guards the index and all mutations, because both the worker
// goroutine and the HTTP handlers write here. Long I/O — writing a 2 MB PNG — happens
// OUTSIDE the lock; only the rename and the index update are inside it.
//
// Go has no RLock-style reentrant mutex, which the Python version uses because some of its
// operations nest. The port avoids the need instead: every exported method takes the lock
// at most once and calls only `...Locked` helpers beneath it. That is a real constraint on
// edits to this file — a helper that takes the lock again deadlocks rather than nesting.
type FileStore struct {
	root    string
	docsDir string

	mu        sync.Mutex
	records   map[int]*model.Document
	apiKeys   map[int]*model.ApiKey
	settings  map[string]string
	nextDocID int
	nextKeyID int
}

// Open creates the store, scanning the directory once.
func Open(root string) (*FileStore, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("store: resolve %s: %w", root, err)
	}
	s := &FileStore{
		root:      abs,
		docsDir:   filepath.Join(abs, "documents"),
		records:   map[int]*model.Document{},
		apiKeys:   map[int]*model.ApiKey{},
		settings:  map[string]string{},
		nextDocID: 1,
		nextKeyID: 1,
	}
	if err := os.MkdirAll(s.docsDir, 0o755); err != nil {
		return nil, fmt.Errorf("store: create %s: %w", s.docsDir, err)
	}
	s.scan()
	return s, nil
}

// Wipe empties the data directory. Called before Open when configured.
//
// Deliberate, and the reason the "ephemeral" promise is true: `docker restart` KEEPS the
// writable layer, so the absence of a volume is not enough on its own.
//
// **The CONTENTS go, the directory stays** — and that is not a detail. Removing the directory
// itself needs write permission on its PARENT, which a non-root container does not have for
// /app, and a directory that is a MOUNT POINT can never be unlinked at all, by anyone. So
// `os.RemoveAll(dir)` fails in both of the configurations this service is actually deployed in;
// it was found the first time the image ran, as `unlinkat /app/data: permission denied`, with
// the store then unusable. Emptying achieves the same observable result — nothing survives a
// restart — and works as non-root, and works under `-v`.
func Wipe(root string) (int64, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return 0, err
	}
	size := dirSize(abs)
	entries, err := os.ReadDir(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil // nothing to wipe is success, not an error
		}
		return size, fmt.Errorf("store: wipe %s: %w", abs, err)
	}
	for _, e := range entries {
		if err := os.RemoveAll(filepath.Join(abs, e.Name())); err != nil {
			return size, fmt.Errorf("store: wipe %s: %w", filepath.Join(abs, e.Name()), err)
		}
	}
	return size, nil
}

// scan rebuilds the in-memory index from disk. Cheap: N small JSON reads.
//
// A corrupt record is SKIPPED with a log line rather than failing the scan. The rest of
// the scratch data is still perfectly usable, and a service that refuses to start because
// one of two hundred files is truncated is worse than one that starts with 199.
func (s *FileStore) scan() {
	entries, err := os.ReadDir(s.docsDir)
	if err != nil {
		slog.Warn("[STORE] cannot list documents", "dir", s.docsDir, "err", err)
		return
	}
	sort.Slice(entries, func(a, b int) bool { return entries[a].Name() < entries[b].Name() })

	loaded := 0
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		file := filepath.Join(s.docsDir, entry.Name(), "record.json")
		data, err := os.ReadFile(file)
		if err != nil {
			continue
		}
		var rec model.Document
		if err := json.Unmarshal(data, &rec); err != nil {
			slog.Warn("[STORE] skipping unreadable record", "file", file, "err", err)
			continue
		}
		if rec.Quality == nil {
			rec.Quality = map[string]any{}
		}
		s.records[rec.ID] = &rec
		if rec.ID+1 > s.nextDocID {
			s.nextDocID = rec.ID + 1
		}
		loaded++
	}

	if data, err := os.ReadFile(s.apiKeysPath()); err == nil {
		var keys []*model.ApiKey
		if err := json.Unmarshal(data, &keys); err != nil {
			slog.Warn("[STORE] api_keys.json unreadable — starting with none", "err", err)
		} else {
			for _, k := range keys {
				s.apiKeys[k.ID] = k
				if k.ID+1 > s.nextKeyID {
					s.nextKeyID = k.ID + 1
				}
			}
		}
	}

	if data, err := os.ReadFile(s.settingsPath()); err == nil {
		var values map[string]string
		if err := json.Unmarshal(data, &values); err != nil {
			slog.Warn("[STORE] settings.json unreadable — using defaults", "err", err)
		} else {
			s.settings = values
		}
	}

	if loaded > 0 {
		slog.Info("[STORE] recovered documents", "count", loaded, "dir", s.docsDir)
	}
}

func (s *FileStore) apiKeysPath() string  { return filepath.Join(s.root, "api_keys.json") }
func (s *FileStore) settingsPath() string { return filepath.Join(s.root, "settings.json") }

// DocDir is the artifact directory for one document.
func (s *FileStore) DocDir(id int) string {
	return filepath.Join(s.docsDir, strconv.Itoa(id))
}

func (s *FileStore) Backend() string   { return "files" }
func (s *FileStore) IsEphemeral() bool { return true }

// -- documents --------------------------------------------------------------

func (s *FileStore) NextDocumentID() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := s.nextDocID
	s.nextDocID++
	return id
}

func (s *FileStore) AllRecords() []*model.Document {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*model.Document, 0, len(s.records))
	for _, r := range s.records {
		out = append(out, r.Clone())
	}
	// Deterministic order regardless of map iteration, which Go randomises. Callers sort
	// by their own key afterwards, but an unstable base order makes equal keys shuffle
	// between requests — visible in the UI as rows jumping.
	sort.Slice(out, func(a, b int) bool { return out[a].ID < out[b].ID })
	return out
}

// GetRecord returns a COPY with the lazily-stored result attached.
//
// A copy rather than the indexed instance: callers mutate what they get back, and sharing
// would let one request's edit leak into another's view.
func (s *FileStore) GetRecord(id int) *model.Document {
	s.mu.Lock()
	rec := s.records[id]
	s.mu.Unlock()
	if rec == nil {
		return nil
	}
	out := rec.Clone()
	// Loaded OUTSIDE the lock: it is a file read of up to 100 KB, and holding the index
	// lock across it would serialise every other reader for no benefit.
	out.Result = s.LoadResultPayload(id)
	return out
}

// PutRecord persists a record and indexes it.
//
// The FILE IS WRITTEN BEFORE THE INDEX ENTRY, and the order matters: a record is what
// makes a document visible to the worker, so indexing it before the bytes exist lets the
// drain loop claim a document whose file is not written yet.
func (s *FileStore) PutRecord(rec *model.Document) *model.Document {
	dir := s.DocDir(rec.ID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		slog.Error("[STORE] cannot create document dir", "dir", dir, "err", err)
		return rec
	}
	if err := AtomicWriteJSON(filepath.Join(dir, "record.json"), rec); err != nil {
		slog.Error("[STORE] cannot write record", "id", rec.ID, "err", err)
		return rec
	}
	stored := rec.Clone()
	s.mu.Lock()
	s.records[rec.ID] = stored
	if rec.ID+1 > s.nextDocID {
		s.nextDocID = rec.ID + 1
	}
	s.mu.Unlock()
	return rec
}

func (s *FileStore) DropRecord(id int) {
	s.mu.Lock()
	delete(s.records, id)
	s.mu.Unlock()
	// Outside the lock: removing a directory of multi-megabyte artifacts is slow, and
	// nothing else can reach the record now that it is out of the index.
	_ = os.RemoveAll(s.DocDir(id))
}

// -- api keys ---------------------------------------------------------------

func (s *FileStore) AllApiKeys() []*model.ApiKey {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*model.ApiKey, 0, len(s.apiKeys))
	for _, k := range s.apiKeys {
		copied := *k
		out = append(out, &copied)
	}
	sort.Slice(out, func(a, b int) bool { return out[a].ID < out[b].ID })
	return out
}

func (s *FileStore) NextApiKeyID() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := s.nextKeyID
	s.nextKeyID++
	return id
}

func (s *FileStore) PutApiKey(key *model.ApiKey) (*model.ApiKey, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	stored := *key
	s.apiKeys[key.ID] = &stored
	if key.ID+1 > s.nextKeyID {
		s.nextKeyID = key.ID + 1
	}
	return key, s.flushApiKeysLocked()
}

func (s *FileStore) DropApiKey(id int) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.apiKeys[id]; !ok {
		return false, nil
	}
	delete(s.apiKeys, id)
	return true, s.flushApiKeysLocked()
}

// flushApiKeysLocked assumes the lock is held. Named so the assumption is visible at the
// call site — see the note on FileStore about non-reentrancy.
func (s *FileStore) flushApiKeysLocked() error {
	keys := make([]*model.ApiKey, 0, len(s.apiKeys))
	for _, k := range s.apiKeys {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(a, b int) bool { return keys[a].ID < keys[b].ID })
	return AtomicWriteJSON(s.apiKeysPath(), keys)
}

// -- settings ---------------------------------------------------------------

func (s *FileStore) AllSettings() map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]string, len(s.settings))
	for k, v := range s.settings {
		out[k] = v
	}
	return out
}

func (s *FileStore) SetSettings(values map[string]string) (map[string]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k, v := range values {
		s.settings[k] = v
	}
	out := make(map[string]string, len(s.settings))
	for k, v := range s.settings {
		out[k] = v
	}
	return out, AtomicWriteJSON(s.settingsPath(), s.settings)
}

// -- results ----------------------------------------------------------------

func (s *FileStore) SaveResultPayload(id int, payload map[string]any) error {
	dir := s.DocDir(id)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return AtomicWriteJSON(filepath.Join(dir, "result.json"), payload)
}

func (s *FileStore) LoadResultPayload(id int) map[string]any {
	data, err := os.ReadFile(filepath.Join(s.DocDir(id), "result.json"))
	if err != nil {
		return nil
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		slog.Warn("[STORE] unreadable result.json", "id", id, "err", err)
		return nil
	}
	return out
}

// -- queries ----------------------------------------------------------------
// Implemented over the in-memory index. Correct at this scale (a few hundred records) and
// honest about it: a SQL backend answers the same questions with real queries.

func (s *FileStore) QueryDocuments(q Query) ([]*model.Document, int) {
	rows := s.AllRecords()

	if q.Status != "" {
		rows = filter(rows, func(r *model.Document) bool { return r.Status == q.Status })
	}
	// '__none__' means "unrecognised", which is not the same as "no doc_type": a failed
	// document has neither, and the UI offers one filter for both.
	if q.DocType == "__none__" {
		rows = filter(rows, func(r *model.Document) bool { return !r.Recognised })
	} else if q.DocType != "" {
		rows = filter(rows, func(r *model.Document) bool {
			return r.DocType != nil && strings.HasPrefix(*r.DocType, q.DocType)
		})
	}
	if start, ok := parseDay(q.DateFrom); ok {
		rows = filter(rows, func(r *model.Document) bool {
			return r.CreatedAt.Set && !r.CreatedAt.Time.Before(start)
		})
	}
	if end, ok := parseDay(q.DateTo); ok {
		// Inclusive of the whole named day, which is what a date picker means by "to".
		limit := end.AddDate(0, 0, 1)
		rows = filter(rows, func(r *model.Document) bool {
			return r.CreatedAt.Set && r.CreatedAt.Time.Before(limit)
		})
	}
	if needle := strings.ToLower(strings.TrimSpace(q.Search)); needle != "" {
		rows = filter(rows, func(r *model.Document) bool {
			return strings.Contains(r.SearchText, needle)
		})
	}

	total := len(rows)

	column := q.SortBy
	if !SortColumns[column] {
		column = "created_at"
	}
	desc := q.SortDir != "asc"
	sortRows(rows, column, desc)

	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 20
	}
	page := q.Page
	if page < 1 {
		page = 1
	}
	offset := (page - 1) * pageSize
	if offset > len(rows) {
		offset = len(rows)
	}
	end := offset + pageSize
	if end > len(rows) {
		end = len(rows)
	}
	return rows[offset:end], total
}

// sortRows orders by one whitelisted column.
//
// NULLS LAST IN BOTH DIRECTIONS, matching the SQL backend: a queued document has no
// doc_conf and must not lead an ascending sort. That is why the comparison is on a
// (isNull, value) pair rather than on the value alone.
//
// SliceStable, not Slice: equal keys must keep their previous relative order, or rows jump
// between refreshes of the list page for no visible reason.
func sortRows(rows []*model.Document, column string, desc bool) {
	sort.SliceStable(rows, func(a, b int) bool {
		an, av := sortKey(rows[a], column)
		bn, bv := sortKey(rows[b], column)
		if an != bn {
			// A null sorts last whichever direction is requested, so the test is on
			// nullness alone and is not reversed below.
			return bn
		}
		if av == bv {
			return false
		}
		if desc {
			return av > bv
		}
		return av < bv
	})
}

// sortKey returns (isNull, comparable) for a column.
//
// Every column reduces to a STRING key so that one comparator covers dates, numbers and
// text alike, instead of a type switch inside the comparator. Numbers go through numKey,
// which renders them in a lexicographically ordered fixed width; timestamps use RFC 3339,
// which sorts correctly as text by construction.
func sortKey(r *model.Document, column string) (isNull bool, key string) {
	switch column {
	case "filename":
		return false, strings.ToLower(r.Filename)
	case "status":
		return false, r.Status
	case "doc_type":
		if r.DocType == nil {
			return true, ""
		}
		return false, *r.DocType
	case "doc_conf":
		if r.DocConf == nil {
			return true, ""
		}
		return false, numKey(*r.DocConf)
	case "processing_ms":
		if r.ProcessingMs == nil {
			return true, ""
		}
		return false, numKey(float64(*r.ProcessingMs))
	case "size_bytes":
		return false, numKey(float64(r.SizeBytes))
	default: // created_at
		if !r.CreatedAt.Set {
			return true, ""
		}
		return false, r.CreatedAt.Time.UTC().Format(time.RFC3339Nano)
	}
}

// numKey renders a number as a fixed-width, lexicographically ordered string.
//
// This exists so ONE string comparator can order every column, numeric and textual alike,
// instead of a type switch in the comparator. The offset keeps negatives ordered correctly;
// the width is chosen to cover every value these columns can hold (a byte count, a
// millisecond count, a 0..1 confidence).
func numKey(v float64) string {
	return fmt.Sprintf("%020.6f", v+1e9)
}

func (s *FileStore) NextQueuedID() (int, bool) {
	var best *model.Document
	for _, r := range s.AllRecords() {
		if r.Status != model.StatusQueued {
			continue
		}
		if best == nil || earlier(r, best) {
			best = r
		}
	}
	if best == nil {
		return 0, false
	}
	return best.ID, true
}

func (s *FileStore) QueuePosition(id int) (int, bool) {
	queued := filter(s.AllRecords(), func(r *model.Document) bool {
		return r.Status == model.StatusQueued
	})
	sort.SliceStable(queued, func(a, b int) bool { return earlier(queued[a], queued[b]) })
	for i, r := range queued {
		if r.ID == id {
			return i, true
		}
	}
	return 0, false
}

// earlier is FIFO by creation, with the ID as the tie-breaker.
//
// The tie-break is not decoration: two uploads inside the same clock tick would otherwise
// have an unspecified order, and the queue would not be FIFO in exactly the case where
// somebody is testing it by uploading twice quickly.
func earlier(a, b *model.Document) bool {
	if a.CreatedAt.Set && b.CreatedAt.Set && !a.CreatedAt.Time.Equal(b.CreatedAt.Time) {
		return a.CreatedAt.Time.Before(b.CreatedAt.Time)
	}
	return a.ID < b.ID
}

func (s *FileStore) CountByStatus() map[string]int {
	counts := map[string]int{
		model.StatusQueued: 0, model.StatusProcessing: 0,
		model.StatusDone: 0, model.StatusFailed: 0,
	}
	for _, r := range s.AllRecords() {
		counts[r.Status]++
	}
	return counts
}

func (s *FileStore) AggregateStats() Stats {
	rows := s.AllRecords()
	counts := s.CountByStatus()

	out := Stats{
		Queued: counts[model.StatusQueued], Processing: counts[model.StatusProcessing],
		Done: counts[model.StatusDone], Failed: counts[model.StatusFailed],
		Total: len(rows),
	}
	var sum, n int
	for _, r := range rows {
		if r.Recognised {
			out.Recognised++
		}
		if r.Status == model.StatusDone && r.ProcessingMs != nil && *r.ProcessingMs > 0 {
			sum += *r.ProcessingMs
			n++
		}
	}
	if n > 0 {
		avg := int(float64(sum)/float64(n) + 0.5)
		out.AvgProcessingMs = &avg
	}
	return out
}

func (s *FileStore) DiskUsageBytes() int64 { return dirSize(s.docsDir) }

func dirSize(root string) int64 {
	var total int64
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			// A vanished file mid-walk is normal here (the worker writes while the
			// status page reads), so an error skips the entry rather than the walk.
			return nil
		}
		if info, err := d.Info(); err == nil {
			total += info.Size()
		}
		return nil
	})
	return total
}

func filter(rows []*model.Document, keep func(*model.Document) bool) []*model.Document {
	out := rows[:0]
	for _, r := range rows {
		if keep(r) {
			out = append(out, r)
		}
	}
	return out
}

// parseDay accepts YYYY-MM-DD.
//
// A HALF-TYPED DATE DISABLES THE FILTER rather than erroring: the list page sends the
// field on every keystroke, and rejecting "2026-0" would make the page flash an error while
// somebody is still typing.
func parseDay(value string) (time.Time, bool) {
	if value == "" {
		return time.Time{}, false
	}
	t, err := time.Parse("2006-01-02", value)
	if err != nil {
		return time.Time{}, false
	}
	return t.UTC(), true
}
