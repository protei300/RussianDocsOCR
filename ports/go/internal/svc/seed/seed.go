// Package seed populates an empty store with pre-computed sample documents.
//
// A blank log is a bad first impression and an unhelpful one: there is nothing to click, so
// nothing demonstrates what the service does. Seeding means the box overlay, the field table and
// the timings are visible the moment the page loads, across every supported document type.
//
// **The results are pre-computed, not re-derived.** `service/seed_data/` holds one finished
// recognition per document type — the view model, the rendered canvas and a thumbnail —
// generated once by `service/tools/build_seed_data.py` and committed. Seeding is therefore a
// FILE COPY: no GPU, no model load, no minute of startup latency, and the same rows every time
// regardless of the host's hardware.
//
// **THIS PORT READS THE SAME DIRECTORY AS THE PYTHON SERVICE.** That is the point, and it is
// worth being explicit about: the seeded corpus is ONE artifact with one generator
// (`service/tools/build_seed_data.py`), consumed by every port. A second copy under `ports/go/`
// would drift from the first the moment recognition changed, and then two services would
// disagree about what the reference behaviour is while both looked internally consistent. The
// directory's name says `service/` for historical reasons — it predates the ports — and moving
// it would touch the Python service, its builder, the docs and CLAUDE.md for a rename; the
// SHARING is what matters, not the path.
//
// Three rules keep this from becoming a nuisance, all carried over from the reference:
//
//   - Only into an EMPTY store. With a database configured the first run seeds and later runs
//     find the rows already there, so nothing piles up and a deleted document stays deleted.
//   - Only ANONYMISED repository samples. Never a user upload, never a local personal file —
//     everything seeded here is visible to anyone who can reach the UI.
//   - ONE PER DOCUMENT TYPE, in the manifest's order, so the log shows the breadth of what the
//     library handles rather than nineteen driving licences.
//
// Re-run the builder after any change to recognition, or the seeded rows quietly describe an
// older version's behaviour.
//
// Port of service/core/seed.py.
package seed

import (
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/repo"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// Entry is one manifest row. Field names match the committed manifest.json exactly.
type Entry struct {
	Slug        string `json:"slug"`
	Sample      string `json:"sample"`
	Filename    string `json:"filename"`
	OriginalExt string `json:"original_ext"`
	ContentType string `json:"content_type"`
	SizeBytes   int64  `json:"size_bytes"`
	SearchText  string `json:"search_text"`
}

// Dir returns the shared seed directory for a repository root.
func Dir(repoRoot string) string { return filepath.Join(repoRoot, "service", "seed_data") }

// IfEmpty inserts the pre-computed samples when the store holds nothing.
//
// `limit` caps how many are inserted; 0 means all available. Returns how many were added.
//
// NEVER RETURNS AN ERROR: a service that cannot seed its demo data must still start and accept
// real uploads. Every failure path logs and continues, and one bad fixture does not stop the
// others.
func IfEmpty(db store.DocumentStore, repoRoot string, limit int) int {
	total := 0
	for _, n := range db.CountByStatus() {
		total += n
	}
	if total > 0 {
		return 0
	}

	dir := Dir(repoRoot)
	entries, err := loadManifest(dir)
	if err != nil || len(entries) == 0 {
		slog.Warn("[SEED] no pre-computed data — the log starts empty; run "+
			"`python service/tools/build_seed_data.py`", "dir", dir, "err", err)
		return 0
	}
	if limit > 0 && limit < len(entries) {
		entries = entries[:limit]
	}

	added := 0
	for _, entry := range entries {
		if err := seedOne(db, repoRoot, dir, entry); err != nil {
			slog.Warn("[SEED] skipping fixture", "slug", entry.Slug, "err", err)
			continue
		}
		added++
	}
	slog.Info("[SEED] inserted pre-computed sample documents", "count", added)
	return added
}

func loadManifest(dir string) ([]Entry, error) {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var entries []Entry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

func seedOne(db store.DocumentStore, repoRoot, seedDir string, entry Entry) error {
	entryDir := filepath.Join(seedDir, entry.Slug)

	payloadBytes, err := os.ReadFile(filepath.Join(entryDir, "result.json"))
	if err != nil {
		return err
	}
	var payload map[string]any
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return err
	}

	// The original is NOT duplicated into the fixture set — it is the repository sample the
	// result was computed from, which is also what keeps the seed data committable.
	sample := filepath.Join(repoRoot, filepath.FromSlash(entry.Sample))
	data, err := os.ReadFile(sample)
	if err != nil {
		return err
	}

	// Same BYTES-BEFORE-ROW ordering as an upload. Safe either way here, because seeding
	// finishes before the worker starts — but two orderings for one invariant is how the
	// unsafe one survives a refactor.
	id := repo.ReserveID(db)
	if _, err := repo.SaveOriginal(db, id, data, entry.OriginalExt); err != nil {
		return err
	}

	rec := model.NewDocument(id, entry.Filename, entry.ContentType, entry.SizeBytes,
		entry.OriginalExt)
	if w, h, ok := repo.DecodeDimensions(data); ok {
		rec.OriginalW, rec.OriginalH = &w, &h
	}
	rec.SearchText = entry.SearchText
	// Timestamps are NOW rather than the build time, so the log's relative dates ("2 minutes
	// ago") stay sane however old the committed fixtures are.
	now := model.At(model.UtcNow())
	rec.CreatedAt, rec.StartedAt = now, now
	rec = repo.Create(db, rec)

	dstDir, err := repo.DocDir(db, rec.ID)
	if err != nil {
		return err
	}
	for _, name := range []string{"canvas.jpg", "thumb.jpg"} {
		// A missing preview is not fatal: the fields are the product and the picture is a
		// convenience, exactly as in the worker's own canvas-write path.
		if err := copyFile(filepath.Join(entryDir, name), filepath.Join(dstDir, name)); err != nil &&
			!os.IsNotExist(err) {
			slog.Warn("[SEED] could not copy artifact", "slug", entry.Slug, "file", name, "err", err)
		}
	}

	// `timings.total` is the library's own value, in SECONDS (spec/viewmodel.md), while the
	// record stores milliseconds.
	processingMs := 0
	if timings, ok := payload["timings"].(map[string]any); ok {
		if total, ok := timings["total"].(float64); ok {
			processingMs = int(total*1000 + 0.5)
		}
	}
	if _, err := repo.SaveResult(db, rec, payload, entry.SearchText, processingMs); err != nil {
		return err
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	// Synced before returning: the seeded rows are marked done immediately, so a crash right
	// after startup must not leave a document whose canvas is a zero-length file.
	return out.Sync()
}
