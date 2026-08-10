package api

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"unicode"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/errs"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/repo"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// The document resource: upload, browse, inspect, re-run, delete.
//
// Serialisation is hand-written `row`/`detail` functions rather than derived from the record
// struct, which is the reference's convention and keeps the wire format visible in ONE place
// instead of spread across tags on a type that also has to satisfy the store.

const maxFilenameLen = 200

// safeFilename keeps a DISPLAY NAME only — it never touches the filesystem.
//
// Stored artifacts always use a fixed name, so even a hostile filename cannot escape the
// document directory. This is purely so the UI shows something sensible and bounded; it is
// NOT the path-traversal defence, and treating it as one would be a mistake, because the real
// defence is that the name is never used as a path at all.
func safeFilename(raw string) string {
	name := raw
	if i := strings.LastIndexAny(strings.ReplaceAll(name, "\\", "/"), "/"); i >= 0 {
		name = strings.ReplaceAll(name, "\\", "/")[i+1:]
	}
	name = strings.TrimSpace(name)

	var b strings.Builder
	for _, r := range name {
		if !unicode.IsPrint(r) || strings.ContainsRune(`<>:"|?*`, r) {
			continue
		}
		b.WriteRune(r)
	}
	out := b.String()
	if out == "" {
		out = "upload"
	}
	// Truncated by RUNES, not bytes: a Cyrillic filename cut mid-rune renders as a
	// replacement character, and these names are routinely Cyrillic here.
	runes := []rune(out)
	if len(runes) > maxFilenameLen {
		runes = runes[:maxFilenameLen]
	}
	return string(runes)
}

// row is one line of the document log.
func row(rec *model.Document) map[string]any {
	var base, era *string
	if rec.DocType != nil {
		b, e := splitDocType(*rec.DocType)
		if b != "" {
			base = &b
		}
		if e != "" {
			era = &e
		}
	}
	quality := rec.Quality
	if quality == nil {
		quality = map[string]any{}
	}
	return map[string]any{
		"id":            rec.ID,
		"filename":      rec.Filename,
		"size_bytes":    rec.SizeBytes,
		"status":        rec.Status,
		"doc_type":      rec.DocType,
		"doc_type_base": base,
		"doc_type_era":  era,
		"recognised":    rec.Recognised,
		"doc_conf":      rec.DocConf,
		"quality":       quality,
		"field_count":   rec.FieldCount,
		"device":        rec.Device,
		"processing_ms": rec.ProcessingMs,
		"error":         rec.Error,
		"error_code":    rec.ErrorCode,
		"retry_count":   rec.RetryCount,
		"has_canvas":    rec.HasCanvas,
		"created_at":    rec.CreatedAt,
		"started_at":    rec.StartedAt,
		"finished_at":   rec.FinishedAt,
	}
}

func splitDocType(label string) (base, era string) {
	if i := strings.LastIndex(label, "_"); i >= 0 {
		return label[:i], label[i+1:]
	}
	return label, ""
}

// detail is the row plus the stored view model flattened into it.
//
// The stored result already has the client-facing shape — boxes, fields, canvas dimensions,
// coordinate-space notes — so this adds URLs and the original's dimensions and otherwise
// passes it through. Re-deriving any of it here would create a second definition of the wire
// format.
func detail(rec *model.Document) map[string]any {
	payload := row(rec)
	result := rec.Result
	if result == nil {
		result = map[string]any{}
	}

	canvas := map[string]any{}
	if c, ok := result["canvas"].(map[string]any); ok {
		for k, v := range c {
			canvas[k] = v
		}
	}
	canvas["url"] = fmt.Sprintf("/api/v1/documents/%d/image/canvas", rec.ID)

	payload["canvas"] = canvas
	payload["original"] = map[string]any{
		"url":          fmt.Sprintf("/api/v1/documents/%d/image/original", rec.ID),
		"width":        rec.OriginalW,
		"height":       rec.OriginalH,
		"content_type": rec.ContentType,
	}
	payload["coord_space"] = result["coord_space"]
	payload["coord_space_note"] = result["coord_space_note"]
	payload["boxes"] = orEmptyList(result["boxes"])
	payload["fields"] = orEmptyList(result["fields"])
	payload["ocr"] = orEmptyMap(result["ocr"])
	payload["quality"] = orEmptyMap(result["quality"])
	payload["timings"] = orEmptyMap(result["timings"])
	payload["address"] = result["address"]
	return payload
}

// orEmptyList and orEmptyMap keep a missing key from becoming a JSON null where the client
// expects a container. The SPA iterates `boxes` and `fields` unconditionally, so a null there
// is a runtime error in the browser rather than an empty table.
func orEmptyList(v any) any {
	if list, ok := v.([]any); ok && list != nil {
		return list
	}
	return []any{}
}

func orEmptyMap(v any) any {
	if m, ok := v.(map[string]any); ok && m != nil {
		return m
	}
	return map[string]any{}
}

// handleUpload accepts one image and queues it. 202 with the FULL LIST ROW, so the SPA can
// insert the row without a second request.
//
// Everything cheap is checked HERE, so a bad upload fails immediately with an actionable
// message instead of becoming a mysterious failed job a minute later.
func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request, _ *Identity) {
	limit := s.cfg.MaxUploadBytes()
	// The body is capped BEFORE reading, so an oversized upload cannot exhaust memory
	// while being measured. Reading limit+1 is what distinguishes "at the limit" from
	// "over it".
	r.Body = http.MaxBytesReader(w, r.Body, limit+1)

	if err := r.ParseMultipartForm(8 << 20); err != nil {
		// **An oversized upload fails HERE, not at the size check below.** MaxBytesReader
		// aborts the read as soon as the cap is passed, so ParseMultipartForm is what returns
		// the error — and reporting it as a malformed request would tell the user to fix their
		// client when the actual problem is a 40 MB file. The size check further down only
		// catches a body that fit within the cap.
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeJSON(w, http.StatusRequestEntityTooLarge, errorBody{
				Detail: fmt.Sprintf("File exceeds the %d MB limit", s.cfg.MaxUploadMB)})
			return
		}
		writeError(w, clientError(errs.ErrBadRequest, "expected a multipart upload with a 'file' part"))
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		writeError(w, clientError(errs.ErrBadRequest, "no 'file' part in the upload"))
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		writeError(w, clientError(errs.ErrBadRequest, "could not read the upload"))
		return
	}
	if int64(len(data)) > limit {
		writeJSON(w, http.StatusRequestEntityTooLarge,
			errorBody{Detail: fmt.Sprintf("File exceeds the %d MB limit", s.cfg.MaxUploadMB)})
		return
	}
	if len(data) == 0 {
		writeError(w, clientError(errs.ErrBadRequest, "Empty upload"))
		return
	}

	if repo.IsPDF(data) {
		// Called out separately because people WILL try it, and "unsupported image type"
		// does not tell them what to do about it.
		writeJSON(w, http.StatusUnsupportedMediaType, errorBody{
			Detail: "PDF is not supported — upload a JPEG, PNG, WEBP, BMP or TIFF image"})
		return
	}
	ext, mediaType, ok := repo.SniffImage(data)
	if !ok {
		// Sniffed from MAGIC BYTES, not the client's Content-Type, which is
		// attacker-controlled and wrong often enough to be useless.
		writeJSON(w, http.StatusUnsupportedMediaType,
			errorBody{Detail: "Unsupported file type — expected an image"})
		return
	}
	width, height, decoded := repo.DecodeDimensions(data)
	if !decoded {
		writeError(w, clientError(errs.ErrImageUnreadable, "The image could not be decoded — it may be corrupt"))
		return
	}

	filename := safeFilename(header.Filename)

	// **BYTES FIRST, ROW SECOND.** The record is what makes the document visible to the
	// worker, so writing it before the file leaves a window in which the drain loop can
	// claim a document whose original does not exist yet — reporting a perfectly good
	// upload as failed. See repo.ReserveID.
	id := repo.ReserveID(s.db)
	if _, err := repo.SaveOriginal(s.db, id, data, ext); err != nil {
		writeError(w, err)
		return
	}
	rec := model.NewDocument(id, filename, mediaType, int64(len(data)), ext)
	rec.OriginalW, rec.OriginalH = &width, &height
	rec.SearchText = strings.ToLower(filename)
	rec = repo.Create(s.db, rec)

	s.worker.NotifyNewWork()
	slog.Info("[API] queued document", "doc", rec.ID, "filename", filename, "bytes", len(data))

	out := row(rec)
	if pos, ok := repo.QueuePosition(s.db, rec.ID); ok {
		out["queue_position"] = pos
	} else {
		out["queue_position"] = nil
	}
	writeJSON(w, http.StatusAccepted, out)
}

// handleList serves one page of the document log.
func (s *Server) handleList(w http.ResponseWriter, r *http.Request, _ *Identity) {
	q := r.URL.Query()

	// The filter parameter is named `status` on the wire. Keeping that name is a client
	// dependency, not a preference.
	statusFilter := q.Get("status")
	if statusFilter != "" && !model.ValidStatuses[statusFilter] {
		writeError(w, clientError(errs.ErrBadRequest, "Invalid status"))
		return
	}
	sortDir := q.Get("sort_dir")
	if sortDir != "asc" && sortDir != "desc" {
		sortDir = "desc"
	}
	// Bounds copied from the reference's own declarations (service/api/documents.py:173-174):
	// page is ge=1 with NO upper bound, page_size is ge=1 le=100. Out of range is a 422, not a
	// clamp — see params.go.
	page, err := queryInt(q, "page", 1, 1, 0)
	if err != nil {
		writeError(w, err)
		return
	}
	pageSize, err := queryInt(q, "page_size", 20, 1, 100)
	if err != nil {
		writeError(w, err)
		return
	}

	rows, total := repo.GetAll(s.db, store.Query{
		Status:   statusFilter,
		DocType:  q.Get("doc_type"),
		Search:   q.Get("search"),
		DateFrom: q.Get("date_from"),
		DateTo:   q.Get("date_to"),
		Page:     page,
		PageSize: pageSize,
		SortBy:   q.Get("sort_by"),
		SortDir:  sortDir,
	})

	items := make([]map[string]any, 0, len(rows))
	for _, rec := range rows {
		items = append(items, row(rec))
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"items":     items,
		"total":     total,
		"page":      page,
		"page_size": pageSize,
		"stats":     repo.Stats(s.db),
	})
}

// clampInt is gone: silent clamping answered 200 to requests the reference rejects with 422.
// Bounded query integers go through queryInt in params.go, which reproduces the reference's
// pydantic-shaped rejection.

func (s *Server) handleGetDocument(w http.ResponseWriter, r *http.Request, _ *Identity, id int) {
	rec := repo.GetByID(s.db, id)
	if rec == nil {
		writeError(w, clientError(errs.ErrNotFound, "Document not found"))
		return
	}
	writeJSON(w, http.StatusOK, detail(rec))
}

// handleProgress returns live progress, a queue position, or a terminal state.
//
// **200 with a JSON null when there is nothing to report — never 404.** The polling client
// would otherwise raise an error toast every two seconds for a document that finished
// perfectly well.
func (s *Server) handleProgress(w http.ResponseWriter, r *http.Request, _ *Identity, id int) {
	rec := repo.GetByID(s.db, id)
	if rec == nil {
		writeError(w, clientError(errs.ErrNotFound, "Document not found"))
		return
	}

	if live := s.worker.DocumentProgress(id); live != nil {
		writeJSON(w, http.StatusOK, live)
		return
	}

	switch rec.Status {
	case model.StatusQueued:
		position := 0
		if p, ok := repo.QueuePosition(s.db, id); ok {
			position = p
		}
		pos := position
		writeJSON(w, http.StatusOK, map[string]any{
			"step":  "queued",
			"label": fmt.Sprintf("Queued (#%d)", position+1),
			"pct":   0,
			// The estimate is "everything ahead of me at the current average", which is
			// honest about being a guess and tracks reality because the average is an EMA
			// of real completions.
			"eta_sec":        round1(float64(position) * s.worker.AverageDurationSec()),
			"queue_position": &pos,
		})
	case model.StatusDone, model.StatusFailed:
		pct := 0
		if rec.Status == model.StatusDone {
			pct = 100
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"step":           rec.Status,
			"label":          strings.ToUpper(rec.Status[:1]) + rec.Status[1:],
			"pct":            pct,
			"eta_sec":        nil,
			"queue_position": nil,
		})
	default:
		// A JSON null body, deliberately. See the function note.
		writeJSON(w, http.StatusOK, nil)
	}
}

func round1(v float64) float64 {
	return float64(int(v*10+0.5)) / 10
}

// handleImage serves an artifact.
//
// `no-cache` means REVALIDATE, not "do not store": http.ServeFile still sends ETag and
// Last-Modified, so a repeat request costs a 304 with no body. `max-age` would be wrong here
// — Reprocess overwrites canvas.png and thumb.jpg at the SAME URL, so the browser would keep
// showing the previous recognition's image while the field table beside it was already new.
func (s *Server) handleImage(w http.ResponseWriter, r *http.Request, _ *Identity,
	id int, kind string) {

	switch kind {
	case "original", "canvas", "thumb":
	default:
		writeError(w, clientError(errs.ErrNotFound, "Unknown image kind"))
		return
	}
	path, mediaType, ok := repo.OpenArtifact(s.db, id, kind)
	if !ok {
		writeError(w, clientError(errs.ErrNotFound, "Image not available"))
		return
	}
	w.Header().Set("Content-Type", mediaType)
	w.Header().Set("Cache-Control", "private, no-cache")
	http.ServeFile(w, r, path)
}

func (s *Server) handleReprocess(w http.ResponseWriter, r *http.Request, _ *Identity, id int) {
	rec := repo.GetByID(s.db, id)
	if rec == nil {
		writeError(w, clientError(errs.ErrNotFound, "Document not found"))
		return
	}
	if repo.ActiveStatuses[rec.Status] {
		writeError(w, clientError(errs.ErrConflict, "Document is already %s", rec.Status))
		return
	}
	rec = repo.Requeue(s.db, rec)
	s.worker.NotifyNewWork()
	writeJSON(w, http.StatusOK, row(rec))
}

// handleDelete returns 204 and an EMPTY BODY.
func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request, _ *Identity, id int) {
	rec := repo.GetByID(s.db, id)
	if rec == nil {
		writeError(w, clientError(errs.ErrNotFound, "Document not found"))
		return
	}
	repo.Delete(s.db, rec)
	writeNoContent(w)
}

// handlePurge clears the scratch store. SESSION ONLY — not something an integration does.
func (s *Server) handlePurge(w http.ResponseWriter, r *http.Request, _ *Identity) {
	removed := 0
	for _, rec := range s.db.AllRecords() {
		// An in-flight job is left alone: deleting its record while the worker holds it
		// produces a "document vanished" failure that looks like a bug.
		if rec.Status == model.StatusProcessing {
			continue
		}
		repo.Delete(s.db, rec)
		removed++
	}
	slog.Info("[API] purged documents", "count", removed)
	writeJSON(w, http.StatusOK, map[string]any{"deleted": removed})
}
