package api

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/errs"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/logging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/repo"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/sysinfo"
)

// --- auth -------------------------------------------------------------------

// handlePinLogin exchanges the PIN for a session JWT.
//
// The failure message deliberately does not distinguish "wrong PIN" from "malformed request":
// there is nothing useful for a legitimate user in the difference, and there is something
// useful in it for somebody guessing.
func (s *Server) handlePinLogin(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Pin string `json:"pin"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, clientError(errs.ErrBadRequest, `expected {"pin": "..."}`))
		return
	}
	if !auth.VerifyPin(s.authCfg(), body.Pin) {
		// Logged, because repeated failures are the only signal available without rate
		// limiting — see the note in the auth package about what a PIN is and is not.
		slog.Warn("[API] PIN login rejected")
		writeJSON(w, http.StatusUnauthorized, errorBody{Detail: "Incorrect PIN"})
		return
	}
	token, err := auth.CreateAccessToken(s.authCfg(), "operator")
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"access_token": token,
		"token_type":   "bearer",
		"user":         map[string]any{"name": sessionUser.Name, "role": sessionUser.Role},
	})
}

// --- api keys ---------------------------------------------------------------

// ephemeralKeyNote is rendered verbatim by the keys page.
//
// Surfaced so the UI can WARN rather than letting a restart quietly delete a key somebody
// pasted into a config somewhere. The text is copied from the reference so both services say
// the same thing.
const ephemeralKeyNote = "Keys created here live in ephemeral storage and are lost when " +
	"the service restarts. The default key comes from the environment and always exists."

func (s *Server) handleListKeys(w http.ResponseWriter, r *http.Request, _ *Identity) {
	keys, err := repo.PublicApiKeys(s.db, s.authCfg())
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"items": keys,
		// api-keys/Index.vue reads `res.note` and renders it as a banner. Omitting it left
		// an empty warning div on the page.
		"note": ephemeralKeyNote,
	})
}

// handleCreateKey mints a key and returns the PLAINTEXT exactly once.
func (s *Server) handleCreateKey(w http.ResponseWriter, r *http.Request, _ *Identity) {
	var body struct {
		Label string `json:"label"`
	}
	// A missing body is fine — the label is optional and defaults. Erroring here would make
	// "create a key" require a payload for no reason.
	_ = json.NewDecoder(r.Body).Decode(&body)

	rec, plaintext, err := repo.CreateApiKey(s.db, body.Label)
	if err != nil {
		writeError(w, err)
		return
	}
	slog.Info("[API] api key created", "id", rec.ID, "label", rec.Label)

	out := rec.Public()
	// The ONLY response that ever carries the full key. After this it exists nowhere but
	// the caller's hands and a sha256 in the store.
	out["key"] = plaintext
	out["warning"] = "Copy this key now — it will not be shown again."
	writeJSON(w, http.StatusCreated, out)
}

// handleDeleteKey refuses to delete the default key.
//
// 409, not 403: the request is well-formed and the caller is allowed to delete keys — it is
// the STATE that forbids this one. Deleting it would also be silently undone by the next
// restart, since it is derived from the environment rather than stored.
func (s *Server) handleDeleteKey(w http.ResponseWriter, r *http.Request, _ *Identity, id int) {
	if id == repo.DefaultKeyID {
		writeError(w, clientError(errs.ErrConflict,
			"The default key comes from the environment and cannot be deleted"))
		return
	}
	removed, err := repo.DeleteApiKey(s.db, id)
	if err != nil {
		writeError(w, err)
		return
	}
	if !removed {
		writeError(w, clientError(errs.ErrNotFound, "Key not found"))
		return
	}
	writeNoContent(w)
}

// --- settings ---------------------------------------------------------------

func (s *Server) handleGetSettings(w http.ResponseWriter, r *http.Request, _ *Identity) {
	writeJSON(w, http.StatusOK, map[string]any{
		"schema": repo.SettingsSchema(),
		"values": repo.AllSettings(s.db, s.cfg),
	})
}

// handlePutSettings validates and stores, reporting which changes need a restart.
//
// `restart_required` is not decoration: compute_device and ocr_mode are baked into the
// pipeline at construction, so a UI that reported "saved" and left the runtime alone would be
// lying about something an operator can verify on the status page.
func (s *Server) handlePutSettings(w http.ResponseWriter, r *http.Request, _ *Identity) {
	// **The body is WRAPPED: {"values": {...}}.** settings/Index.vue posts
	// `Api.put('/settings', { values })`, and the reference declares a SettingsUpdate model
	// with a single `values` field. An earlier version here parsed the object FLAT, which was
	// the worst possible failure: `values` is not a schema key, so the whitelist dropped it,
	// nothing was stored, and the page reported success. Exactly the "reports saved while
	// discarding the value" outcome the settings layer is written to avoid.
	var body struct {
		Values map[string]any `json:"values"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Values == nil {
		writeError(w, clientError(errs.ErrBadRequest,
			`expected {"values": {...}}`))
		return
	}
	values, restart, err := repo.BulkUpdateSettings(s.db, s.cfg, body.Values)
	if err != nil {
		writeError(w, err)
		return
	}
	if restart == nil {
		// An empty ARRAY, not null: the page assigns it straight to a list it iterates.
		restart = []string{}
	}
	if len(restart) > 0 {
		slog.Info("[API] settings changed, restart required", "keys", restart)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"values": values,
		// The schema travels back with the values, matching the reference, so a client that
		// only ever calls PUT still has everything it needs to render the form.
		"schema":           repo.SettingsSchema(),
		"restart_required": restart,
	})
}

// --- logs -------------------------------------------------------------------

// handleLogs serves the ring buffer.
//
// **The response key is `entries` and the count parameter is `n`.** Both are fixed by the
// shared frontend: logs/Index.vue sends `{ n: 400 }` and reads `res.entries`. An earlier
// version returned `{"items": ...}` and accepted `limit`, which produced a valid response the
// page could not read — an EMPTY logs page with a 200 and no error anywhere. The same class of
// mistake as the status block: when the UI is shared, the UI owns the wire format.
//
// `count` is sent alongside because the reference sends it. The page derives its "N lines"
// label from the array length, so nothing reads it today — but a client that trusted the
// documented shape would.
func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request, _ *Identity) {
	q := r.URL.Query()
	// Bounds copied from the reference (1..2000), not from the buffer capacity: asking for
	// more than the buffer holds is not an error, it just returns everything there is.
	n, err := queryInt(q, "n", 200, 1, 2000)
	if err != nil {
		writeError(w, err)
		return
	}
	entries := logging.Entries(n, q.Get("level"), q.Get("search"))
	writeJSON(w, http.StatusOK, map[string]any{
		"count":   len(entries),
		"entries": entries,
	})
}

// --- status -----------------------------------------------------------------

// handleStatus reports what the service is actually doing.
//
// **The field names are fixed by the SHARED FRONTEND.** `web/` is reused unchanged by every
// port, so status/Index.vue is the contract — it reads `server.cpu_pct`, `gpu.vram_used_gb`,
// `service.data_is_ephemeral` and the rest BY NAME. An earlier version of this handler returned
// a thinner, more Go-shaped block, and the status page rendered completely empty.
//
// `device` and `ocr_device` come through SEPARATELY on purpose: with GPU detectors the OCR
// engines still run on CPU, and a page that just says "GPU active" invites a bug report the
// first time somebody watches nvidia-smi during recognition.
func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request, _ *Identity) {
	stats := repo.Stats(s.db)

	writeJSON(w, http.StatusOK, map[string]any{
		"server": sysinfo.ReadServer(),
		// nil when there is no GPU, no driver, or a CPU-only container. The status page then
		// shows the compute block alone, which is the part that answers whether the GPU is
		// being used at all.
		"gpu":     sysinfo.ReadGPU(),
		"compute": s.rt.Info(),
		"service": map[string]any{
			"uptime_sec":           int(time.Since(s.startedAt).Seconds()),
			"version":              s.cfg.GitCommit,
			"documents_queued":     stats.Queued,
			"documents_processing": stats.Processing,
			"documents_done":       stats.Done,
			"documents_failed":     stats.Failed,
			"documents_total":      stats.Total,
			"recognised":           stats.Recognised,
			"avg_processing_ms":    stats.AvgProcessingMs,
			"data_dir_mb":          round1(float64(s.db.DiskUsageBytes()) / 1e6),
			// The SPA reads this from `service`, not from `storage`. The Python service
			// puts it only under `storage`, so its own status page always renders
			// "Retained" — a real defect on that side, recorded in the progress log rather
			// than reproduced here.
			"data_is_ephemeral": s.db.IsEphemeral(),
		},
		// Which backend is live, so an operator can tell at a glance whether what they are
		// looking at survives a restart.
		"storage": map[string]any{
			"backend":   s.db.Backend(),
			"ephemeral": s.db.IsEphemeral(),
		},
	})
}

// handleHealth is the container healthcheck. No auth, no store access, no runtime dependency.
//
// It reports OK while the models are still loading, deliberately: the service IS healthy then
// — it accepts uploads and queues them. Gating health on the runtime would make Docker kill
// the container during the fifteen seconds it needs to start.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":  "ok",
		"runtime": s.rt.Info().State,
	})
}
