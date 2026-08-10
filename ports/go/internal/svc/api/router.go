package api

import (
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/errs"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/runtime"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/worker"
)

// Prefix is the API root. Versioned, because a published REST contract that cannot change
// shape is a published REST contract that gets replaced by a second service.
const Prefix = "/api/v1"

// Server holds the dependencies every handler needs.
//
// Explicit fields rather than a service locator or context values: a handler's dependencies
// are then visible in one place, and the .NET and Kotlin ports get constructor injection
// without a framework.
type Server struct {
	db     store.DocumentStore
	rt     *runtime.Runtime
	worker *worker.Worker
	cfg    config.Settings

	startedAt time.Time
	webRoot   string
}

func NewServer(db store.DocumentStore, rt *runtime.Runtime, wk *worker.Worker,
	cfg config.Settings, webRoot string) *Server {

	return &Server{db: db, rt: rt, worker: wk, cfg: cfg,
		startedAt: time.Now(), webRoot: webRoot}
}

// Handler builds the routing table.
//
// net/http's ServeMux with method patterns, no third-party router. The routes below are the
// whole surface, and reading them as a PERMISSION LIST is the point: `guard(requireSession,
// ...)` versus `guard(requireApiOrSession, ...)` says who may call what, at the place the
// route is declared.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// --- auth: no credential required, obviously ---------------------------
	mux.HandleFunc("POST "+Prefix+"/auth/pin-login", s.handlePinLogin)

	// --- documents: API key OR session ------------------------------------
	// The same routes serve the bundled SPA and third-party integrations, which is why they
	// accept either credential rather than being duplicated per audience.
	mux.HandleFunc("POST "+Prefix+"/documents", s.guard(s.requireApiOrSession, s.handleUpload))
	mux.HandleFunc("GET "+Prefix+"/documents", s.guard(s.requireApiOrSession, s.handleList))
	mux.HandleFunc("POST "+Prefix+"/documents/purge", s.guard(s.requireSession, s.handlePurge))

	mux.HandleFunc("GET "+Prefix+"/documents/{id}",
		s.guard(s.requireApiOrSession, s.withID(s.handleGetDocument)))
	mux.HandleFunc("DELETE "+Prefix+"/documents/{id}",
		s.guard(s.requireApiOrSession, s.withID(s.handleDelete)))
	mux.HandleFunc("GET "+Prefix+"/documents/{id}/progress",
		s.guard(s.requireApiOrSession, s.withID(s.handleProgress)))
	mux.HandleFunc("POST "+Prefix+"/documents/{id}/reprocess",
		s.guard(s.requireApiOrSession, s.withID(s.handleReprocess)))
	mux.HandleFunc("GET "+Prefix+"/documents/{id}/image/{kind}",
		s.guard(s.requireApiOrSession, func(w http.ResponseWriter, r *http.Request, id *Identity) {
			docID, err := pathID(r)
			if err != nil {
				writeError(w, err)
				return
			}
			s.handleImage(w, r, id, docID, r.PathValue("kind"))
		}))

	// --- operator surface: session only -----------------------------------
	// An integration has no business managing keys, settings or logs, so these do not accept
	// an API key at all.
	mux.HandleFunc("GET "+Prefix+"/api-keys", s.guard(s.requireSession, s.handleListKeys))
	mux.HandleFunc("POST "+Prefix+"/api-keys", s.guard(s.requireSession, s.handleCreateKey))
	mux.HandleFunc("DELETE "+Prefix+"/api-keys/{id}",
		s.guard(s.requireSession, s.withID(s.handleDeleteKey)))
	mux.HandleFunc("GET "+Prefix+"/settings", s.guard(s.requireSession, s.handleGetSettings))
	mux.HandleFunc("PUT "+Prefix+"/settings", s.guard(s.requireSession, s.handlePutSettings))
	mux.HandleFunc("GET "+Prefix+"/logs", s.guard(s.requireSession, s.handleLogs))
	mux.HandleFunc("GET "+Prefix+"/status", s.guard(s.requireSession, s.handleStatus))

	// --- health: no prefix, no auth, for the container ---------------------
	mux.HandleFunc("GET /health", s.handleHealth)

	// --- the SPA, as a catch-all ------------------------------------------
	mux.HandleFunc("/", s.handleSPA)

	return s.withMiddleware(mux)
}

// withID adapts a handler that needs the {id} path value.
//
// Parsed once, here, so no handler repeats it and none of them can disagree about what a
// non-numeric id means (a 404, because the route does not exist for that path — not a 400,
// which would suggest the request could be fixed).
func (s *Server) withID(h func(http.ResponseWriter, *http.Request, *Identity, int)) func(
	http.ResponseWriter, *http.Request, *Identity) {

	return func(w http.ResponseWriter, r *http.Request, id *Identity) {
		docID, err := pathID(r)
		if err != nil {
			writeError(w, err)
			return
		}
		h(w, r, id, docID)
	}
}

func pathID(r *http.Request) (int, error) {
	raw := r.PathValue("id")
	v, err := strconv.Atoi(raw)
	if err != nil || v < 0 {
		return 0, clientError(errs.ErrNotFound, "not a document id")
	}
	return v, nil
}

// withMiddleware adds CORS, request logging and panic recovery.
//
// Recovery is OUTERMOST so it covers everything, including the logger. A panic in one handler
// must not take down a service that is mid-way through recognising a document in another
// goroutine — and Go's default behaviour for a panic in a handler is to kill the connection
// silently, which is indistinguishable from a network fault.
func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				slog.Error("[API] panic in handler", "method", r.Method,
					"path", r.URL.Path, "panic", rec)
				writeJSON(w, http.StatusInternalServerError,
					errorBody{Detail: "Internal server error"})
			}
		}()

		if origins := s.cfg.CorsOrigins(); len(origins) > 0 {
			origin := r.Header.Get("Origin")
			for _, allowed := range origins {
				// Exact match only. A wildcard reflected back with credentials enabled is
				// the classic CORS mistake, and this service authenticates every route
				// that matters.
				if allowed == origin {
					w.Header().Set("Access-Control-Allow-Origin", origin)
					w.Header().Set("Vary", "Origin")
					w.Header().Set("Access-Control-Allow-Headers",
						"Authorization, Content-Type, X-API-Key")
					w.Header().Set("Access-Control-Allow-Methods",
						"GET, POST, PUT, DELETE, OPTIONS")
					break
				}
			}
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}
		}

		next.ServeHTTP(w, r)
	})
}

// handleSPA serves the built frontend, falling back to index.html for client-side routes.
//
// Two things here are security-relevant rather than cosmetic:
//
//   - the resolved path is checked to be INSIDE the web root after symlink resolution, so a
//     crafted path cannot escape it. filepath.Clean alone is not enough on a tree that may
//     contain links;
//   - anything under the API prefix that reached here is a 404 in JSON, not the SPA. Serving
//     HTML for an unknown API route makes a client's JSON parse fail with a message about
//     '<', which is a genuinely confusing way to learn a route was misspelled.
func (s *Server) handleSPA(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, Prefix) {
		writeJSON(w, http.StatusNotFound, errorBody{Detail: "Not found"})
		return
	}
	if s.webRoot == "" {
		writeJSON(w, http.StatusNotFound, errorBody{
			Detail: "No frontend build found; run `npm run build` in web/"})
		return
	}

	rel := strings.TrimPrefix(r.URL.Path, "/")
	if rel == "" {
		rel = "index.html"
	}
	candidate := filepath.Join(s.webRoot, filepath.Clean("/"+rel))

	root, err := filepath.EvalSymlinks(s.webRoot)
	if err != nil {
		writeJSON(w, http.StatusNotFound, errorBody{Detail: "Not found"})
		return
	}
	if resolved, err := filepath.EvalSymlinks(candidate); err == nil {
		if !strings.HasPrefix(resolved, root) {
			// Outside the web root: treated as not found rather than forbidden, so a
			// prober learns nothing about the filesystem layout.
			writeJSON(w, http.StatusNotFound, errorBody{Detail: "Not found"})
			return
		}
		if info, err := os.Stat(resolved); err == nil && !info.IsDir() {
			http.ServeFile(w, r, resolved)
			return
		}
	}

	// A client-side route: hand back index.html and let the SPA router resolve it.
	index := filepath.Join(root, "index.html")
	if info, err := os.Stat(index); err == nil && !info.IsDir() {
		// no-cache on the shell only: the hashed asset files under /assets are immutable
		// and get the server's default caching, but a cached index.html pins the client to
		// an old bundle after a deploy.
		w.Header().Set("Cache-Control", "no-cache")
		http.ServeFile(w, r, index)
		return
	}
	writeJSON(w, http.StatusNotFound, errorBody{Detail: "Not found"})
}

// FindWebRoot locates a built frontend, or "" if there is none.
//
// Tries web/dist first and then web/, matching the reference: dist is the production build,
// while the bare directory is what a developer has before running the bundler. Returning ""
// rather than failing is deliberate — the API is fully usable without a UI, and an integration
// does not care that npm was never run.
func FindWebRoot(repoRoot string) string {
	if repoRoot == "" {
		return ""
	}
	for _, rel := range []string{filepath.Join("web", "dist"), "web"} {
		candidate := filepath.Join(repoRoot, rel)
		if info, err := os.Stat(filepath.Join(candidate, "index.html")); err == nil && !info.IsDir() {
			return candidate
		}
	}
	return ""
}
