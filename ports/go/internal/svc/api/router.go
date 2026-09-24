package api

import (
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
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

	// authMode is the EFFECTIVE mode, resolved once from AUTH_MODE and the storage backend by
	// auth.ResolveMode; downgradeReason says why it differs from what was configured, or is
	// nil. No handler reads cfg.AuthMode.
	authMode        string
	downgradeReason *string
	// throttle is per-process state, like the store index: the service is pinned to one
	// process, so an in-memory counter is the whole mechanism.
	throttle *auth.Throttle
}

func NewServer(db store.DocumentStore, rt *runtime.Runtime, wk *worker.Worker,
	cfg config.Settings, webRoot string) *Server {

	mode, reason := auth.ResolveMode(cfg.AuthMode, db.Backend())
	return &Server{db: db, rt: rt, worker: wk, cfg: cfg,
		startedAt: time.Now(), webRoot: webRoot,
		authMode: mode, downgradeReason: reason,
		throttle: auth.NewThrottle(cfg.LoginMaxAttempts, cfg.LoginLockoutSeconds)}
}

// route is one entry of the API surface: method, path, the guard it declares, and the handler.
type route struct {
	Method string
	Path   string
	// Guard is nil for a public route.
	Guard   *Guard
	handler http.HandlerFunc
}

// GuardName is the declared guard's name, "public" for none — what the route-table test reads.
func (rt route) GuardName() string {
	if rt.Guard == nil {
		return guardPublic
	}
	return rt.Guard.Name
}

// routes is the whole API surface, as data.
//
// A list rather than a sequence of mux.HandleFunc calls, so the route-table test can compare
// EXACTLY what is served — every route, each with its named guard — against ports/AUTH.md §5.
// Handler registers this list and nothing else, so the table under test and the table in
// service cannot drift apart.
//
// Reading it as a PERMISSION LIST is the point: who may call what is visible at the place the
// route is declared. Role levels, lowest first:
//
//	documents, read          require_api_or_viewer    UI and integrations alike
//	documents, write         require_api_or_operator
//	status                   require_viewer           a session only; an integration has no
//	                                                  business reading service internals
//	purge, keys, settings,   require_admin            operator surface — an API key never
//	logs, users, audit                                reaches it
func (s *Server) routes() []route {
	h := func(g *Guard, fn func(http.ResponseWriter, *http.Request, *Identity)) http.HandlerFunc {
		return s.guard(g, fn)
	}
	return []route{
		// --- health: no prefix, no auth, for the container ---------------------
		{"GET", "/health", nil, s.handleHealth},

		// --- auth: public, obviously — these are how a caller gets a credential ---
		{"GET", Prefix + "/auth/config", nil, s.handleAuthConfig},
		{"POST", Prefix + "/auth/pin-login", nil, s.handlePinLogin},
		{"POST", Prefix + "/auth/login", nil, s.handleLogin},
		// The only two routes a session that owes a password change may reach.
		{"GET", Prefix + "/auth/me", requireSessionAllowPasswordChange,
			h(requireSessionAllowPasswordChange, s.handleMe)},
		{"POST", Prefix + "/auth/change-password", requireSessionAllowPasswordChange,
			h(requireSessionAllowPasswordChange, s.handleChangePassword)},

		// --- documents: API key OR session, by role ----------------------------
		// The same routes serve the bundled SPA and third-party integrations, which is why they
		// accept either credential rather than being duplicated per audience.
		{"GET", Prefix + "/documents", requireApiOrViewer, h(requireApiOrViewer, s.handleList)},
		{"GET", Prefix + "/documents/{id}", requireApiOrViewer,
			h(requireApiOrViewer, s.withID(s.handleGetDocument))},
		{"GET", Prefix + "/documents/{id}/progress", requireApiOrViewer,
			h(requireApiOrViewer, s.withID(s.handleProgress))},
		{"GET", Prefix + "/documents/{id}/image/{kind}", requireApiOrViewer,
			h(requireApiOrViewer, func(w http.ResponseWriter, r *http.Request, id *Identity) {
				docID, err := pathID(r)
				if err != nil {
					writeError(w, err)
					return
				}
				s.handleImage(w, r, id, docID, r.PathValue("kind"))
			})},
		{"POST", Prefix + "/documents", requireApiOrOperator, h(requireApiOrOperator, s.handleUpload)},
		{"POST", Prefix + "/documents/{id}/reprocess", requireApiOrOperator,
			h(requireApiOrOperator, s.withID(s.handleReprocess))},
		{"DELETE", Prefix + "/documents/{id}", requireApiOrOperator,
			h(requireApiOrOperator, s.withID(s.handleDelete))},
		// Purge is the one document route that is session-only and admin-only: it empties the
		// store for everyone, which is service management rather than document work.
		{"POST", Prefix + "/documents/purge", requireAdmin, h(requireAdmin, s.handlePurge)},

		// --- operator surface: session only -----------------------------------
		{"GET", Prefix + "/status", requireViewer, h(requireViewer, s.handleStatus)},
		{"GET", Prefix + "/api-keys", requireAdmin, h(requireAdmin, s.handleListKeys)},
		{"POST", Prefix + "/api-keys", requireAdmin, h(requireAdmin, s.handleCreateKey)},
		{"DELETE", Prefix + "/api-keys/{id}", requireAdmin,
			h(requireAdmin, s.withID(s.handleDeleteKey))},
		{"GET", Prefix + "/settings", requireAdmin, h(requireAdmin, s.handleGetSettings)},
		{"PUT", Prefix + "/settings", requireAdmin, h(requireAdmin, s.handlePutSettings)},
		{"GET", Prefix + "/logs", requireAdmin, h(requireAdmin, s.handleLogs)},

		// --- user management: admin, users mode (404 in PIN mode) --------------
		{"GET", Prefix + "/users", requireAdmin, h(requireAdmin, s.handleListUsers)},
		{"POST", Prefix + "/users", requireAdmin, h(requireAdmin, s.handleCreateUser)},
		{"PATCH", Prefix + "/users/{id}", requireAdmin,
			h(requireAdmin, s.withID(s.handleUpdateUser))},
		{"POST", Prefix + "/users/{id}/password", requireAdmin,
			h(requireAdmin, s.withID(s.handleResetPassword))},
		{"DELETE", Prefix + "/users/{id}", requireAdmin,
			h(requireAdmin, s.withID(s.handleDeleteUser))},
		// The action log answers in BOTH modes — see handleListAudit.
		{"GET", Prefix + "/users/audit/entries", requireAdmin, h(requireAdmin, s.handleListAudit)},
	}
}

// Handler builds the routing table from routes().
//
// net/http's ServeMux with method patterns, no third-party router. `/users/audit/entries` and
// `/users/{id}` do not collide: the mux prefers the more specific pattern, and the literal
// segments win.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	for _, rt := range s.routes() {
		mux.HandleFunc(rt.Method+" "+rt.Path, rt.handler)
	}

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
						"GET, POST, PUT, PATCH, DELETE, OPTIONS")
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
