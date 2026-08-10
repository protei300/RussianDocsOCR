package api

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
	svclog "github.com/protei300/RussianDocsOCR/ports/go/internal/svc/logging"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/runtime"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/worker"
)

// THE WIRE CONTRACT OF THE OPERATOR PAGES.
//
// These tests exist because the same mistake was made twice, and both times the failure was
// silent: a 200 response, well-formed JSON, no error anywhere, and a page that rendered
// completely empty.
//
//	/status  returned a Go-shaped `server` block instead of cpu_pct/cpu_name/ram_used_gb/...
//	/logs    returned {"items": ...} where the page reads res.entries, and accepted `limit`
//	         where the page sends `n`
//
// **`web/` is reused UNCHANGED by every port, so the SPA owns the wire format.** The key lists
// below are transcribed from the Vue sources named in each test; when a page starts reading a
// new field, this test is where that gets noticed rather than in a browser.
//
// Deliberately asserting KEYS and not values: the values are host-dependent and the point is
// the shape.

func newTestServer(t *testing.T) *Server {
	t.Helper()
	// A logger, so /logs has something to return. Nothing else in this setup logs, so a line
	// is emitted explicitly — the ring buffer is genuinely empty otherwise, and an assertion
	// that assumed incidental log traffic would be testing luck.
	svclog.Setup("DEBUG")
	slog.Info("[TEST] contract test started")

	db, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Defaults()
	cfg.DefaultApiKey = "rdk_test_contract_key"
	rt := runtime.New() // never initialised: no models are needed to check a wire shape
	wk := worker.New(db, rt, cfg)
	return NewServer(db, rt, wk, cfg, "")
}

// sessionToken mints a real session, because these routes are session-only and going through
// the actual guard is part of what is being checked.
func sessionToken(t *testing.T, s *Server) string {
	t.Helper()
	token, err := auth.CreateAccessToken(s.authCfg(), "operator")
	if err != nil {
		t.Fatal(err)
	}
	return token
}

func getJSON(t *testing.T, s *Server, path, token string) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("GET %s: HTTP %d, body %s", path, rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("GET %s: response is not a JSON object: %v", path, err)
	}
	return out
}

func requireKeys(t *testing.T, where string, obj map[string]any, keys ...string) {
	t.Helper()
	for _, key := range keys {
		if _, ok := obj[key]; !ok {
			present := make([]string, 0, len(obj))
			for k := range obj {
				present = append(present, k)
			}
			sort.Strings(present)
			t.Errorf("%s is missing %q; present: %v", where, key, present)
		}
	}
}

func child(t *testing.T, obj map[string]any, key string) map[string]any {
	t.Helper()
	sub, ok := obj[key].(map[string]any)
	if !ok {
		t.Fatalf("%q is not an object: %T", key, obj[key])
	}
	return sub
}

// Field names transcribed from web/src/views/pages/status/Index.vue.
func TestStatusWireContract(t *testing.T) {
	s := newTestServer(t)
	body := getJSON(t, s, Prefix+"/status", sessionToken(t, s))

	requireKeys(t, "status", body, "server", "gpu", "compute", "service", "storage")

	requireKeys(t, "status.server", child(t, body, "server"),
		"cpu_pct", "cpu_name", "cpu_cores", "cpu_threads",
		"ram_used_gb", "ram_total_gb", "disk_used_gb", "disk_total_gb")

	requireKeys(t, "status.compute", child(t, body, "compute"),
		"state", "device", "ocr_device", "providers", "fell_back",
		"model_format", "ocr_mode", "load_ms", "warmup_ms", "library_version")

	requireKeys(t, "status.service", child(t, body, "service"),
		"uptime_sec", "version", "documents_queued", "documents_processing",
		"documents_done", "documents_failed", "documents_total", "recognised",
		"avg_processing_ms", "data_dir_mb",
		// Read from `service`, not from `storage`. The Python service omits it, so its own
		// status page always shows "Retained" — a defect on that side, not a reason to match.
		"data_is_ephemeral")

	// `gpu` is nullable by design: no GPU, no driver, or a CPU-only container. The key must
	// still be PRESENT, or the page cannot tell "absent" from "not reported".
	if gpu, present := body["gpu"]; !present {
		t.Error("status is missing the gpu key; it must be present and null when there is none")
	} else if gpu != nil {
		requireKeys(t, "status.gpu", child(t, body, "gpu"),
			"name", "utilization_pct", "vram_used_gb", "vram_total_gb", "temperature_c")
	}
}

// Field names transcribed from web/src/views/pages/logs/Index.vue, which sends `{ n: 400 }` and
// reads `res.entries`.
func TestLogsWireContract(t *testing.T) {
	s := newTestServer(t)
	token := sessionToken(t, s)

	body := getJSON(t, s, Prefix+"/logs?n=50", token)
	requireKeys(t, "logs", body, "count", "entries")

	entries, ok := body["entries"].([]any)
	if !ok {
		t.Fatalf("logs.entries is not an array: %T", body["entries"])
	}
	// newTestServer emits a line, so this cannot legitimately be empty — an empty array here
	// means the handler is not reading the buffer at all.
	if len(entries) == 0 {
		t.Fatal("logs.entries is empty; the ring buffer is not being read")
	}
	first, ok := entries[0].(map[string]any)
	if !ok {
		t.Fatalf("logs.entries[0] is not an object: %T", entries[0])
	}
	requireKeys(t, "logs.entries[0]", first, "ts", "level", "logger", "message", "exc")

	// `ts` is UNIX SECONDS as a number: the page does `new Date(ts * 1000)`, so an ISO string
	// there renders as "Invalid Date".
	if _, ok := first["ts"].(float64); !ok {
		t.Errorf("logs.entries[0].ts is %T, want a number of seconds", first["ts"])
	}
	// The level string becomes a CSS class (`lv-` + level), so the vocabulary is fixed.
	level, _ := first["level"].(string)
	switch level {
	case "DEBUG", "INFO", "WARNING", "ERROR":
	default:
		t.Errorf("logs.entries[0].level = %q, outside the vocabulary the stylesheet knows", level)
	}
}

// `n` is the parameter the page sends. An implementation that only understood `limit` returned
// its default and looked like it worked.
func TestLogsRespectsN(t *testing.T) {
	s := newTestServer(t)
	token := sessionToken(t, s)

	body := getJSON(t, s, Prefix+"/logs?n=1", token)
	entries, _ := body["entries"].([]any)
	if len(entries) != 1 {
		t.Fatalf("n=1 returned %d entries; the count parameter is not named `n`", len(entries))
	}
	if count, _ := body["count"].(float64); int(count) != len(entries) {
		t.Errorf("count = %v but %d entries were returned", body["count"], len(entries))
	}
}

// The operator pages are session-only: an API key must NOT open them, because an integration
// has no business reading logs or managing settings.
func TestOperatorPagesRejectApiKeys(t *testing.T) {
	s := newTestServer(t)
	for _, path := range []string{"/status", "/logs", "/settings", "/api-keys"} {
		req := httptest.NewRequest(http.MethodGet, Prefix+path, nil)
		req.Header.Set("X-API-Key", "rdk_test_contract_key")
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		if rec.Code != http.StatusUnauthorized {
			t.Errorf("GET %s with an API key: HTTP %d, want 401", path, rec.Code)
		}
		// 401 must carry the header that makes it mean "retry with credentials".
		if rec.Header().Get("WWW-Authenticate") == "" {
			t.Errorf("GET %s: 401 without WWW-Authenticate", path)
		}
	}
}

// The error body is `{"detail": "<string>"}` everywhere, including for routing failures — the
// SPA's fetch wrapper reads `detail` and nothing else.
func TestErrorBodyShape(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, Prefix+"/documents/999", nil)
	req.Header.Set("X-API-Key", "rdk_test_contract_key")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("HTTP %d, want 404", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("error body is not JSON: %s", rec.Body.String())
	}
	detail, ok := body["detail"].(string)
	if !ok {
		t.Fatalf("error body has no string `detail`: %s", rec.Body.String())
	}
	// The sentinel's own name must not leak into the message — the defect that made a 409
	// read "conflict: The default key ...".
	for _, leaked := range []string{"not found:", "conflict:", "unauthorized:", "bad request:"} {
		if len(detail) >= len(leaked) && detail[:len(leaked)] == leaked {
			t.Errorf("detail %q starts with an internal sentinel name", detail)
		}
	}
}

// Field names transcribed from web/src/api/index.ts and the api-keys / settings pages. Each of
// these was a real defect found by code review AFTER the manual UI test, which is the argument
// for having them: a shape mismatch here is invisible from the server side.

// The keys page renders `res.note` as a banner. Omitting it left an empty warning div.
func TestApiKeysListCarriesNote(t *testing.T) {
	s := newTestServer(t)
	body := getJSON(t, s, Prefix+"/api-keys", sessionToken(t, s))

	requireKeys(t, "api-keys", body, "items", "note")
	if note, _ := body["note"].(string); note == "" {
		t.Error("note is empty; the page renders it verbatim as an ephemerality warning")
	}
	items, ok := body["items"].([]any)
	if !ok || len(items) == 0 {
		t.Fatalf("items is not a non-empty array: %T", body["items"])
	}
	// The default key always exists, and the row must never carry the hash.
	first, _ := items[0].(map[string]any)
	requireKeys(t, "api-keys.items[0]", first,
		"id", "label", "prefix", "masked", "is_default", "created_at", "last_used_at")
	if _, leaked := first["key_hash"]; leaked {
		t.Error("the key hash reached the UI payload")
	}
}

// **The PUT body is WRAPPED.** settings/Index.vue posts `{ values }`, and parsing it flat meant
// the whitelist dropped everything, nothing was stored, and the page reported success.
func TestSettingsPutTakesAWrappedBody(t *testing.T) {
	s := newTestServer(t)
	token := sessionToken(t, s)

	req := httptest.NewRequest(http.MethodPut, Prefix+"/settings",
		strings.NewReader(`{"values":{"docconf":"0.7"}}`))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d, body %s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	// The reference returns all three; the schema travels with the values so a client that
	// only ever calls PUT can still render the form.
	requireKeys(t, "settings PUT", body, "values", "schema", "restart_required")

	values := child(t, body, "values")
	if got := values["docconf"]; got != "0.7" {
		t.Errorf("docconf = %v, want \"0.7\" — the value was not stored, which is exactly "+
			"what a flat body parse looked like", got)
	}

	// And it must actually have persisted, not merely been echoed.
	after := getJSON(t, s, Prefix+"/settings", token)
	if got := child(t, after, "values")["docconf"]; got != "0.7" {
		t.Errorf("after a re-read docconf = %v; the write did not persist", got)
	}
}

// A rejected setting is 400, matching the reference — not 422, which is what FastAPI uses for
// its OWN validation errors and is therefore the tempting wrong answer.
func TestSettingsValidationIsBadRequest(t *testing.T) {
	s := newTestServer(t)

	req := httptest.NewRequest(http.MethodPut, Prefix+"/settings",
		strings.NewReader(`{"values":{"docconf":"5"}}`))
	req.Header.Set("Authorization", "Bearer "+sessionToken(t, s))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("HTTP %d, want 400", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	detail, _ := body["detail"].(string)
	// The message must name the BOUND, because that is the only actionable thing a form can
	// show — "invalid value" would not be.
	if !strings.Contains(detail, "docconf") || !strings.Contains(detail, "1") {
		t.Errorf("detail %q does not name the setting and its bound", detail)
	}
}

// The created key is the one and only time the plaintext exists outside the caller's hands, and
// the page reads `res.key` to show it once.
func TestCreateKeyReturnsThePlaintextOnce(t *testing.T) {
	s := newTestServer(t)

	req := httptest.NewRequest(http.MethodPost, Prefix+"/api-keys",
		strings.NewReader(`{"label":"integration"}`))
	req.Header.Set("Authorization", "Bearer "+sessionToken(t, s))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("HTTP %d, want 201, body %s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	requireKeys(t, "created key", body, "id", "label", "prefix", "masked", "key", "warning")

	key, _ := body["key"].(string)
	if !strings.HasPrefix(key, "rdk_") {
		t.Errorf("key %q lacks the rdk_ prefix", key)
	}
	// It must NOT come back on a subsequent list: only the hash is stored.
	list := getJSON(t, s, Prefix+"/api-keys", sessionToken(t, s))
	items, _ := list["items"].([]any)
	for _, raw := range items {
		row, _ := raw.(map[string]any)
		if row["is_default"] == true {
			continue // the generated default is revealed on purpose; see the repo layer
		}
		if _, present := row["key"]; present {
			t.Error("a stored key returned its plaintext on a later list")
		}
	}
}

// Deleting the environment key is refused with 409 — the request is well-formed and the caller
// is allowed to delete keys, but the STATE forbids this one, and "deleting" it would only be
// undone by the next restart.
func TestDeleteDefaultKeyIsConflict(t *testing.T) {
	s := newTestServer(t)

	req := httptest.NewRequest(http.MethodDelete, Prefix+"/api-keys/0", nil)
	req.Header.Set("Authorization", "Bearer "+sessionToken(t, s))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusConflict {
		t.Fatalf("HTTP %d, want 409", rec.Code)
	}
}

// TestBoundedQueryParamsMatchTheReference pins the pydantic-shaped 422.
//
// The expected bodies below were CAPTURED from the running reference, not written from memory —
// the whole class of defect this guards against is a plausible-looking hand-written shape. Before
// this existed the port CLAMPED silently, so `page_size=500` answered 200 with 100 rows: a
// successful reply to a request the reference rejects, which no amount of server-side testing
// would reveal.
func TestBoundedQueryParamsMatchTheReference(t *testing.T) {
	s := newTestServer(t)
	token := sessionToken(t, s)

	cases := []struct {
		path  string
		key   bool // API key instead of a session (the document list accepts either)
		typ   string
		param string
		msg   string
		input string
		ctx   map[string]int
	}{
		{path: "/api/v1/documents?page_size=500", key: true,
			typ: "less_than_equal", param: "page_size",
			msg: "Input should be less than or equal to 100", input: "500",
			ctx: map[string]int{"le": 100}},
		{path: "/api/v1/documents?page_size=0", key: true,
			typ: "greater_than_equal", param: "page_size",
			msg: "Input should be greater than or equal to 1", input: "0",
			ctx: map[string]int{"ge": 1}},
		{path: "/api/v1/documents?page=0", key: true,
			typ: "greater_than_equal", param: "page",
			msg: "Input should be greater than or equal to 1", input: "0",
			ctx: map[string]int{"ge": 1}},
		{path: "/api/v1/documents?page_size=abc", key: true,
			typ: "int_parsing", param: "page_size",
			msg:   "Input should be a valid integer, unable to parse string as an integer",
			input: "abc"},
		// An EMPTY value is a parse failure, not the default. Verified against the reference.
		{path: "/api/v1/documents?page_size=", key: true,
			typ: "int_parsing", param: "page_size",
			msg:   "Input should be a valid integer, unable to parse string as an integer",
			input: ""},
		{path: "/api/v1/logs?n=99999",
			typ: "less_than_equal", param: "n",
			msg: "Input should be less than or equal to 2000", input: "99999",
			ctx: map[string]int{"le": 2000}},
	}

	for _, c := range cases {
		req := httptest.NewRequest(http.MethodGet, c.path, nil)
		if c.key {
			req.Header.Set("X-API-Key", "rdk_test_contract_key")
		} else {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		if rec.Code != http.StatusUnprocessableEntity {
			t.Errorf("GET %s: HTTP %d, want 422; body %s", c.path, rec.Code, rec.Body.String())
			continue
		}
		var body struct {
			Detail []struct {
				Type  string         `json:"type"`
				Loc   []any          `json:"loc"`
				Msg   string         `json:"msg"`
				Input string         `json:"input"`
				Ctx   map[string]int `json:"ctx"`
			} `json:"detail"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Errorf("GET %s: detail is not the pydantic list shape: %v (%s)",
				c.path, err, rec.Body.String())
			continue
		}
		if len(body.Detail) != 1 {
			t.Errorf("GET %s: %d detail entries, want 1", c.path, len(body.Detail))
			continue
		}
		d := body.Detail[0]
		if d.Type != c.typ {
			t.Errorf("GET %s: type %q, want %q", c.path, d.Type, c.typ)
		}
		if len(d.Loc) != 2 || d.Loc[0] != "query" || d.Loc[1] != c.param {
			t.Errorf("GET %s: loc %v, want [query %s]", c.path, d.Loc, c.param)
		}
		if d.Msg != c.msg {
			t.Errorf("GET %s: msg %q, want %q", c.path, d.Msg, c.msg)
		}
		if d.Input != c.input {
			t.Errorf("GET %s: input %q, want %q", c.path, d.Input, c.input)
		}
		if c.ctx == nil {
			if d.Ctx != nil {
				t.Errorf("GET %s: ctx %v, want it ABSENT for a parse failure", c.path, d.Ctx)
			}
		} else {
			for k, v := range c.ctx {
				if d.Ctx[k] != v {
					t.Errorf("GET %s: ctx[%s]=%d, want %d", c.path, k, d.Ctx[k], v)
				}
			}
		}
	}
}

// TestAbsentQueryParamUsesTheDefault is the other half: aligning on rejection must not turn a
// missing parameter into an error. The SPA omits page_size on its first load.
func TestAbsentQueryParamUsesTheDefault(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/documents", nil)
	req.Header.Set("X-API-Key", "rdk_test_contract_key")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("bare GET /documents: HTTP %d, body %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out["page_size"] != float64(20) || out["page"] != float64(1) {
		t.Errorf("defaults not applied: page=%v page_size=%v", out["page"], out["page_size"])
	}
}
