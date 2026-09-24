package api

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/runtime"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/worker"
)

// THE ROUTE TABLE of ports/AUTH.md §5, transcribed. This is the whole surface: every route
// listed, nothing unlisted, each with the named guard.
//
// A test that only asked "is there A guard" passed on the first Python version while a viewer
// could delete everything — every document route was guarded, by a guard that checked only
// that the caller was someone. So this compares NAMES.
var authContractRoutes = []struct{ method, path, guard string }{
	{"GET", "/health", "public"},
	{"GET", "/api/v1/auth/config", "public"},
	{"POST", "/api/v1/auth/pin-login", "public"},
	{"POST", "/api/v1/auth/login", "public"},
	{"GET", "/api/v1/auth/me", "require_session_allow_password_change"},
	{"POST", "/api/v1/auth/change-password", "require_session_allow_password_change"},
	{"GET", "/api/v1/documents", "require_api_or_viewer"},
	{"GET", "/api/v1/documents/{id}", "require_api_or_viewer"},
	{"GET", "/api/v1/documents/{id}/progress", "require_api_or_viewer"},
	{"GET", "/api/v1/documents/{id}/image/{kind}", "require_api_or_viewer"},
	{"POST", "/api/v1/documents", "require_api_or_operator"},
	{"POST", "/api/v1/documents/{id}/reprocess", "require_api_or_operator"},
	{"DELETE", "/api/v1/documents/{id}", "require_api_or_operator"},
	{"POST", "/api/v1/documents/purge", "require_admin"},
	{"GET", "/api/v1/status", "require_viewer"},
	{"GET", "/api/v1/api-keys", "require_admin"},
	{"POST", "/api/v1/api-keys", "require_admin"},
	{"DELETE", "/api/v1/api-keys/{id}", "require_admin"},
	{"GET", "/api/v1/settings", "require_admin"},
	{"PUT", "/api/v1/settings", "require_admin"},
	{"GET", "/api/v1/logs", "require_admin"},
	{"GET", "/api/v1/users", "require_admin"},
	{"POST", "/api/v1/users", "require_admin"},
	{"PATCH", "/api/v1/users/{id}", "require_admin"},
	{"POST", "/api/v1/users/{id}/password", "require_admin"},
	{"DELETE", "/api/v1/users/{id}", "require_admin"},
	{"GET", "/api/v1/users/audit/entries", "require_admin"},
}

func TestRouteTableMatchesTheContract(t *testing.T) {
	s := newTestServer(t)
	want := map[string]string{}
	for _, r := range authContractRoutes {
		want[r.method+" "+r.path] = r.guard
	}
	got := map[string]string{}
	for _, r := range s.routes() {
		key := r.Method + " " + r.Path
		if _, dup := got[key]; dup {
			t.Errorf("route %s is declared twice", key)
		}
		got[key] = r.GuardName()
	}
	var keys []string
	for k := range want {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		g, ok := got[k]
		switch {
		case !ok:
			t.Errorf("contract route %s is not served", k)
		case g != want[k]:
			t.Errorf("%s is guarded by %s, contract says %s", k, g, want[k])
		}
	}
	for k := range got {
		if _, ok := want[k]; !ok {
			t.Errorf("route %s is served but not in the contract", k)
		}
	}
}

// The table above is DECLARED data; this checks the real handler honours it. Each guarded route,
// called with no credential, must answer 401 with WWW-Authenticate — before any body is read —
// and each public one must not.
func TestEveryGuardedRouteRefusesAnonymousCallers(t *testing.T) {
	s := newTestServer(t)
	handler := s.Handler()
	for _, r := range authContractRoutes {
		path := strings.NewReplacer("{id}", "1", "{kind}", "original").Replace(r.path)
		req := httptest.NewRequest(r.method, path, strings.NewReader("{}"))
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if r.guard == "public" {
			if rec.Code == http.StatusUnauthorized {
				t.Errorf("public %s %s answered 401", r.method, path)
			}
			continue
		}
		if rec.Code != http.StatusUnauthorized || rec.Header().Get("WWW-Authenticate") != "Bearer" {
			t.Errorf("anonymous %s %s: HTTP %d, WWW-Authenticate %q; want 401 Bearer",
				r.method, path, rec.Code, rec.Header().Get("WWW-Authenticate"))
		}
	}
}

// --- users-mode helpers ----------------------------------------------------------

const testSecret = "unit-test-secret-0123456789abcdef"

func newModeServer(t *testing.T, mode string) *Server {
	t.Helper()
	db, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Defaults()
	cfg.AuthMode = mode
	cfg.JwtSecret = testSecret
	cfg.DefaultApiKey = "rdk_test_contract_key"
	rt := runtime.New()
	return NewServer(db, rt, worker.New(db, rt, cfg), cfg, "")
}

func (s *Server) testAccount(t *testing.T, name, role string, mustChange bool) (*model.User, string) {
	t.Helper()
	u, err := s.db.PutUser(model.NewUser(s.db.NextUserID(), name, role, "x", "", mustChange))
	if err != nil {
		t.Fatal(err)
	}
	token, err := s.accountToken(u)
	if err != nil {
		t.Fatal(err)
	}
	return u, token
}

func call(s *Server, method, path, token, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	return rec
}

func detailOf(rec *httptest.ResponseRecorder) string {
	var body struct {
		Detail string `json:"detail"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	return body.Detail
}

// forgeToken signs arbitrary claims with the test secret, the way an attacker holding it would.
func forgeToken(claims string) string {
	enc := base64.RawURLEncoding
	signing := enc.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`)) + "." +
		enc.EncodeToString([]byte(claims))
	mac := hmac.New(sha256.New, []byte(testSecret))
	mac.Write([]byte(signing))
	return signing + "." + enc.EncodeToString(mac.Sum(nil))
}

func exp() string { return strconv.FormatInt(time.Now().Add(time.Hour).Unix(), 10) }

// **PIN mode REFUSES a token carrying uid** — refused, not ignored. Every PIN session is the
// administrator, so ignoring the uid would promote a viewer's still-valid token to full control
// the moment the service is switched back to PIN. `"uid": null` counts as carrying one.
func TestPinModeRefusesAccountTokens(t *testing.T) {
	s := newModeServer(t, "pin")
	pin := forgeToken(`{"sub":"operator","name":"Operator","role":"admin","exp":` + exp() + `}`)
	if rec := call(s, "GET", Prefix+"/status", pin, ""); rec.Code != http.StatusOK {
		t.Fatalf("control: a PIN token was refused: %d %s", rec.Code, rec.Body)
	}
	for _, claims := range []string{
		`{"sub":"viewer1","uid":2,"tv":1,"role":"viewer","exp":` + exp() + `}`,
		`{"sub":"operator","role":"admin","uid":null,"exp":` + exp() + `}`,
		`{"sub":"operator","role":"admin","uid":0,"exp":` + exp() + `}`,
	} {
		if rec := call(s, "GET", Prefix+"/status", forgeToken(claims), ""); rec.Code != http.StatusUnauthorized {
			t.Errorf("PIN mode admitted %s: HTTP %d", claims, rec.Code)
		}
	}
}

// Users mode: a PIN-era token (no uid) is worthless, and the identity comes from the store.
func TestUsersModeResolvesFromTheStore(t *testing.T) {
	s := newModeServer(t, "users")
	pin := forgeToken(`{"sub":"operator","name":"Operator","role":"admin","exp":` + exp() + `}`)
	if rec := call(s, "GET", Prefix+"/status", pin, ""); rec.Code != http.StatusUnauthorized {
		t.Errorf("users mode admitted a PIN-era token: %d", rec.Code)
	}

	u, token := s.testAccount(t, "viewer1", model.RoleViewer, false)
	// A token that CLAIMS admin but belongs to a viewer: the role comes from the store.
	liar := forgeToken(`{"sub":"viewer1","uid":` + strconv.Itoa(u.ID) +
		`,"tv":1,"role":"admin","exp":` + exp() + `}`)
	if rec := call(s, "GET", Prefix+"/settings", liar, ""); rec.Code != http.StatusForbidden {
		t.Errorf("the token's role claim was trusted: %d", rec.Code)
	}
	// A stale token_version is dead.
	stale := forgeToken(`{"sub":"viewer1","uid":` + strconv.Itoa(u.ID) +
		`,"tv":0,"role":"viewer","exp":` + exp() + `}`)
	if rec := call(s, "GET", Prefix+"/status", stale, ""); rec.Code != http.StatusUnauthorized {
		t.Errorf("a stale token_version was accepted: %d", rec.Code)
	}
	// A deactivated account's token is dead at once.
	if rec := call(s, "GET", Prefix+"/status", token, ""); rec.Code != http.StatusOK {
		t.Fatalf("control: viewer token refused: %d %s", rec.Code, rec.Body)
	}
	u.IsActive = false
	if _, err := s.db.PutUser(u); err != nil {
		t.Fatal(err)
	}
	if rec := call(s, "GET", Prefix+"/status", token, ""); rec.Code != http.StatusUnauthorized {
		t.Errorf("a disabled account's token was accepted: %d", rec.Code)
	}
}

// Roles on real routes, with the exact texts, and the guard before the body: a viewer's PUT with
// an empty object and an upload with no file part are both 403, not a validation error.
func TestRolesAndTheRestrictedSession(t *testing.T) {
	s := newModeServer(t, "users")
	_, viewer := s.testAccount(t, "viewer1", model.RoleViewer, false)
	_, operator := s.testAccount(t, "op1", model.RoleOperator, false)
	_, restricted := s.testAccount(t, "fresh", model.RoleAdmin, true)

	for _, c := range []struct {
		token, method, path, body string
		status                    int
		detail                    string
	}{
		{viewer, "GET", Prefix + "/documents", "", 200, ""},
		{viewer, "POST", Prefix + "/documents", "", 403, "This action requires the operator role"},
		{viewer, "DELETE", Prefix + "/documents/1", "", 403, "This action requires the operator role"},
		{viewer, "PUT", Prefix + "/settings", "{}", 403, "This action requires the admin role"},
		{viewer, "GET", Prefix + "/users/audit/entries", "", 403, "This action requires the admin role"},
		{operator, "DELETE", Prefix + "/documents/999", "", 404, ""},
		{operator, "POST", Prefix + "/documents/purge", "", 403, "This action requires the admin role"},
		{operator, "GET", Prefix + "/users", "", 403, "This action requires the admin role"},
		{restricted, "GET", Prefix + "/documents", "", 403, "password_change_required"},
		{restricted, "GET", Prefix + "/status", "", 403, "password_change_required"},
		{restricted, "GET", Prefix + "/users", "", 403, "password_change_required"},
		{restricted, "GET", Prefix + "/auth/me", "", 200, ""},
	} {
		rec := call(s, c.method, c.path, c.token, c.body)
		if rec.Code != c.status || (c.detail != "" && detailOf(rec) != c.detail) {
			t.Errorf("%s %s: %d %q, want %d %q", c.method, c.path, rec.Code, detailOf(rec),
				c.status, c.detail)
		}
	}

	// An API key reaches the documents and nothing else: 401, not 403, everywhere else.
	for _, path := range []string{"/status", "/users", "/logs", "/users/audit/entries"} {
		req := httptest.NewRequest("GET", Prefix+path, nil)
		req.Header.Set("X-API-Key", "rdk_test_contract_key")
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized || detailOf(rec) != "Sign in to use this endpoint" {
			t.Errorf("API key on %s: %d %q", path, rec.Code, detailOf(rec))
		}
	}
}

// /auth/me in PIN mode: the keys an operator identity does not have are null, not absent.
func TestMeInPinModeHasNullAccountFields(t *testing.T) {
	s := newModeServer(t, "pin")
	pin := forgeToken(`{"sub":"operator","name":"Operator","role":"admin","exp":` + exp() + `}`)
	rec := call(s, "GET", Prefix+"/auth/me", pin, "")
	var body struct {
		Mode string         `json:"mode"`
		User map[string]any `json:"user"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Mode != "pin" || body.User["name"] != "Operator" || body.User["role"] != "admin" {
		t.Errorf("me: %s", rec.Body)
	}
	for _, key := range []string{"username", "user_id", "must_change_password"} {
		v, present := body.User[key]
		if !present || v != nil {
			t.Errorf("me.user.%s = %v (present %v), want null", key, v, present)
		}
	}
}

// The ADMIN_PASSWORD rule on /auth/config: demo credentials only for the documented default,
// and only while the seeded account still owes its change.
func TestDemoCredentialsOnlyForTheDefault(t *testing.T) {
	s := newModeServer(t, "users")
	s.testAccount(t, "admin", model.RoleAdmin, true)
	demo := func() any {
		rec := call(s, "GET", Prefix+"/auth/config", "", "")
		var body map[string]any
		_ = json.Unmarshal(rec.Body.Bytes(), &body)
		return body["demo_credentials"]
	}
	if d, ok := demo().(map[string]any); !ok || d["username"] != "admin" || d["password"] != "1234" {
		t.Errorf("demo credentials missing with the default password: %v", demo())
	}
	s.cfg.AdminPassword = "Real-Secret-1"
	if demo() != nil {
		t.Error("a real ADMIN_PASSWORD caused credentials to be advertised")
	}
}
