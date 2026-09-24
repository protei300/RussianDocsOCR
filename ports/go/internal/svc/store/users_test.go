package store

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
)

// The account half of the repository contract suite — written against DocumentStore, like the
// rest of this package's tests, so a SQL backend inherits it unchanged.

func putUser(t *testing.T, s DocumentStore, id int, name, role string) *model.User {
	t.Helper()
	u, err := s.PutUser(model.NewUser(id, name, role, "$argon2id$stub", "", false))
	if err != nil {
		t.Fatal(err)
	}
	return u
}

// **Reads return copies.** The first Python version handed out the indexed instances, which let
// every repository function edit shared state before its own checks had finished — and made the
// "last administrator" rule a check-then-act race against whoever else held the same object.
// Each accessor is checked separately: a single leaking one is enough.
func TestUserReadsReturnCopies(t *testing.T) {
	s := openTemp(t)
	putUser(t, s, 1, "petrov", model.RoleViewer)

	got := s.GetUser(1)
	got.Role = model.RoleAdmin
	got.TokenVersion = 99
	if again := s.GetUser(1); again.Role != model.RoleViewer || again.TokenVersion != 1 {
		t.Errorf("mutating GetUser's result changed the store: %+v", again)
	}

	found := s.FindUser("PETROV")
	found.IsActive = false
	if again := s.GetUser(1); !again.IsActive {
		t.Error("mutating FindUser's result changed the store")
	}

	all := s.AllUsers()
	all[0].PasswordHash = "tampered"
	if again := s.GetUser(1); again.PasswordHash == "tampered" {
		t.Error("mutating AllUsers' result changed the store")
	}
}

// PutUser stores ITS OWN copy: a caller that keeps editing the value it passed in must not be
// editing the index.
func TestPutUserStoresACopy(t *testing.T) {
	s := openTemp(t)
	u := model.NewUser(1, "petrov", model.RoleViewer, "h", "", false)
	if _, err := s.PutUser(u); err != nil {
		t.Fatal(err)
	}
	u.Role = model.RoleAdmin
	if got := s.GetUser(1); got.Role != model.RoleViewer {
		t.Errorf("editing the value after PutUser changed the store: role %q", got.Role)
	}
}

func TestFindUserIsCaseInsensitiveAndTrims(t *testing.T) {
	s := openTemp(t)
	putUser(t, s, 1, "Admin", model.RoleAdmin)
	for _, q := range []string{"admin", "ADMIN", " Admin ", "aDmIn"} {
		if s.FindUser(q) == nil {
			t.Errorf("FindUser(%q) found nothing", q)
		}
	}
	if s.FindUser("аdmin") != nil { // Cyrillic а
		t.Error("a Cyrillic look-alike matched")
	}
}

func TestUsersSurviveReopenInTheSharedFormat(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	u := model.NewUser(1, "petrov", model.RoleOperator, "$argon2id$x", "Иван Петров", true)
	if _, err := s.PutUser(u); err != nil {
		t.Fatal(err)
	}
	putUser(t, s, 2, "sidorov", model.RoleViewer)

	raw, err := os.ReadFile(filepath.Join(dir, "users.json"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	// Exactly the fields the other three services read, UTF-8 without escaping.
	for _, field := range []string{`"id"`, `"username"`, `"role"`, `"password_hash"`,
		`"display_name"`, `"is_active"`, `"must_change_password"`, `"token_version"`,
		`"created_at"`, `"last_login_at"`} {
		if !strings.Contains(text, field) {
			t.Errorf("users.json lacks %s", field)
		}
	}
	if !strings.Contains(text, "Иван Петров") {
		t.Error("the display name was escaped rather than written as UTF-8")
	}
	if !strings.HasPrefix(strings.TrimSpace(text), "[") {
		t.Error("users.json is not a JSON array")
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	got := reopened.GetUser(1)
	if got == nil || got.DisplayName != "Иван Петров" || !got.MustChangePassword ||
		got.TokenVersion != 1 || !got.CreatedAt.Set || got.LastLoginAt.Set {
		t.Fatalf("user did not round-trip: %+v", got)
	}
	if next := reopened.NextUserID(); next != 3 {
		t.Errorf("NextUserID after reopen = %d, want 3", next)
	}
}

// Unreadable users.json / audit.jsonl at startup: log and start empty, never crash.
func TestUnreadableAccountFilesStartEmpty(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "users.json"), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "audit.jsonl"),
		[]byte("garbage\n{\"id\": 7, \"action\": \"login\", \"actor\": \"pin\", \"at\": null}\n"),
		0o644); err != nil {
		t.Fatal(err)
	}
	s, err := Open(dir)
	if err != nil {
		t.Fatalf("an unreadable users.json stopped the store opening: %v", err)
	}
	if n := len(s.AllUsers()); n != 0 {
		t.Errorf("%d users from a corrupt file", n)
	}
	// One bad line is one bad line: the good one survives, and ids continue after it.
	rows := s.RecentAudit(10, "", "")
	if len(rows) != 1 || rows[0].ID != 7 {
		t.Fatalf("audit after a bad line: %+v", rows)
	}
	if e := s.AppendAudit(model.AuditEntry{Action: "login", Actor: "pin"}); e.ID != 8 {
		t.Errorf("next audit id %d, want 8", e.ID)
	}
}

func TestAuditIsNewestFirstAndFilters(t *testing.T) {
	s := openTemp(t)
	for _, e := range []model.AuditEntry{
		{Action: "login", Actor: "admin"},
		{Action: "login_failed", Actor: "Viewer1"},
		{Action: "user.delete", Actor: "admin", Detail: "role admin->viewer"},
		{Action: "login", Actor: "viewer2"},
	} {
		s.AppendAudit(e)
	}
	all := s.RecentAudit(100, "", "")
	if len(all) != 4 || all[0].ID != 4 || all[3].ID != 1 {
		t.Fatalf("not newest first: %+v", all)
	}
	if got := s.RecentAudit(2, "", ""); len(got) != 2 || got[0].ID != 4 {
		t.Errorf("limit: %+v", got)
	}
	if got := s.RecentAudit(100, "login", ""); len(got) != 2 {
		t.Errorf("action is an exact match: %+v", got)
	}
	if got := s.RecentAudit(100, "", "VIEWER"); len(got) != 2 {
		t.Errorf("actor is a case-insensitive substring: %+v", got)
	}
	// The filter applies BEFORE the limit.
	if got := s.RecentAudit(1, "user.delete", ""); len(got) != 1 || got[0].ID != 3 {
		t.Errorf("filter then limit: %+v", got)
	}
}

// The file is appended to one line per entry, without Go's HTML escaping ("->" must not become
// "->"), and trimmed to the last AuditMaxEntries — which the reopen then honours.
func TestAuditFileFormatAndCap(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	s.AppendAudit(model.AuditEntry{Action: "user.update", Actor: "admin",
		TargetType: "user", TargetID: "2", Detail: "petrov: role admin->viewer"})
	raw, err := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "admin->viewer") {
		t.Errorf("the detail was HTML-escaped on disk: %s", raw)
	}
	for _, field := range []string{`"id"`, `"action"`, `"actor"`, `"target_type"`,
		`"target_id":"2"`, `"detail"`, `"at"`} {
		if !strings.Contains(string(raw), field) {
			t.Errorf("audit line lacks %s: %s", field, raw)
		}
	}

	for i := 0; i < AuditMaxEntries+5; i++ {
		s.AppendAudit(model.AuditEntry{Action: "login", Actor: "pin"})
	}
	rows := s.RecentAudit(AuditMaxEntries*2, "", "")
	if len(rows) != AuditMaxEntries {
		t.Fatalf("kept %d entries, want %d", len(rows), AuditMaxEntries)
	}
	reopened, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	again := reopened.RecentAudit(AuditMaxEntries*2, "", "")
	if len(again) != AuditMaxEntries || again[0].ID != rows[0].ID {
		t.Errorf("after reopen: %d entries, newest id %d (want %d, %d)",
			len(again), again[0].ID, AuditMaxEntries, rows[0].ID)
	}
}

// A failed audit write must not fail the action: AppendAudit still returns the entry.
func TestAuditWriteFailureIsSwallowed(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	// A directory where the file should be makes every append fail.
	if err := os.Mkdir(filepath.Join(dir, "audit.jsonl"), 0o755); err != nil {
		t.Fatal(err)
	}
	e := s.AppendAudit(model.AuditEntry{Action: "login", Actor: "pin"})
	if e.ID != 1 {
		t.Errorf("entry id %d", e.ID)
	}
	if got := s.RecentAudit(10, "", ""); len(got) != 1 {
		t.Errorf("the in-memory log lost the entry: %+v", got)
	}
}
