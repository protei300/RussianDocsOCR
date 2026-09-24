package repo

import (
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/passwords"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// The account rules that a black-box test cannot see: what happens when two requests overlap.
// Each mirrors a test in tests/service/test_auth_security.py.

const strong = "Str0ng-Pass"

// putAccount stores an account directly, skipping the 64 MiB hash where the test does not sign
// in with it.
func putAccount(t *testing.T, db store.DocumentStore, name, role, hash string) *model.User {
	t.Helper()
	u, err := db.PutUser(model.NewUser(db.NextUserID(), name, role, hash, "", false))
	if err != nil {
		t.Fatal(err)
	}
	return u
}

func ptr[T any](v T) *T { return &v }

func ruleMessage(err error) string {
	var rule *UserError
	if errors.As(err, &rule) {
		return rule.Error()
	}
	return ""
}

// TestConcurrentDemotionCannotRemoveBothAdmins forces the interleaving the lock exists for.
//
// Two administrators each demote the other. Without the lock, both requests read the account
// list while the other is still an admin, both "last admin" checks pass, and the service ends
// with no administrator at all. A plain two-goroutine race almost never hits that window — the
// first Python test was green with the lock removed — so the test holds it open: a barrier
// inside countActiveAdmins, AFTER the list is read, waits until BOTH goroutines are there.
//
// With the lock the second goroutine cannot enter the check while the first is inside it, so
// the barrier times out for the first, the first demotion completes, and the second then sees
// no other administrator and is refused. Without the lock both reach the barrier, both proceed
// on stale counts, and the assertion below fails.
func TestConcurrentDemotionCannotRemoveBothAdmins(t *testing.T) {
	db := newStore(t)
	a := putAccount(t, db, "admin1", model.RoleAdmin, "x")
	b := putAccount(t, db, "admin2", model.RoleAdmin, "x")

	var mu sync.Mutex
	arrived := 0
	both := make(chan struct{})
	hookAdminCount = func() {
		mu.Lock()
		arrived++
		if arrived == 2 {
			close(both)
		}
		mu.Unlock()
		select {
		case <-both:
		case <-time.After(750 * time.Millisecond):
		}
	}
	t.Cleanup(func() { hookAdminCount = nil })

	errs := make([]error, 2)
	var wg sync.WaitGroup
	for i, target := range []*model.User{a, b} {
		wg.Add(1)
		go func(i int, target *model.User) {
			defer wg.Done()
			_, errs[i] = UpdateUser(db, target, ptr(model.RoleViewer), nil, nil)
		}(i, target)
	}
	wg.Wait()

	admins := 0
	for _, u := range db.AllUsers() {
		if u.Role == model.RoleAdmin && u.IsActive {
			admins++
		}
	}
	if admins != 1 {
		t.Fatalf("%d active administrators after two concurrent demotions, want exactly 1 "+
			"(errors: %v, %v)", admins, errs[0], errs[1])
	}
	refused := 0
	for _, err := range errs {
		if ruleMessage(err) == "Cannot demote the last active administrator" {
			refused++
		} else if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
	if refused != 1 {
		t.Errorf("%d demotions refused, want exactly 1", refused)
	}
}

// TestSignInDoesNotUndoADemotion demotes the account in the window between the password
// verification (slow, unlocked) and the write-back of last_login_at.
//
// Writing back the copy loaded before hashing would restore the admin role AND the old
// token_version — silently undoing the demotion and resurrecting every session it revoked. The
// re-read inside the lock is what prevents it.
func TestSignInDoesNotUndoADemotion(t *testing.T) {
	db := newStore(t)
	hash, err := passwords.Hash(strong)
	if err != nil {
		t.Fatal(err)
	}
	target := putAccount(t, db, "admin1", model.RoleAdmin, hash)
	putAccount(t, db, "admin2", model.RoleAdmin, "x") // so the demotion itself is allowed

	demoted := false
	hookAfterVerify = func() {
		if _, err := UpdateUser(db, target, ptr(model.RoleViewer), nil, nil); err != nil {
			t.Errorf("demotion during sign-in: %v", err)
		}
		demoted = true
	}
	t.Cleanup(func() { hookAfterVerify = nil })

	signedIn, err := Authenticate(db, "admin1", strong)
	if err != nil {
		t.Fatal(err)
	}
	if !demoted {
		t.Fatal("the hook never ran; the test did not exercise the window")
	}
	stored := db.GetUser(target.ID)
	if stored.Role != model.RoleViewer || stored.TokenVersion != 2 {
		t.Fatalf("the sign-in undid the demotion: stored role %q, token_version %d",
			stored.Role, stored.TokenVersion)
	}
	if !stored.LastLoginAt.Set {
		t.Error("last_login_at was not recorded")
	}
	// And the sign-in itself reports the account as it is NOW, so the token it mints carries
	// the new token version and role.
	if signedIn == nil || signedIn.Role != model.RoleViewer || signedIn.TokenVersion != 2 {
		t.Errorf("sign-in returned a stale account: %+v", signedIn)
	}
}

// A password changed, or the account disabled, while the sign-in was hashing: refused.
func TestSignInRefusedIfTheAccountChangedWhileHashing(t *testing.T) {
	db := newStore(t)
	hash, err := passwords.Hash(strong)
	if err != nil {
		t.Fatal(err)
	}
	target := putAccount(t, db, "petrov", model.RoleViewer, hash)

	hookAfterVerify = func() {
		if _, err := UpdateUser(db, target, nil, nil, ptr(false)); err != nil {
			t.Errorf("deactivate: %v", err)
		}
	}
	t.Cleanup(func() { hookAfterVerify = nil })
	if u, _ := Authenticate(db, "petrov", strong); u != nil {
		t.Error("an account disabled mid-sign-in was signed in")
	}
}

func TestAuthenticateAnswersNilForEveryFailure(t *testing.T) {
	db := newStore(t)
	hash, err := passwords.Hash(strong)
	if err != nil {
		t.Fatal(err)
	}
	putAccount(t, db, "petrov", model.RoleViewer, hash)
	disabled := putAccount(t, db, "sidorov", model.RoleViewer, hash)
	disabled.IsActive = false
	if _, err := db.PutUser(disabled); err != nil {
		t.Fatal(err)
	}
	putAccount(t, db, "broken", model.RoleViewer, "$argon2id$v=19$garbage")

	for _, c := range []struct{ name, password string }{
		{"petrov", "wrong"}, {"nobody", strong}, {"sidorov", strong}, {"broken", strong},
	} {
		if u, err := Authenticate(db, c.name, c.password); u != nil || err != nil {
			t.Errorf("Authenticate(%q) = %v, %v; want nil, nil", c.name, u, err)
		}
	}
	if u, err := Authenticate(db, "  PETROV ", strong); u == nil || err != nil {
		t.Errorf("case-insensitive, trimmed sign-in failed: %v", err)
	}
}

// A hash with older parameters is upgraded at the one moment the plaintext is available.
func TestSignInRehashesOldParameters(t *testing.T) {
	db := newStore(t)
	old := "$argon2id$v=19$m=19456,t=2,p=1$rppcAWOP4qJuFb6Dc52G3g$NoDmjBcYZrj9DJvzNb421/YYxfGj1D+TxplD/tAEero"
	putAccount(t, db, "legacy", model.RoleViewer, old)
	if u, err := Authenticate(db, "legacy", "Пароль-42"); u == nil || err != nil {
		t.Fatalf("interop vector did not sign in: %v", err)
	}
	stored := db.FindUser("legacy")
	if !strings.HasPrefix(stored.PasswordHash, "$argon2id$v=19$m=65536,t=3,p=4$") {
		t.Errorf("hash not upgraded: %s", stored.PasswordHash)
	}
	if !passwords.Verify(stored.PasswordHash, "Пароль-42") {
		t.Error("the upgraded hash does not verify")
	}
}

func TestLastAdminRules(t *testing.T) {
	db := newStore(t)
	admin := putAccount(t, db, "admin", model.RoleAdmin, "x")
	viewer := putAccount(t, db, "viewer", model.RoleViewer, "x")

	if _, err := UpdateUser(db, admin, ptr(model.RoleOperator), nil, nil); ruleMessage(err) !=
		"Cannot demote the last active administrator" {
		t.Errorf("demote last admin: %v", err)
	}
	if _, err := UpdateUser(db, admin, nil, nil, ptr(false)); ruleMessage(err) !=
		"Cannot deactivate the last active administrator" {
		t.Errorf("deactivate last admin: %v", err)
	}
	if err := DeleteUser(db, admin, nil); ruleMessage(err) !=
		"Cannot delete the last active administrator" {
		t.Errorf("delete last admin: %v", err)
	}
	if err := DeleteUser(db, admin, ptr(admin.ID)); ruleMessage(err) !=
		"You cannot delete your own account" {
		t.Errorf("self-delete: %v", err)
	}
	// An INACTIVE second admin does not count.
	second := putAccount(t, db, "admin2", model.RoleAdmin, "x")
	second.IsActive = false
	if _, err := db.PutUser(second); err != nil {
		t.Fatal(err)
	}
	if _, err := UpdateUser(db, admin, ptr(model.RoleViewer), nil, nil); err == nil {
		t.Error("an inactive admin was counted as another administrator")
	}
	// Promoting the viewer makes the demotion legal.
	if _, err := UpdateUser(db, viewer, ptr(model.RoleAdmin), nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := UpdateUser(db, admin, ptr(model.RoleViewer), nil, nil); err != nil {
		t.Errorf("demotion with another active admin: %v", err)
	}
}

// Authority changes bump token_version; a display-name change does not.
func TestTokenVersionBumps(t *testing.T) {
	db := newStore(t)
	putAccount(t, db, "admin", model.RoleAdmin, "x")
	u := putAccount(t, db, "petrov", model.RoleViewer, "x")

	step := func(what string, role, name *string, active *bool, want int) {
		t.Helper()
		got, err := UpdateUser(db, u, role, name, active)
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if got.TokenVersion != want {
			t.Errorf("%s: token_version %d, want %d", what, got.TokenVersion, want)
		}
	}
	step("rename", nil, ptr("  Пётр  "), nil, 1)
	if got := db.GetUser(u.ID).DisplayName; got != "Пётр" {
		t.Errorf("display name not trimmed: %q", got)
	}
	step("same role", ptr(model.RoleViewer), nil, nil, 1)
	step("role change", ptr(model.RoleOperator), nil, nil, 2)
	step("deactivate", nil, nil, ptr(false), 3)
	step("reactivate", nil, nil, ptr(true), 4)

	reset, err := ResetPassword(db, u, "An0ther-Pass")
	if err != nil {
		t.Fatal(err)
	}
	if reset.TokenVersion != 5 || !reset.MustChangePassword {
		t.Errorf("reset: token_version %d, must_change %v", reset.TokenVersion, reset.MustChangePassword)
	}
	changed, err := ChangePassword(db, u, strong, ptr("An0ther-Pass"))
	if err != nil {
		t.Fatal(err)
	}
	if changed.TokenVersion != 6 || changed.MustChangePassword {
		t.Errorf("change: token_version %d, must_change %v",
			changed.TokenVersion, changed.MustChangePassword)
	}
	if _, err := ChangePassword(db, u, "Wr0ng-Guess", ptr("not-it")); ruleMessage(err) !=
		"Current password is incorrect" {
		t.Errorf("wrong current: %v", err)
	}
	if _, err := ChangePassword(db, u, strong, ptr(strong)); ruleMessage(err) !=
		"The new password must differ from the current one" {
		t.Errorf("same password: %v", err)
	}
}

func TestCreateUserValidation(t *testing.T) {
	db := newStore(t)
	putAccount(t, db, "admin", model.RoleAdmin, "x")
	cases := []struct{ name, password, role, want string }{
		{" ", strong, "viewer", "Username is required"},
		{"аdmin", strong, "viewer", "Username may contain only Latin letters"}, // Cyrillic а
		{".dot", strong, "viewer", "Username may contain only Latin letters"},
		{strings.Repeat("a", 65), strong, "viewer", "Username may contain only Latin letters"},
		{"petrov", strong, "root", "Unknown role 'root'"},
		{"petrov", strong, "o'b", `Unknown role "o'b"`},
		{"petrov", "short1A", "viewer", "Password needs: at least 8 characters"},
		{"ADMIN", strong, "viewer", "User 'ADMIN' already exists"},
	}
	for _, c := range cases {
		_, err := CreateUser(db, c.name, c.password, c.role, "", true)
		if !strings.HasPrefix(ruleMessage(err), c.want) {
			t.Errorf("CreateUser(%q, role %q): %v, want %q", c.name, c.role, err, c.want)
		}
	}
	u, err := CreateUser(db, "  petrov ", strong, "operator", "  Пётр ", true)
	if err != nil {
		t.Fatal(err)
	}
	if u.Username != "petrov" || u.DisplayName != "Пётр" || u.TokenVersion != 1 ||
		!u.MustChangePassword || !u.IsActive || u.ID != 2 {
		t.Errorf("created %+v", u)
	}
}

func TestSeedAdminOnlyIntoAnEmptyStore(t *testing.T) {
	db := newStore(t)
	// The seeded password bypasses the rules: "1234" fails every one of them.
	u, err := SeedAdmin(db, "admin", "1234")
	if err != nil || u == nil {
		t.Fatalf("seed: %v", err)
	}
	if u.Role != model.RoleAdmin || u.DisplayName != "Administrator" || !u.MustChangePassword {
		t.Errorf("seeded %+v", u)
	}
	again, err := SeedAdmin(db, "other", "1234")
	if err != nil || again != nil {
		t.Errorf("a second seed created %v (%v)", again, err)
	}
	if _, err := SeedAdmin(newStore(t), "bad name!", "1234"); err == nil {
		t.Error("an invalid seed username was accepted")
	}
}
