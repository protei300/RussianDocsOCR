package repo

import (
	"regexp"
	"strings"
	"sync"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/passwords"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/store"
)

// User accounts: the whole lifecycle, and the rules that keep it usable.
//
// Only reachable when AUTH_MODE=users. In PIN mode nothing here is called and the store stays
// empty — the API answers 404 rather than returning an empty list.
//
// **The rules worth stating before the code, because each is the kind that only bites in
// production.**
//
// *You cannot lock everyone out.* Deleting, deactivating or demoting the last active
// administrator is refused. It sounds like an edge case until someone demotes themselves to
// "viewer" to test the role and discovers there is no longer an account that can undo it.
//
// *That check is only a check if it is atomic with the change.* Two administrators demoting
// each other at the same moment each see the other still active, both checks pass, and the
// service ends with none. So every mutation below runs under ONE lock and re-reads the account
// inside it: check and change are a single step, not two with a gap between them. The first
// Python version had the gap — and handed out live objects from the store, so the gap was also
// a window in which one request could see another's half-applied edit.
//
// *Changing anything that affects authority bumps the token version.* Password, role, active
// flag — each one invalidates every token already issued for that account. Skipping it on a
// role change is the subtle case: a demoted administrator otherwise keeps administrator rights
// for the rest of the token's eight-hour life, and nothing in the UI hints at it.
//
// *Usernames are ASCII.* A name is an identity, and Unicode lets two different identities look
// the same: "admin" and "аdmin" (Cyrillic а) pass a case-insensitive uniqueness check and are
// indistinguishable on screen. Display names can be anything — they are for reading, not for
// trusting.
//
// Port of service/repositories/users.py. Signatures follow the repository contract (store
// first, plain functions), so a SQL-backed implementation is a body swap; there the lock becomes
// a transaction — SELECT … FOR UPDATE on the rows involved, or a serialisable transaction around
// the admin count.

// userWrite is the one lock for every change to an account.
//
// **Go has no re-entrant mutex, and this code does not need one.** Python's is an RLock "because
// a mutation may call a helper that also takes it"; here every exported mutation takes it
// exactly ONCE, at the top of its critical section, and the helpers it calls beneath —
// requireNotLastAdmin, countActiveAdmins, freshUser — never take it. That is a real constraint
// on edits to this file: a helper that locks again deadlocks instead of nesting. It mirrors the
// FileStore's own rule, and the .NET/Kotlin ports can use a plain lock the same way.
//
// Package state, like the reference's module-level _WRITE: the invariant it protects ("at least
// one active admin") is a property of the one store the process has.
//
// Logins do not hold it while hashing — see Authenticate — so a slow Argon2 verification never
// blocks administration.
var userWrite sync.Mutex

// Test hooks. nil in production; set only by this package's tests, which is why they are
// unexported. Each one marks the exact point a test needs to interleave another goroutine, so a
// race is forced rather than hoped for — a plain two-goroutine test passes with the lock
// removed, as the first Python test proved.
var (
	// hookAdminCount runs inside countActiveAdmins, after the account list has been read and
	// before it is counted.
	hookAdminCount func()
	// hookAfterVerify runs in Authenticate after the (slow, unlocked) password verification and
	// before the write-back.
	hookAfterVerify func()
)

// usernamePattern: letters, digits, dot, underscore, hyphen; starting with a letter or digit;
// at most 64. ASCII only — see the note above.
var usernamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

// UserError is a rule violation. The message is written to be shown: "Cannot demote the last
// active administrator" is the whole explanation, and swallowing it into a generic 400 would
// leave the operator guessing which rule they hit.
type UserError struct{ msg string }

func (e *UserError) Error() string { return e.msg }

func userError(msg string) error { return &UserError{msg: msg} }

func AllUsers(db store.DocumentStore) []*model.User { return db.AllUsers() }

func GetUser(db store.DocumentStore, id int) *model.User { return db.GetUser(id) }

func FindUser(db store.DocumentStore, username string) *model.User { return db.FindUser(username) }

// countActiveAdmins counts active administrators other than `excluding`. Caller holds userWrite.
func countActiveAdmins(db store.DocumentStore, excluding int) int {
	users := db.AllUsers()
	if hookAdminCount != nil {
		hookAdminCount()
	}
	n := 0
	for _, u := range users {
		if u.Role == model.RoleAdmin && u.IsActive && u.ID != excluding {
			n++
		}
	}
	return n
}

// requireNotLastAdmin refuses the change when `user` is the last active administrator: it is
// an admin, it is active, and no OTHER account is both. Caller holds userWrite.
func requireNotLastAdmin(db store.DocumentStore, user *model.User, what string) error {
	if user.Role == model.RoleAdmin && user.IsActive && countActiveAdmins(db, user.ID) == 0 {
		return userError("Cannot " + what + " the last active administrator")
	}
	return nil
}

func normaliseUsername(username string) (string, error) {
	name := strings.TrimSpace(username)
	if name == "" {
		return "", userError("Username is required")
	}
	if !usernamePattern.MatchString(name) {
		return "", userError("Username may contain only Latin letters, digits, '.', '_' and '-', " +
			"must start with a letter or digit, and be at most 64 characters")
	}
	return name, nil
}

func checkRole(role string) error {
	if !model.IsRole(role) {
		return userError("Unknown role " + auth.PyRepr(role))
	}
	return nil
}

// freshUser re-reads inside the lock. The copy the caller holds may already be stale. Caller
// holds userWrite.
func freshUser(db store.DocumentStore, id int) (*model.User, error) {
	user := db.GetUser(id)
	if user == nil {
		return nil, userError("No such user")
	}
	return user, nil
}

// CreateUser validates, hashes, and stores a new account.
func CreateUser(db store.DocumentStore, username, password, role, displayName string,
	mustChangePassword bool) (*model.User, error) {

	name, err := normaliseUsername(username)
	if err != nil {
		return nil, err
	}
	if err := checkRole(role); err != nil {
		return nil, err
	}
	if complaint := passwords.Validate(password); complaint != "" {
		return nil, userError(complaint)
	}
	// Hashed BEFORE taking the lock: it is the slow part, and it depends on nothing the lock
	// protects.
	hash, err := passwords.Hash(password)
	if err != nil {
		return nil, err
	}

	userWrite.Lock()
	defer userWrite.Unlock()
	// Uniqueness and id allocation are checked and used in one step, or two concurrent
	// creations of "petrov" would both succeed.
	if db.FindUser(name) != nil {
		return nil, userError("User " + auth.PyRepr(name) + " already exists")
	}
	user := model.NewUser(db.NextUserID(), name, role, hash,
		strings.TrimSpace(displayName), mustChangePassword)
	return db.PutUser(user)
}

// SeedAdmin creates the bootstrap administrator when there are NO users at all. Returns nil
// when users already exist.
//
// The one place that bypasses the password policy, and it has to: the default password is
// printed on the login page precisely so the demo can be used, and it fails every composition
// rule. MustChangePassword is what makes that defensible — the account can do nothing else
// until it is changed.
func SeedAdmin(db store.DocumentStore, username, password string) (*model.User, error) {
	name, err := normaliseUsername(username)
	if err != nil {
		return nil, err
	}
	hash, err := passwords.Hash(password)
	if err != nil {
		return nil, err
	}
	userWrite.Lock()
	defer userWrite.Unlock()
	if len(db.AllUsers()) > 0 {
		return nil, nil
	}
	user := model.NewUser(db.NextUserID(), name, model.RoleAdmin, hash, "Administrator", true)
	return db.PutUser(user)
}

// --- the timing decoy ------------------------------------------------------------

// A real Argon2 hash of a value nobody knows, used only to spend the same time verifying a
// password for a username that does not exist. Computed ONCE: hashing on every failed login
// would itself be a timing signal. Python computes it at import; here PrepareTimingDecoy is
// called from main at startup in users mode, and the sync.Once makes a call from anywhere else
// harmless.
var (
	decoyOnce sync.Once
	decoyHash string
)

// PrepareTimingDecoy computes the decoy hash now rather than on the first unknown-user login,
// whose extra ~80 ms would otherwise be exactly the signal the decoy exists to hide.
func PrepareTimingDecoy() { timingDecoy() }

func timingDecoy() string {
	decoyOnce.Do(func() {
		// On the (practically impossible) failure of the RNG the decoy stays "", Verify fails
		// fast on it, and that one login's timing differs — degraded, not broken.
		decoyHash, _ = passwords.Hash("no-such-user-timing-decoy")
	})
	return decoyHash
}

// Authenticate verifies credentials. nil for wrong user, wrong password, or disabled account.
//
// One answer for all three on purpose: an endpoint that distinguished "no such user" from
// "wrong password" would let anyone enumerate accounts. The password is still verified for a
// missing user — against the decoy — so the response time does not reveal which case it was.
//
// The verification runs OUTSIDE the write lock (it is the slow part); only the bookkeeping
// afterwards takes it, and it RE-READS the account first. Writing back the copy loaded before
// hashing would silently undo anything an administrator changed during those ~80 ms — a
// demotion, say, and with it the token-version bump that revoked the old sessions.
func Authenticate(db store.DocumentStore, username, password string) (*model.User, error) {
	user := db.FindUser(username)
	if user == nil {
		passwords.Verify(timingDecoy(), password)
		return nil, nil
	}
	if !passwords.Verify(user.PasswordHash, password) {
		return nil, nil
	}
	var rehash string
	if passwords.NeedsRehash(user.PasswordHash) {
		h, err := passwords.Hash(password)
		if err != nil {
			return nil, err
		}
		rehash = h
	}
	if hookAfterVerify != nil {
		hookAfterVerify()
	}

	userWrite.Lock()
	defer userWrite.Unlock()
	fresh := db.GetUser(user.ID)
	// Re-checked on the FRESH copy: the account may have been disabled, deleted or had its
	// password changed while this request was busy hashing.
	if fresh == nil || !fresh.IsActive || fresh.PasswordHash != user.PasswordHash {
		return nil, nil
	}
	if rehash != "" {
		fresh.PasswordHash = rehash
	}
	fresh.LastLoginAt = model.At(model.StampNow())
	return db.PutUser(fresh)
}

// ChangePassword sets a new password for the account itself. Bumps the token version, ending
// every session the account has — including the one asking.
//
// currentPassword nil skips the current-password check; the endpoint always passes it, because
// a token left open on an unattended machine should not be enough to take an account over.
func ChangePassword(db store.DocumentStore, user *model.User, newPassword string,
	currentPassword *string) (*model.User, error) {

	if complaint := passwords.Validate(newPassword); complaint != "" {
		return nil, userError(complaint)
	}
	newHash, err := passwords.Hash(newPassword)
	if err != nil {
		return nil, err
	}

	userWrite.Lock()
	defer userWrite.Unlock()
	fresh, err := freshUser(db, user.ID)
	if err != nil {
		return nil, err
	}
	if currentPassword != nil && !passwords.Verify(fresh.PasswordHash, *currentPassword) {
		return nil, userError("Current password is incorrect")
	}
	if passwords.Verify(fresh.PasswordHash, newPassword) {
		return nil, userError("The new password must differ from the current one")
	}
	fresh.PasswordHash = newHash
	fresh.MustChangePassword = false
	fresh.TokenVersion++
	return db.PutUser(fresh)
}

// ResetPassword is an administrator setting someone else's password; they must change it at
// their next sign-in.
func ResetPassword(db store.DocumentStore, user *model.User, newPassword string) (*model.User, error) {
	if complaint := passwords.Validate(newPassword); complaint != "" {
		return nil, userError(complaint)
	}
	newHash, err := passwords.Hash(newPassword)
	if err != nil {
		return nil, err
	}
	userWrite.Lock()
	defer userWrite.Unlock()
	fresh, err := freshUser(db, user.ID)
	if err != nil {
		return nil, err
	}
	fresh.PasswordHash = newHash
	fresh.MustChangePassword = true
	fresh.TokenVersion++
	return db.PutUser(fresh)
}

// UpdateUser changes role, display name or active flag; nil means unchanged. Any authority
// change invalidates the account's tokens; a display-name change does not.
func UpdateUser(db store.DocumentStore, user *model.User, role, displayName *string,
	isActive *bool) (*model.User, error) {

	if role != nil {
		if err := checkRole(*role); err != nil {
			return nil, err
		}
	}

	userWrite.Lock()
	defer userWrite.Unlock()
	fresh, err := freshUser(db, user.ID)
	if err != nil {
		return nil, err
	}
	authorityChanged := false

	if role != nil && *role != fresh.Role {
		if *role != model.RoleAdmin {
			if err := requireNotLastAdmin(db, fresh, "demote"); err != nil {
				return nil, err
			}
		}
		fresh.Role = *role
		authorityChanged = true
	}
	if isActive != nil && *isActive != fresh.IsActive {
		if !*isActive {
			// Checked against `fresh` as already modified above, exactly as the reference
			// does: a request that both demotes and deactivates the last admin is refused by
			// the demotion check first.
			if err := requireNotLastAdmin(db, fresh, "deactivate"); err != nil {
				return nil, err
			}
		}
		fresh.IsActive = *isActive
		authorityChanged = true
	}
	if displayName != nil {
		fresh.DisplayName = strings.TrimSpace(*displayName)
	}
	if authorityChanged {
		fresh.TokenVersion++
	}
	return db.PutUser(fresh)
}

// DeleteUser removes an account. actingUserID is the administrator doing it, nil when unknown.
func DeleteUser(db store.DocumentStore, user *model.User, actingUserID *int) error {
	if actingUserID != nil && user.ID == *actingUserID {
		// Not a safety rule so much as a usability one: there is no undo, and deleting the
		// account you are signed in as is never what was meant. Checked first, before the lock.
		return userError("You cannot delete your own account")
	}
	userWrite.Lock()
	defer userWrite.Unlock()
	fresh, err := freshUser(db, user.ID)
	if err != nil {
		return err
	}
	if err := requireNotLastAdmin(db, fresh, "delete"); err != nil {
		return err
	}
	_, err = db.DropUser(fresh.ID)
	return err
}
