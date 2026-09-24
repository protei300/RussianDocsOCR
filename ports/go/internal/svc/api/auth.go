package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/passwords"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/repo"
)

// Sign-in for the browser UI: a shared PIN, or a named account.
//
// Which one is live is decided by AUTH_MODE and resolved in auth.ResolveMode. Nothing here reads
// the raw setting — it asks for the effective mode, so a misconfiguration that was downgraded to
// PIN behaves consistently everywhere instead of half-working.
//
// GET /auth/config is the only endpoint here that answers before anyone has authenticated, and
// it is what makes one frontend serve both modes: the login page asks what to render rather
// than guessing.
//
// Security notes on this file specifically:
//
//   - Failed sign-ins are THROTTLED per (identity, address) and recorded in the audit log. The
//     submitted PIN or password is never logged — writing rejected credentials to disk is its
//     own small leak, and rejected ones are often a typo away from the real one.
//   - The reply to a bad username and to a bad password is identical, including its timing
//     (repo.Authenticate verifies against a decoy hash when the account does not exist).
//     Distinguishing them hands over a list of valid usernames.
//   - A successful sign-in for an account that still owes a password change returns a
//     RESTRICTED token: real, but refused by every guard except the password change. The
//     response says so with must_change_password so the UI can route straight to the change
//     form instead of bouncing off a 403.
//
// Port of service/api/auth.py.

// Request-body limits, from the reference's pydantic Field declarations. Counted in CHARACTERS
// (runes), as pydantic counts them — a byte limit would cut a Cyrillic display name at half the
// length a Latin one gets.
const (
	maxPinLen         = 32
	maxUsernameLen    = 64
	maxPasswordLen    = 256
	maxDisplayNameLen = 128
	// maxAuthBodyBytes bounds what these endpoints read at all. Generous against the limits
	// above (256 four-byte runes twice over), and far below anything that costs memory.
	maxAuthBodyBytes = 64 << 10
)

// decodeBody reads one JSON object into dst.
//
// **422 for a body that does not fit the schema**, as the reference answers (FastAPI's own
// validation status). Its `detail` is a plain string rather than pydantic's list of objects:
// the SPA shows `detail` as text, and the contract accepts 400 or 422 for these without
// inspecting the shape. See DEVIATIONS D-14.
func decodeBody(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxAuthBodyBytes)
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		return statusError(http.StatusUnprocessableEntity, "Request body must be a JSON object "+
			"with the documented fields")
	}
	return nil
}

// requireText checks a required string field's presence and length, in characters.
func requireText(field string, value *string, maxLen int) (string, error) {
	if value == nil {
		return "", statusError(http.StatusUnprocessableEntity, fmt.Sprintf("%s: field required", field))
	}
	if n := utf8.RuneCountInString(*value); n < 1 || n > maxLen {
		return "", statusError(http.StatusUnprocessableEntity,
			fmt.Sprintf("%s: must be between 1 and %d characters", field, maxLen))
	}
	return *value, nil
}

// optionalText checks an optional string field's length, in characters.
func optionalText(field string, value *string, maxLen int) error {
	if value != nil && utf8.RuneCountInString(*value) > maxLen {
		return statusError(http.StatusUnprocessableEntity,
			fmt.Sprintf("%s: must be at most %d characters", field, maxLen))
	}
	return nil
}

// tooManyAttempts is the 429 with its Retry-After, in whole seconds.
func tooManyAttempts(seconds int) error {
	return &httpError{status: http.StatusTooManyRequests,
		detail:  fmt.Sprintf("Too many attempts. Try again in %d s", seconds),
		headers: map[string]string{"Retry-After": strconv.Itoa(seconds)}}
}

// asUserError turns a repository rule violation into a 400 carrying its message, and passes
// anything else — a store write that failed — through to the 500 mapping.
func asUserError(err error) error {
	var rule *repo.UserError
	if errors.As(err, &rule) {
		return statusError(http.StatusBadRequest, rule.Error())
	}
	return err
}

// accountToken mints the token for a named account. tv is what kills stale sessions — see the
// gate in deps.go.
func (s *Server) accountToken(user *model.User) (string, error) {
	uid, tv := user.ID, user.TokenVersion
	return auth.CreateAccessToken(s.authCfg(), auth.Claims{
		Sub: user.Username, Name: user.Name(), Role: user.Role, Uid: &uid, Tv: &tv,
	})
}

// demoCredentials returns the seeded credentials ONLY while publishing them is harmless.
//
// Two conditions, and both are necessary.
//
// *The configured password is the built-in demo one.* The first Python version returned
// ADMIN_PASSWORD unconditionally — so an operator who set it to a real secret had it printed on
// the login page for every anonymous visitor. A value from the environment is a secret by
// default; only the documented demo value is not.
//
// *The seeded account still owes its password change.* After that, "admin/1234" is false, and
// advertising a credential that does not work is noise at best and a hint about the account
// name at worst.
func (s *Server) demoCredentials() map[string]string {
	if s.cfg.AdminPassword != config.DefaultAdminPassword {
		return nil
	}
	seeded := repo.FindUser(s.db, s.cfg.AdminUsername)
	if seeded == nil || !seeded.MustChangePassword {
		return nil
	}
	return map[string]string{"username": seeded.Username, "password": config.DefaultAdminPassword}
}

// handleAuthConfig is what the login page needs before anyone has authenticated.
//
// Deliberately reachable without a token, and deliberately says nothing that is not already
// visible: the mode, the password rules, and — only in the demo default — the seeded
// credentials, which are printed on the page anyway.
func (s *Server) handleAuthConfig(w http.ResponseWriter, r *http.Request) {
	payload := map[string]any{
		"mode":         s.authMode,
		"pin_required": s.authMode == auth.PinMode,
		// Named so the UI can hide user management without inferring it from the mode
		// string; the two are the same today and need not stay that way.
		"users_enabled":    s.authMode == auth.UsersMode,
		"downgrade_reason": s.downgradeReason,
	}
	if s.authMode == auth.UsersMode {
		payload["password_rules"] = passwords.RulesForUI()
		if demo := s.demoCredentials(); demo != nil {
			payload["demo_credentials"] = demo
		}
	}
	writeJSON(w, http.StatusOK, payload)
}

// handlePinLogin exchanges the PIN for a session JWT.
func (s *Server) handlePinLogin(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Pin *string `json:"pin"`
	}
	if err := decodeBody(w, r, &body); err != nil {
		writeError(w, err)
		return
	}
	pin, err := requireText("pin", body.Pin, maxPinLen)
	if err != nil {
		writeError(w, err)
		return
	}
	if s.authMode == auth.UsersMode {
		// Not 401: the credential is not wrong, the endpoint is not in service.
		writeError(w, statusError(http.StatusConflict, "This service is configured for named "+
			"accounts; sign in with a username and password"))
		return
	}
	client := clientAddress(r)
	if blocked := s.throttle.BlockedFor("pin", client); blocked > 0 {
		writeError(w, tooManyAttempts(blocked))
		return
	}

	if !auth.VerifyPin(s.authCfg(), pin) {
		// Logged without the attempted value — writing rejected PINs to disk would be its own
		// small credential leak.
		s.throttle.NoteFailure("pin", client)
		repo.Audit(s.db, "login_failed", "pin", "", "", "")
		slog.Warn("[API] rejected PIN sign-in attempt")
		writeError(w, statusError(http.StatusUnauthorized, "Wrong PIN"))
		return
	}

	s.throttle.Clear("pin", client)
	repo.Audit(s.db, "login", "pin", "", "", "")
	operator := pinIdentity()
	token, err := auth.CreateAccessToken(s.authCfg(), auth.Claims{
		Sub: "operator", Name: operator.Name, Role: operator.Role})
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"access_token": token,
		"token_type":   "bearer",
		"user":         map[string]any{"name": operator.Name, "role": operator.Role},
	})
}

// handleLogin signs a named account in.
func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Username *string `json:"username"`
		Password *string `json:"password"`
	}
	if err := decodeBody(w, r, &body); err != nil {
		writeError(w, err)
		return
	}
	username, err := requireText("username", body.Username, maxUsernameLen)
	if err != nil {
		writeError(w, err)
		return
	}
	password, err := requireText("password", body.Password, maxPasswordLen)
	if err != nil {
		writeError(w, err)
		return
	}
	if s.authMode != auth.UsersMode {
		writeError(w, statusError(http.StatusConflict, "This service is configured for PIN sign-in"))
		return
	}
	client := clientAddress(r)
	if blocked := s.throttle.BlockedFor(username, client); blocked > 0 {
		writeError(w, tooManyAttempts(blocked))
		return
	}

	user, err := repo.Authenticate(s.db, username, password)
	if err != nil {
		writeError(w, err)
		return
	}
	if user == nil {
		s.throttle.NoteFailure(username, client)
		// The username is recorded because it is what makes the log useful when someone is
		// walking a list of accounts. The password never is. The client address is
		// deliberately NOT recorded: it is personal data, and the audit log's rule is that
		// nothing personal goes in — the reference once wrote it on every sign-in. The
		// throttle still uses the address, in memory, for the life of the process.
		repo.Audit(s.db, "login_failed", strings.TrimSpace(username), "", "", "")
		slog.Warn("[API] rejected sign-in for " + auth.PyRepr(username))
		writeError(w, statusError(http.StatusUnauthorized, "Wrong username or password"))
		return
	}

	s.throttle.Clear(username, client)
	repo.Audit(s.db, "login", user.Username, "", "", "")
	token, err := s.accountToken(user)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"access_token":         token,
		"token_type":           "bearer",
		"user":                 user.Public(),
		"must_change_password": user.MustChangePassword,
	})
}

// handleMe says who the current token belongs to.
//
// Behind the PERMISSIVE guard so the change-password screen can show who is signed in while the
// session is still restricted. Fields the identity does not have are null — the pointer fields
// of Identity encode that directly.
func (s *Server) handleMe(w http.ResponseWriter, r *http.Request, id *Identity) {
	writeJSON(w, http.StatusOK, map[string]any{
		"mode": s.authMode,
		"user": map[string]any{
			"username":             id.Username,
			"name":                 id.Name,
			"role":                 id.Role,
			"user_id":              id.UserID,
			"must_change_password": id.MustChangePassword,
		},
	})
}

// handleChangePassword changes your own password, ending every session you have — this one too.
//
// The current password is required even though the caller is already authenticated: a token
// left open on an unattended machine should not be enough to take an account over permanently.
func (s *Server) handleChangePassword(w http.ResponseWriter, r *http.Request, id *Identity) {
	var body struct {
		CurrentPassword *string `json:"current_password"`
		NewPassword     *string `json:"new_password"`
	}
	if err := decodeBody(w, r, &body); err != nil {
		writeError(w, err)
		return
	}
	current, err := requireText("current_password", body.CurrentPassword, maxPasswordLen)
	if err != nil {
		writeError(w, err)
		return
	}
	next, err := requireText("new_password", body.NewPassword, maxPasswordLen)
	if err != nil {
		writeError(w, err)
		return
	}
	if s.authMode != auth.UsersMode {
		writeError(w, statusError(http.StatusConflict, "There are no accounts in PIN mode"))
		return
	}
	var user *model.User
	if id.UserID != nil {
		user = repo.GetUser(s.db, *id.UserID)
	}
	if user == nil {
		writeError(w, statusError(http.StatusUnauthorized, "Session no longer valid"))
		return
	}

	if _, err := repo.ChangePassword(s.db, user, next, &current); err != nil {
		writeError(w, asUserError(err))
		return
	}
	repo.Audit(s.db, "password.change", user.Username, "user", strconv.Itoa(user.ID), "")
	// No fresh token on purpose: the version bump has just invalidated this one, and handing
	// back a new one would quietly defeat the point of asking the user to sign in again with
	// the password they have just chosen.
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "reauthenticate": true})
}
