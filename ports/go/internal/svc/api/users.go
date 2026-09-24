package api

import (
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/passwords"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/repo"
)

// User management — administrators only, and only in AUTH_MODE=users.
//
// **In PIN mode every route here answers 404, not an empty list.** "There are no users" and
// "users are not a concept in this configuration" are different answers, and a UI that cannot
// tell them apart shows an empty management page nobody can make work. The frontend asks
// /auth/config and hides the section entirely; the 404 is the backstop for anyone calling the
// API directly.
//
// Every mutation is audited, with the acting administrator as the actor and the affected
// account as the target — an id and a username, never a password.
//
// Order inside each handler, following the reference: the guard (before anything is read), then
// the body, then the mode, then the account. FastAPI validates the body while resolving the
// route's dependencies, which is why a malformed body is a 422 even in PIN mode.
//
// Port of service/api/users.py.

func (s *Server) requireUsersMode() error {
	if s.authMode != auth.UsersMode {
		return statusError(http.StatusNotFound, "User accounts are disabled (AUTH_MODE=pin)")
	}
	return nil
}

func (s *Server) loadUser(id int) (*model.User, error) {
	user := repo.GetUser(s.db, id)
	if user == nil {
		return nil, statusError(http.StatusNotFound, "No such user")
	}
	return user, nil
}

func publicUsers(users []*model.User) []map[string]any {
	out := make([]map[string]any, 0, len(users))
	for _, u := range users {
		out = append(out, u.Public())
	}
	return out
}

func (s *Server) handleListUsers(w http.ResponseWriter, r *http.Request, _ *Identity) {
	if err := s.requireUsersMode(); err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"items":          publicUsers(repo.AllUsers(s.db)),
		"roles":          model.Roles,
		"password_rules": passwords.RulesForUI(),
	})
}

func (s *Server) handleCreateUser(w http.ResponseWriter, r *http.Request, admin *Identity) {
	var body struct {
		Username    *string `json:"username"`
		Password    *string `json:"password"`
		Role        *string `json:"role"`
		DisplayName *string `json:"display_name"`
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
	if err := optionalText("display_name", body.DisplayName, maxDisplayNameLen); err != nil {
		writeError(w, err)
		return
	}
	role, displayName := model.RoleViewer, ""
	if body.Role != nil {
		role = *body.Role
	}
	if body.DisplayName != nil {
		displayName = *body.DisplayName
	}
	if err := s.requireUsersMode(); err != nil {
		writeError(w, err)
		return
	}

	user, err := repo.CreateUser(s.db, username, password, role, displayName, true)
	if err != nil {
		writeError(w, asUserError(err))
		return
	}
	repo.Audit(s.db, "user.create", adminActor(admin), "user", strconv.Itoa(user.ID),
		fmt.Sprintf("%s as %s", user.Username, user.Role))
	writeJSON(w, http.StatusCreated, user.Public())
}

func (s *Server) handleUpdateUser(w http.ResponseWriter, r *http.Request, admin *Identity, userID int) {
	// Absent and null both mean "unchanged", as in the reference's `str | None = None`.
	var body struct {
		Role        *string `json:"role"`
		DisplayName *string `json:"display_name"`
		IsActive    *bool   `json:"is_active"`
	}
	if err := decodeBody(w, r, &body); err != nil {
		writeError(w, err)
		return
	}
	if err := optionalText("display_name", body.DisplayName, maxDisplayNameLen); err != nil {
		writeError(w, err)
		return
	}
	if err := s.requireUsersMode(); err != nil {
		writeError(w, err)
		return
	}
	user, err := s.loadUser(userID)
	if err != nil {
		writeError(w, err)
		return
	}
	wasRole, wasActive := user.Role, user.IsActive

	user, err = repo.UpdateUser(s.db, user, body.Role, body.DisplayName, body.IsActive)
	if err != nil {
		writeError(w, asUserError(err))
		return
	}

	var changes []string
	if wasRole != user.Role {
		changes = append(changes, fmt.Sprintf("role %s->%s", wasRole, user.Role))
	}
	if wasActive != user.IsActive {
		if user.IsActive {
			changes = append(changes, "activated")
		} else {
			changes = append(changes, "deactivated")
		}
	}
	summary := strings.Join(changes, ", ")
	if summary == "" {
		summary = "profile"
	}
	repo.Audit(s.db, "user.update", adminActor(admin), "user", strconv.Itoa(user.ID),
		user.Username+": "+summary)
	writeJSON(w, http.StatusOK, user.Public())
}

// handleResetPassword sets someone else's password. They must change it at their next sign-in.
func (s *Server) handleResetPassword(w http.ResponseWriter, r *http.Request, admin *Identity, userID int) {
	var body struct {
		NewPassword *string `json:"new_password"`
	}
	if err := decodeBody(w, r, &body); err != nil {
		writeError(w, err)
		return
	}
	newPassword, err := requireText("new_password", body.NewPassword, maxPasswordLen)
	if err != nil {
		writeError(w, err)
		return
	}
	if err := s.requireUsersMode(); err != nil {
		writeError(w, err)
		return
	}
	user, err := s.loadUser(userID)
	if err != nil {
		writeError(w, err)
		return
	}
	user, err = repo.ResetPassword(s.db, user, newPassword)
	if err != nil {
		writeError(w, asUserError(err))
		return
	}
	repo.Audit(s.db, "user.password_reset", adminActor(admin), "user", strconv.Itoa(user.ID),
		user.Username)
	writeJSON(w, http.StatusOK, user.Public())
}

// handleDeleteUser answers 204 with an empty body, like every DELETE here.
func (s *Server) handleDeleteUser(w http.ResponseWriter, r *http.Request, admin *Identity, userID int) {
	if err := s.requireUsersMode(); err != nil {
		writeError(w, err)
		return
	}
	user, err := s.loadUser(userID)
	if err != nil {
		writeError(w, err)
		return
	}
	if err := repo.DeleteUser(s.db, user, admin.UserID); err != nil {
		writeError(w, asUserError(err))
		return
	}
	repo.Audit(s.db, "user.delete", adminActor(admin), "user", strconv.Itoa(userID), user.Username)
	writeNoContent(w)
}

// handleListAudit serves the action log. Administrator only — it names who did what.
//
// Available in BOTH modes: in PIN mode the actor is the literal "pin", and knowing that a
// sign-in failed at a given moment is still worth more than nothing. Only the user-management
// half of this file is mode-gated.
//
// `limit` is CLAMPED to 1..1000 rather than rejected, as the reference does — it declares a
// bare `limit: int = 200` and clamps in the body. Only a non-integer is a 422, in pydantic's
// shape, via queryInt with no bounds.
func (s *Server) handleListAudit(w http.ResponseWriter, r *http.Request, _ *Identity) {
	q := r.URL.Query()
	limit, err := queryInt(q, "limit", 200, math.MinInt, 0)
	if err != nil {
		writeError(w, err)
		return
	}
	limit = max(1, min(limit, 1000))
	entries := repo.RecentAudit(s.db, limit, q.Get("action"), q.Get("actor"))
	writeJSON(w, http.StatusOK, map[string]any{"items": entries, "count": len(entries)})
}
