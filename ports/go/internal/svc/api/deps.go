package api

import (
	"fmt"
	"net"
	"net/http"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/model"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/repo"
)

// The single gate: authentication and authorisation for every route.
//
// Two kinds of caller share one API, and two authentication modes share one gate. Everything
// below exists so that ONE FUNCTION DECIDES, because the alternative — each handler checking for
// itself — grows a hole the day someone adds a route and forgets a line, and that hole is
// silent.
//
//	require_session_allow_password_change
//	                      Any session, INCLUDING one that still owes a password change. Used
//	                      by exactly two routes: /auth/me and /auth/change-password.
//	require_role(min)     A session, no pending password change, role >= min. Guards service
//	                      management — users, API keys, settings, logs, status, purge — so an
//	                      API key never reaches it.
//	require_api_or_role(min)
//	                      An API key at any level, OR a session as require_role(min). Guards
//	                      the document routes, which serve the bundled UI and integrations.
//
// **Fail-safe by shape, and that shape is deliberate.** An account flagged must_change_password
// gets a real token, but every guard refuses it except the one named after what it permits — so
// a route added later with an ordinary guard blocks the restricted session rather than serving
// it. You cannot use the permissive guard by accident.
//
// **The session is re-checked against the store on every request.** A JWT is valid until it
// expires — eight hours here — so a disabled account, a changed password or a demoted role
// would otherwise keep their old authority for the rest of that window. Each request loads the
// user and compares token_version; anything that changes authority bumps it and every issued
// token dies at once. The cost is one map lookup in an in-memory index.
//
// Why not one scheme for machine and human callers: a PIN or a password is a human affordance
// and a poor service credential — personal or shared, and it would have to be embedded in every
// integration. An API key is the opposite. Conflating them forces one into the wrong shape.
//
// Port of service/api/deps.py.

// Identity is who is calling.
//
// The optional fields are POINTERS because /auth/me reports each one as null when the identity
// does not have it — a PIN session has no username, no user id and no pending password change,
// and "0" or "" would be a different, wrong answer.
type Identity struct {
	Kind  string // "session" | "api_key"
	Name  string
	Role  string // viewer | operator | admin, or "service" for an API key
	KeyID int

	UserID             *int
	Username           *string
	MustChangePassword *bool
}

// Identity kinds.
const (
	kindSession = "session"
	kindApiKey  = "api_key"
)

// pinIdentity is the PIN session. There are no accounts in that mode — the PIN authenticates
// "whoever is at the console", nothing finer — so it is granted the top role and every guard
// below is satisfied, which is what keeps PIN deployments behaving exactly as before.
func pinIdentity() *Identity {
	return &Identity{Kind: kindSession, Name: "Operator", Role: model.RoleAdmin}
}

// passwordChangeRequired is the machine-readable reason on the 403 that means "change your
// password first". The UI routes on this string, not on prose — do not reword it.
const passwordChangeRequired = "password_change_required"

// The 401 texts. Fixed by the contract (ports/AUTH.md §5).
const (
	signInRequired     = "Sign in to use this endpoint"
	apiKeyOrSignInText = "Provide an API key in X-API-Key, or sign in"
)

// unauthorised is a 401 carrying WWW-Authenticate, which is what makes the status mean "you may
// retry with credentials" rather than "go away". Every 401 a guard produces goes through here.
func unauthorised(detail string) error {
	return &httpError{status: http.StatusUnauthorized, detail: detail,
		headers: map[string]string{"WWW-Authenticate": "Bearer"}}
}

func forbidden(detail string) error {
	return &httpError{status: http.StatusForbidden, detail: detail}
}

// bearerToken extracts the token from an Authorization header.
//
// Case-insensitive on the scheme, because clients disagree about "Bearer" versus "bearer" and
// rejecting one of them is a support ticket, not a security measure.
func bearerToken(r *http.Request) string {
	header := r.Header.Get("Authorization")
	if len(header) < 7 || !strings.EqualFold(header[:7], "bearer ") {
		return ""
	}
	return strings.TrimSpace(header[7:])
}

// sessionIdentity turns a bearer token into an identity, or nil if it is not usable. This is
// the one function that decides what a token is worth (ports/AUTH.md §5).
func (s *Server) sessionIdentity(r *http.Request) *Identity {
	token := bearerToken(r)
	if token == "" {
		return nil
	}
	claims, err := auth.DecodeAccessToken(s.authCfg(), token)
	if err != nil {
		return nil
	}

	if s.authMode != auth.UsersMode {
		// PIN mode accepts only PIN tokens. A token minted while the service ran with named
		// accounts carries a uid, and it must be REFUSED, not merely have its uid ignored:
		// every PIN session is the administrator, so "ignoring" the uid would promote a
		// viewer's still-valid token to full control the moment the service is switched back
		// to PIN. The reference's earlier version did exactly that while its comment claimed
		// the opposite.
		if claims.UidPresent {
			return nil
		}
		return pinIdentity()
	}

	if claims.Uid == nil {
		return nil // a PIN-era token, worthless in users mode
	}
	user := repo.GetUser(s.db, *claims.Uid)
	if user == nil || !user.IsActive {
		return nil
	}
	if claims.Tv == nil || *claims.Tv != user.TokenVersion {
		// Password changed, role changed or the account was disabled since this token was
		// issued. This is the whole reason token_version exists.
		return nil
	}
	// Everything below comes from the STORE, not from the token: the role in the claims is
	// what the account had when the token was minted, which is exactly what must not count.
	id, username, mustChange := user.ID, user.Username, user.MustChangePassword
	return &Identity{
		Kind: kindSession, Name: user.Name(), Role: user.Role,
		UserID: &id, Username: &username, MustChangePassword: &mustChange,
	}
}

// optionalIdentity identifies a caller on a best-effort basis, returning nil for anonymous.
//
// The session is checked FIRST because it is cheap — an HMAC and a map lookup — while the API
// key path hashes and then scans every stored key.
func (s *Server) optionalIdentity(r *http.Request) *Identity {
	if id := s.sessionIdentity(r); id != nil {
		return id
	}
	if presented := r.Header.Get("X-API-Key"); presented != "" {
		key, err := repo.VerifyApiKey(s.db, s.authCfg(), presented)
		if err == nil && key != nil {
			repo.TouchApiKey(s.db, key)
			return &Identity{Kind: kindApiKey, Name: key.Label, Role: "service", KeyID: key.ID}
		}
	}
	return nil
}

func rejectIfPasswordChangePending(id *Identity) error {
	if id.MustChangePassword != nil && *id.MustChangePassword {
		return forbidden(passwordChangeRequired)
	}
	return nil
}

func requireRoleOf(id *Identity, minimum string) error {
	if !model.RoleAtLeast(id.Role, minimum) {
		return forbidden(fmt.Sprintf("This action requires the %s role", minimum))
	}
	return nil
}

// Guard is a named authentication requirement.
//
// A struct with a Name rather than a bare function, so the route-table test can read WHICH guard
// each route declares and compare it with ports/AUTH.md §5 — not merely whether a guard is
// present. The first Python version's test asked only the latter, and stayed green while a
// viewer could upload, reprocess, delete and purge.
type Guard struct {
	Name  string
	check func(s *Server, r *http.Request) (*Identity, error)
}

// Guard names, as the reference names them.
const (
	guardPublic               = "public"
	guardSessionAllowPwChange = "require_session_allow_password_change"
	guardViewer               = "require_viewer"
	guardOperator             = "require_operator"
	guardAdmin                = "require_admin"
	guardApiOrViewer          = "require_api_or_viewer"
	guardApiOrOperator        = "require_api_or_operator"
)

// requireSessionAllowPasswordChange admits any valid session, INCLUDING a restricted one. Used by
// exactly two routes; everything else must use a role guard.
var requireSessionAllowPasswordChange = &Guard{
	Name: guardSessionAllowPwChange,
	check: func(s *Server, r *http.Request) (*Identity, error) {
		id := s.sessionIdentity(r)
		if id == nil {
			return nil, unauthorised(signInRequired)
		}
		return id, nil
	},
}

// requireRole builds the guard for a session whose role is `minimum` or higher.
//
// In PIN mode every session is the single operator identity with the top role, so these guards
// are satisfied and the service behaves exactly as it did before named accounts existed. That is
// what keeps the change backward compatible rather than merely additive.
func requireRole(minimum string) *Guard {
	return &Guard{
		Name: "require_" + minimum,
		check: func(s *Server, r *http.Request) (*Identity, error) {
			id, err := requireSessionAllowPasswordChange.check(s, r)
			if err != nil {
				return nil, err
			}
			if err := rejectIfPasswordChangePending(id); err != nil {
				return nil, err
			}
			if err := requireRoleOf(id, minimum); err != nil {
				return nil, err
			}
			return id, nil
		},
	}
}

// requireApiOrRole builds the guard for an API key, or a session whose role is `minimum` or
// higher.
//
// An API key is admitted at any level because its scope is the document API and nothing else —
// it cannot reach users, keys, settings, logs or status, which all use requireRole and therefore
// refuse API keys outright.
func requireApiOrRole(minimum string) *Guard {
	return &Guard{
		Name: "require_api_or_" + minimum,
		check: func(s *Server, r *http.Request) (*Identity, error) {
			id := s.optionalIdentity(r)
			if id == nil {
				return nil, unauthorised(apiKeyOrSignInText)
			}
			if id.Kind == kindApiKey {
				return id, nil
			}
			if err := rejectIfPasswordChangePending(id); err != nil {
				return nil, err
			}
			if err := requireRoleOf(id, minimum); err != nil {
				return nil, err
			}
			return id, nil
		},
	}
}

var (
	requireViewer        = requireRole(model.RoleViewer)
	requireOperator      = requireRole(model.RoleOperator)
	requireAdmin         = requireRole(model.RoleAdmin)
	requireApiOrViewer   = requireApiOrRole(model.RoleViewer)
	requireApiOrOperator = requireApiOrRole(model.RoleOperator)
)

// guard wraps a handler with an authentication requirement.
//
// A wrapper rather than a check inside each handler: the check is then IMPOSSIBLE TO FORGET at
// the routing table, where it is also visible — the property FastAPI's Depends provides, and the
// reason the routes read as a permission list.
//
// **The guard runs before the body is read.** Nothing of the request beyond its headers is
// touched until the caller is admitted, so a viewer's upload is a 403, not a 422 about the file
// — and an anonymous caller cannot make the service parse a 20 MB multipart body at all.
func (s *Server) guard(g *Guard, h func(http.ResponseWriter, *http.Request, *Identity)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := g.check(s, r)
		if err != nil {
			writeError(w, err)
			return
		}
		h(w, r, id)
	}
}

// clientAddress is the throttle's notion of "where from": the TCP peer's host.
//
// Never X-Forwarded-For. It is a request header, so any client can set it to anything, and a
// throttle keyed on it is a throttle the attacker chooses the key for. Behind a reverse proxy
// every request then shares the proxy's address — the safe failure: stricter, not bypassable.
func clientAddress(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	if host == "" {
		return "-"
	}
	return host
}

// adminActor is the audit actor for a user-management action: the acting administrator's
// username. "?" is the reference's placeholder should one ever be missing — it cannot be in
// users mode, the only mode these routes serve.
func adminActor(id *Identity) string {
	if id != nil && id.Username != nil {
		return *id.Username
	}
	return "?"
}

func (s *Server) authCfg() auth.Config {
	return auth.Config{
		Pin:              s.cfg.AuthPin,
		JwtSecret:        s.cfg.JwtSecret,
		JwtAlgorithm:     s.cfg.JwtAlgorithm,
		JwtExpireMinutes: s.cfg.JwtExpireMinutes,
		DefaultApiKey:    s.cfg.DefaultApiKey,
	}
}
