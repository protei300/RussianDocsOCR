package api

import (
	"net/http"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/errs"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/repo"
)

// Authentication comes in three levels, because two kinds of caller share one API.
//
//	requireSession        Browser only, backed by the PIN-issued JWT. Guards anything that
//	                      manages the SERVICE — API keys, settings, logs — because those are
//	                      operator concerns and an integration has no business touching them.
//	requireApiOrSession   Either a valid X-API-Key or a valid session JWT. Guards the WORKING
//	                      endpoints, so the same routes serve the bundled UI and third-party
//	                      integrations without duplicating them.
//	optionalIdentity      Never rejects. For endpoints that vary by caller but must stay
//	                      reachable.
//
// Why not one scheme for both: a four-digit PIN is a human affordance and a poor service
// credential — shared, guessable, and it would have to be embedded in every integration. An
// API key is the opposite. Conflating them forces one of the two into the wrong shape.

// Identity is who is calling. There are no user accounts: the PIN authenticates "whoever is
// at the console", nothing finer.
type Identity struct {
	Kind  string // "session" | "api_key"
	Name  string
	Role  string
	KeyID int
}

// sessionUser is the single operator identity.
var sessionUser = Identity{Kind: "session", Name: "Operator", Role: "admin"}

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

// optionalIdentity identifies a caller on a best-effort basis, returning nil for anonymous.
//
// The session is checked FIRST because it is cheap — an HMAC over the token — while the API
// key path hashes and then scans every stored key.
func (s *Server) optionalIdentity(r *http.Request) *Identity {
	if token := bearerToken(r); token != "" {
		if _, err := auth.DecodeAccessToken(s.authCfg(), token); err == nil {
			id := sessionUser
			return &id
		}
	}

	if presented := r.Header.Get("X-API-Key"); presented != "" {
		key, err := repo.VerifyApiKey(s.db, s.authCfg(), presented)
		if err == nil && key != nil {
			repo.TouchApiKey(s.db, key)
			return &Identity{Kind: "api_key", Name: key.Label, Role: "service", KeyID: key.ID}
		}
	}
	return nil
}

// requireSession admits browser sessions only.
func (s *Server) requireSession(r *http.Request) (*Identity, error) {
	token := bearerToken(r)
	if token == "" {
		return nil, clientError(errs.ErrUnauthorized, "Sign in with the PIN to use this endpoint")
	}
	if _, err := auth.DecodeAccessToken(s.authCfg(), token); err != nil {
		return nil, clientError(errs.ErrUnauthorized, "Sign in with the PIN to use this endpoint")
	}
	id := sessionUser
	return &id, nil
}

// requireApiOrSession admits either kind of caller.
func (s *Server) requireApiOrSession(r *http.Request) (*Identity, error) {
	if id := s.optionalIdentity(r); id != nil {
		return id, nil
	}
	return nil, clientError(errs.ErrUnauthorized, "Provide an API key in X-API-Key, or sign in with the PIN")
}

// guard wraps a handler with an authentication requirement.
//
// A wrapper rather than a check inside each handler: the check is then IMPOSSIBLE TO FORGET
// at the routing table, where it is also visible — which is the property FastAPI's Depends
// provides and the reason the routes read as a permission list.
//
// The WWW-Authenticate header accompanies every 401, because that is what makes the status
// code mean "you may retry with credentials" rather than "go away".
func (s *Server) guard(require func(*http.Request) (*Identity, error),
	h func(http.ResponseWriter, *http.Request, *Identity)) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {
		id, err := require(r)
		if err != nil {
			w.Header().Set("WWW-Authenticate", "Bearer")
			writeError(w, err)
			return
		}
		h(w, r, id)
	}
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
