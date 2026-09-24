// Package auth holds the authentication primitives, for two kinds of caller.
//
//   - **The website** signs in — with the shared PIN (AUTH_MODE=pin, the default) or with a
//     named account (AUTH_MODE=users) — and gets a short-lived JWT.
//   - **Machine callers** send an API key in `X-API-Key`. Keys are managed from the UI at
//     runtime, plus one bootstrap key from the environment.
//
// Why the split: a PIN or a password is a human affordance and a terrible service credential —
// short, shared or personal, and it would have to be embedded in every integration. An API key
// is the opposite. Endpoints both kinds of caller use accept either.
//
// Security notes, honestly:
//
//   - Comparison is constant-time. For the PIN that is mostly symbolic against a four-digit
//     space; what actually limits guessing is the failed-login throttle in throttle.go, and
//     the NETWORK BOUNDARY remains the real control.
//   - Only key HASHES are stored. A leaked data directory must not yield working
//     credentials. Passwords are hashed too, but slowly — see svc/passwords for why that is a
//     different algorithm.
//
// Which mode is in force is decided in mode.go, once. The gate that turns a token into an
// identity is svc/api/deps.go.
//
// Port of service/core/auth.py. The JWT is hand-rolled rather than taken from a dependency —
// HS256 with two base64url segments and an HMAC is about forty lines, and it keeps the port's
// dependency list short, which matters for a reference project somebody has to audit.
package auth

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// KeyPrefix makes keys greppable in logs and recognisable when pasted somewhere they should
// not be — the same reason GitHub uses `ghp_`.
const (
	KeyPrefix           = "rdk_"
	KeyPrefixDisplayLen = 10 // 'rdk_' + 6 chars, enough to tell keys apart
)

// Config is what auth needs from the environment tier.
type Config struct {
	Pin              string
	JwtSecret        string
	JwtAlgorithm     string
	JwtExpireMinutes int
	DefaultApiKey    string
}

// DefaultJwtSecret is the value shipped in config.Defaults. Named here so the check below names
// it once; config's literal is pinned to it by a test.
const DefaultJwtSecret = "changeme-in-production"

// Claims is the JWT payload, in both of the shapes the service issues:
//
//	           PIN token      account token
//	sub        "operator"     username
//	name       "Operator"     display name, or username if empty
//	role       "admin"        the account's role
//	uid        ABSENT         integer user id
//	tv         ABSENT         integer token_version
//	exp        seconds        seconds
//
// Uid and Tv are POINTERS because absent and zero are different answers (CONVENTIONS §2): a PIN
// token has no uid at all, and "uid 0" would be an account that does not exist. The gate needs
// the difference in both directions — PIN mode refuses any token that carries a uid, users mode
// refuses any that does not.
type Claims struct {
	Sub  string `json:"sub"`
	Name string `json:"name,omitempty"`
	Role string `json:"role,omitempty"`
	Uid  *int   `json:"uid,omitempty"`
	Tv   *int   `json:"tv,omitempty"`
	Exp  int64  `json:"exp"`

	// UidPresent records that the payload carried a "uid" key AT ALL — including `"uid": null`,
	// which decodes to a nil Uid and would otherwise look exactly like a PIN token. The
	// reference tests `"uid" in claims`; this is that test. Never serialised.
	UidPresent bool `json:"-"`
}

var errBadToken = errors.New("auth: invalid token")

// --- the signing secret -------------------------------------------------------
//
// **A known secret is not a secret.** The default in config is public — it is in this
// repository — so with it anyone can mint a token. In users mode that is a full takeover: the
// administrator is uid 1 and token_version starts at 1, so a forged {"uid": 1, "tv": 1} is a
// guess, not an attack.
//
// So when the secret is unset or still the default, a random one is generated for the life of
// the process and the default NEVER signs anything. The only cost is that sessions do not
// survive a restart — and on this service nothing does: the store is wiped at every start, so a
// session outliving it would point at an account that no longer exists anyway.
var (
	processSecretOnce sync.Once
	processSecret     string
	processSecretErr  error
)

// SecretIsEphemeral reports whether the configured secret is unusable, i.e. whether the
// per-process random one is in force. main logs the consequence at startup.
func SecretIsEphemeral(cfg Config) bool {
	configured := strings.TrimSpace(cfg.JwtSecret)
	return configured == "" || configured == DefaultJwtSecret
}

// SigningSecret is the secret actually in use: the configured one (trimmed, as the reference
// uses it), or 48 random bytes generated once per process.
func SigningSecret(cfg Config) (string, error) {
	if !SecretIsEphemeral(cfg) {
		return strings.TrimSpace(cfg.JwtSecret), nil
	}
	processSecretOnce.Do(func() {
		buf := make([]byte, 48)
		if _, err := rand.Read(buf); err != nil {
			processSecretErr = fmt.Errorf("auth: generate jwt secret: %w", err)
			return
		}
		processSecret = base64.RawURLEncoding.EncodeToString(buf)
	})
	return processSecret, processSecretErr
}

// CreateAccessToken signs a JWT for these claims, valid for the configured window. Exp is set
// here; whatever the caller put in it is overwritten.
func CreateAccessToken(cfg Config, claims Claims) (string, error) {
	if cfg.JwtAlgorithm != "" && cfg.JwtAlgorithm != "HS256" {
		// Refused rather than silently downgraded: a caller who configured RS256 and got
		// HS256 would believe they had asymmetric signing.
		return "", fmt.Errorf("auth: unsupported JWT algorithm %q (only HS256)", cfg.JwtAlgorithm)
	}
	secret, err := SigningSecret(cfg)
	if err != nil {
		return "", err
	}
	header := map[string]string{"alg": "HS256", "typ": "JWT"}
	claims.Exp = time.Now().UTC().Add(time.Duration(cfg.JwtExpireMinutes) * time.Minute).Unix()
	h, err := json.Marshal(header)
	if err != nil {
		return "", err
	}
	c, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	signing := b64(h) + "." + b64(c)
	return signing + "." + b64(sign(signing, secret)), nil
}

// DecodeAccessToken returns the claims, or an error for anything invalid or expired.
//
// Three checks, in this order, and the order is the point:
//
//  1. **The algorithm is pinned.** A header that does not say exactly HS256 is refused before
//     anything else is looked at — so alg=none, or an asymmetric algorithm "verified" with our
//     own secret as its public key, is refused rather than negotiated. The header is the only
//     attacker-controlled JSON parsed before the signature, and all that is read from it is
//     one string compared for equality.
//  2. The signature is verified BEFORE the claims are parsed, with a constant-time compare.
//     Parsing first would mean acting on attacker-controlled JSON; a plain `==` on the MAC
//     leaks how much of it matched.
//  3. Expiry.
func DecodeAccessToken(cfg Config, token string) (*Claims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, errBadToken
	}

	rawHeader, err := unb64(parts[0])
	if err != nil {
		return nil, errBadToken
	}
	var header struct {
		Alg string `json:"alg"`
	}
	if err := json.Unmarshal(rawHeader, &header); err != nil || header.Alg != "HS256" {
		return nil, fmt.Errorf("%w: algorithm not HS256", errBadToken)
	}

	secret, err := SigningSecret(cfg)
	if err != nil {
		return nil, err
	}
	signing := parts[0] + "." + parts[1]
	want := sign(signing, secret)
	got, err := unb64(parts[2])
	if err != nil {
		return nil, errBadToken
	}
	if !hmac.Equal(want, got) {
		return nil, errBadToken
	}

	raw, err := unb64(parts[1])
	if err != nil {
		return nil, errBadToken
	}
	var claims Claims
	if err := json.Unmarshal(raw, &claims); err != nil {
		// Also where a non-integer uid or tv ends up: "uid": "1" or 1.5 does not decode into
		// *int, and a token whose uid is not an integer is no session (ports/AUTH.md §5).
		return nil, errBadToken
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil {
		return nil, errBadToken
	}
	_, claims.UidPresent = keys["uid"]
	if claims.Exp != 0 && time.Now().UTC().Unix() >= claims.Exp {
		return nil, fmt.Errorf("%w: expired", errBadToken)
	}
	return &claims, nil
}

func sign(signing, secret string) []byte {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(signing))
	return mac.Sum(nil)
}

func b64(data []byte) string         { return base64.RawURLEncoding.EncodeToString(data) }
func unb64(s string) ([]byte, error) { return base64.RawURLEncoding.DecodeString(s) }

// VerifyPin compares in constant time. See the package note on what that is and is not
// worth for a four-digit secret.
func VerifyPin(cfg Config, candidate string) bool {
	return subtle.ConstantTimeCompare([]byte(candidate), []byte(cfg.Pin)) == 1
}

// GenerateApiKey mints a fresh key, shown to the user exactly once.
func GenerateApiKey() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("auth: generate key: %w", err)
	}
	return KeyPrefix + base64.RawURLEncoding.EncodeToString(buf), nil
}

func HashApiKey(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

func Prefix(key string) string {
	if len(key) < KeyPrefixDisplayLen {
		return key
	}
	return key[:KeyPrefixDisplayLen]
}

// --- the bootstrap key ------------------------------------------------------
//
// Resolved once per process. Two cases:
//
//	DEFAULT_API_KEY set    -> use it. Stable across restarts, so integrations keep
//	                          working. Treated as a secret the operator already holds, so
//	                          the UI shows it masked.
//	DEFAULT_API_KEY unset  -> generate a random one and log it. Nobody could know it
//	                          otherwise, so the UI DOES reveal it in full. That is the
//	                          deliberate trade, and it only happens when no explicit key
//	                          was configured.
//
// The alternative — a constant fallback in the source — would give every unconfigured
// deployment the same publicly-known key. That is worse than either branch here.
var (
	defaultOnce      sync.Once
	defaultKey       string
	defaultGenerated bool
	defaultErr       error
)

// ResolveDefaultKey returns (key, wasGenerated). Idempotent; safe to call from anywhere.
func ResolveDefaultKey(cfg Config) (string, bool, error) {
	defaultOnce.Do(func() {
		if configured := strings.TrimSpace(cfg.DefaultApiKey); configured != "" {
			defaultKey, defaultGenerated = configured, false
			return
		}
		key, err := GenerateApiKey()
		if err != nil {
			defaultErr = err
			return
		}
		defaultKey, defaultGenerated = key, true
	})
	return defaultKey, defaultGenerated, defaultErr
}
