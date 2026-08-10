// Package auth holds two authentication paths, for two different callers.
//
//   - **The website** signs in with a PIN and gets a short-lived JWT. One shared operator
//     identity; there are no user accounts.
//   - **Machine callers** send an API key in `X-API-Key`. Keys are managed from the UI at
//     runtime, plus one bootstrap key from the environment.
//
// Why the split: a PIN is a human affordance and a terrible service credential — four
// digits, shared, and it would have to be embedded in every integration. An API key is the
// opposite. Endpoints both kinds of caller use accept either.
//
// Security notes, honestly:
//
//   - Comparison is constant-time. For the PIN that is mostly symbolic against a
//     four-digit space: there is no rate limiting or lockout here, and a PIN is not a
//     defence against an attacker who can reach the port. It keeps honest people out of the
//     browser UI; the NETWORK BOUNDARY is the real control.
//   - Only key HASHES are stored. A leaked data directory must not yield working
//     credentials.
//
// Port of service/core/auth.py. The JWT is hand-rolled rather than taken from a dependency
// — HS256 with two base64url segments and an HMAC is about forty lines, and it keeps the
// port's dependency list at three modules, which matters for a reference project somebody
// has to audit.
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

// Claims is the JWT payload. Only what is actually used.
type Claims struct {
	Sub string `json:"sub"`
	Exp int64  `json:"exp"`
}

var errBadToken = errors.New("auth: invalid token")

// CreateAccessToken signs a JWT valid for the configured window.
func CreateAccessToken(cfg Config, subject string) (string, error) {
	if cfg.JwtAlgorithm != "" && cfg.JwtAlgorithm != "HS256" {
		// Refused rather than silently downgraded: a caller who configured RS256 and got
		// HS256 would believe they had asymmetric signing.
		return "", fmt.Errorf("auth: unsupported JWT algorithm %q (only HS256)", cfg.JwtAlgorithm)
	}
	header := map[string]string{"alg": "HS256", "typ": "JWT"}
	claims := Claims{
		Sub: subject,
		Exp: time.Now().UTC().Add(time.Duration(cfg.JwtExpireMinutes) * time.Minute).Unix(),
	}
	h, err := json.Marshal(header)
	if err != nil {
		return "", err
	}
	c, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	signing := b64(h) + "." + b64(c)
	return signing + "." + b64(sign(signing, cfg.JwtSecret)), nil
}

// DecodeAccessToken returns the claims, or an error for anything invalid or expired.
//
// The signature is verified BEFORE the claims are parsed, and with a constant-time compare.
// Parsing first would mean acting on attacker-controlled JSON; a plain `==` on the MAC leaks
// how much of it matched.
func DecodeAccessToken(cfg Config, token string) (*Claims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, errBadToken
	}
	signing := parts[0] + "." + parts[1]
	want := sign(signing, cfg.JwtSecret)
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
		return nil, errBadToken
	}
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
