package auth

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
)

func cfg() Config {
	return Config{Pin: "1234", JwtSecret: "test-secret", JwtAlgorithm: "HS256",
		JwtExpireMinutes: 60}
}

func TestTokenRoundTrip(t *testing.T) {
	token, err := CreateAccessToken(cfg(), "operator")
	if err != nil {
		t.Fatal(err)
	}
	claims, err := DecodeAccessToken(cfg(), token)
	if err != nil {
		t.Fatal(err)
	}
	if claims.Sub != "operator" {
		t.Fatalf("sub = %q", claims.Sub)
	}
	if claims.Exp == 0 {
		t.Error("no expiry was set; a session token without one never ends")
	}
}

// A token signed with a different secret must be rejected. This is the whole point of the
// signature, and a decoder that parses the claims before verifying would accept it.
func TestTokenFromAnotherSecretIsRejected(t *testing.T) {
	other := cfg()
	other.JwtSecret = "attacker-secret"
	token, err := CreateAccessToken(other, "operator")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeAccessToken(cfg(), token); err == nil {
		t.Fatal("a token signed with the wrong secret was accepted")
	}
}

// The classic JWT attack: re-encode the claims and leave the old signature. It must fail,
// which it does because the signature covers header AND claims.
func TestTamperedClaimsAreRejected(t *testing.T) {
	token, err := CreateAccessToken(cfg(), "operator")
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(token, ".")
	forged, err := json.Marshal(Claims{Sub: "admin", Exp: 1 << 40})
	if err != nil {
		t.Fatal(err)
	}
	parts[1] = base64.RawURLEncoding.EncodeToString(forged)
	if _, err := DecodeAccessToken(cfg(), strings.Join(parts, ".")); err == nil {
		t.Fatal("claims were swapped without invalidating the signature")
	}
}

func TestExpiredTokenIsRejected(t *testing.T) {
	expired := cfg()
	expired.JwtExpireMinutes = -1 // already past
	token, err := CreateAccessToken(expired, "operator")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeAccessToken(cfg(), token); err == nil {
		t.Fatal("an expired token was accepted")
	}
}

func TestMalformedTokensAreRejectedNotPanicking(t *testing.T) {
	for _, bad := range []string{"", "a", "a.b", "a.b.c.d", "...", "!!.!!.!!"} {
		if _, err := DecodeAccessToken(cfg(), bad); err == nil {
			t.Errorf("%q was accepted", bad)
		}
	}
}

// An unsupported algorithm is REFUSED rather than silently downgraded: a caller who
// configured RS256 and got HS256 would believe they had asymmetric signing.
func TestUnsupportedAlgorithmIsRefused(t *testing.T) {
	rs := cfg()
	rs.JwtAlgorithm = "RS256"
	if _, err := CreateAccessToken(rs, "operator"); err == nil {
		t.Fatal("RS256 was accepted; it must be refused, not downgraded")
	}
}

func TestVerifyPin(t *testing.T) {
	if !VerifyPin(cfg(), "1234") {
		t.Error("the correct PIN was rejected")
	}
	for _, bad := range []string{"", "123", "12345", "1235"} {
		if VerifyPin(cfg(), bad) {
			t.Errorf("PIN %q was accepted", bad)
		}
	}
}

// The prefix makes a key greppable in logs and recognisable when pasted somewhere it should
// not be, and the hash is what the store keeps.
func TestGeneratedKeyShape(t *testing.T) {
	key, err := GenerateApiKey()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(key, KeyPrefix) {
		t.Fatalf("key %q lacks the %q prefix", key, KeyPrefix)
	}
	if len(Prefix(key)) != KeyPrefixDisplayLen {
		t.Errorf("display prefix %q is not %d chars", Prefix(key), KeyPrefixDisplayLen)
	}
	if HashApiKey(key) == key {
		t.Error("the hash equals the key")
	}
	if len(HashApiKey(key)) != 64 {
		t.Errorf("hash is %d chars, want 64 (sha256 hex)", len(HashApiKey(key)))
	}

	second, err := GenerateApiKey()
	if err != nil {
		t.Fatal(err)
	}
	if second == key {
		t.Fatal("two generated keys were identical")
	}
}

// A short key must not panic in Prefix. Reachable: DEFAULT_API_KEY is operator-supplied and
// nothing requires it to look like a generated one.
func TestPrefixHandlesShortKeys(t *testing.T) {
	if got := Prefix("abc"); got != "abc" {
		t.Fatalf("Prefix(short) = %q", got)
	}
}
