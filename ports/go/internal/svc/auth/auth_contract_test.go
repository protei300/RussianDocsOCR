package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/config"
)

// The token and mode rules of ports/AUTH.md §1 and §4.

// forge builds a token by hand, the way an attacker would: arbitrary header, arbitrary claims,
// an HMAC with whatever secret they believe is in use.
func forge(header, claims, secret string) string {
	enc := base64.RawURLEncoding
	signing := enc.EncodeToString([]byte(header)) + "." + enc.EncodeToString([]byte(claims))
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(signing))
	return signing + "." + enc.EncodeToString(mac.Sum(nil))
}

func future() string { return `,"exp":` + strconv.FormatInt(time.Now().Add(time.Hour).Unix(), 10) }

// config's literal default and this package's constant must be one value: the ephemeral-secret
// check compares against the constant, and a drift would let the shipped default sign tokens.
func TestDefaultSecretConstantMatchesConfig(t *testing.T) {
	if config.Defaults().JwtSecret != DefaultJwtSecret {
		t.Fatalf("config default %q != auth.DefaultJwtSecret %q",
			config.Defaults().JwtSecret, DefaultJwtSecret)
	}
}

// **The published default never signs anything.** With it (or with nothing) configured, a random
// per-process secret is used — so a token HMAC'd with the value from this repository is refused,
// and a token the service minted does not verify under the default.
func TestPublishedDefaultSecretSignsNothing(t *testing.T) {
	for _, configured := range []string{DefaultJwtSecret, "", "   ", "  " + DefaultJwtSecret + "\t"} {
		c := cfg()
		c.JwtSecret = configured
		if !SecretIsEphemeral(c) {
			t.Errorf("JWT_SECRET=%q was treated as a real secret", configured)
		}
		secret, err := SigningSecret(c)
		if err != nil {
			t.Fatal(err)
		}
		if secret == DefaultJwtSecret || strings.TrimSpace(secret) == "" {
			t.Fatalf("JWT_SECRET=%q signs with %q", configured, secret)
		}
		if len(secret) < 64 { // 48 random bytes, base64url
			t.Errorf("the process secret is only %d characters", len(secret))
		}

		forged := forge(`{"alg":"HS256","typ":"JWT"}`,
			`{"sub":"operator","role":"admin"`+future()+`}`, DefaultJwtSecret)
		if _, err := DecodeAccessToken(c, forged); err == nil {
			t.Errorf("JWT_SECRET=%q: a token signed with the published default was accepted",
				configured)
		}
	}
	// A real secret is used as configured (trimmed), so tokens survive a restart.
	c := cfg()
	c.JwtSecret = "  real-secret  "
	if SecretIsEphemeral(c) {
		t.Error("a real secret was treated as ephemeral")
	}
	if got, _ := SigningSecret(c); got != "real-secret" {
		t.Errorf("SigningSecret = %q, want the trimmed configured value", got)
	}
}

// The process secret is generated ONCE: two tokens minted in one process verify each other.
func TestProcessSecretIsStable(t *testing.T) {
	c := cfg()
	c.JwtSecret = ""
	token, err := CreateAccessToken(c, pinClaims())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeAccessToken(c, token); err != nil {
		t.Fatalf("a token minted with the process secret did not verify: %v", err)
	}
}

// **The algorithm is pinned.** A header that says anything but HS256 is refused — even when the
// signature is a perfectly good HS256 MAC over the token, which is exactly the confusion attack.
func TestAlgorithmIsPinned(t *testing.T) {
	claims := `{"sub":"operator","role":"admin"` + future() + `}`
	for _, header := range []string{
		`{"alg":"HS512","typ":"JWT"}`,
		`{"alg":"none","typ":"JWT"}`,
		`{"alg":"RS256","typ":"JWT"}`,
		`{"alg":"hs256","typ":"JWT"}`,
		`{"typ":"JWT"}`,
		`not json`,
	} {
		if _, err := DecodeAccessToken(cfg(), forge(header, claims, cfg().JwtSecret)); err == nil {
			t.Errorf("header %s was accepted", header)
		}
	}
	// alg=none with an empty signature, the classic form.
	enc := base64.RawURLEncoding
	none := enc.EncodeToString([]byte(`{"alg":"none"}`)) + "." + enc.EncodeToString([]byte(claims)) + "."
	if _, err := DecodeAccessToken(cfg(), none); err == nil {
		t.Error("an alg=none token was accepted")
	}
	// And the control: the same claims under an HS256 header verify.
	if _, err := DecodeAccessToken(cfg(), forge(`{"alg":"HS256","typ":"JWT"}`, claims,
		cfg().JwtSecret)); err != nil {
		t.Errorf("a well-formed HS256 token was refused: %v", err)
	}
}

// uid/tv must be distinguishable between ABSENT and ZERO, and `"uid": null` counts as present —
// the gate refuses it in PIN mode, as the reference's `"uid" in claims` does.
func TestUidPresenceIsDistinguished(t *testing.T) {
	secret := cfg().JwtSecret
	cases := []struct {
		claims  string
		present bool
		uid     *int
	}{
		{`{"sub":"operator"` + future() + `}`, false, nil},
		{`{"sub":"x","uid":0,"tv":0` + future() + `}`, true, new(int)},
		{`{"sub":"x","uid":null` + future() + `}`, true, nil},
	}
	for _, c := range cases {
		got, err := DecodeAccessToken(cfg(), forge(`{"alg":"HS256"}`, c.claims, secret))
		if err != nil {
			t.Fatalf("%s: %v", c.claims, err)
		}
		if got.UidPresent != c.present {
			t.Errorf("%s: UidPresent %v, want %v", c.claims, got.UidPresent, c.present)
		}
		if (got.Uid == nil) != (c.uid == nil) {
			t.Errorf("%s: Uid %v", c.claims, got.Uid)
		}
	}
	// A uid that is not an integer is not a session in any mode: refused at decode.
	for _, bad := range []string{`"1"`, `1.5`, `true`} {
		claims := `{"sub":"x","uid":` + bad + `,"tv":1` + future() + `}`
		if _, err := DecodeAccessToken(cfg(), forge(`{"alg":"HS256"}`, claims, secret)); err == nil {
			t.Errorf("uid %s was accepted", bad)
		}
	}
}

// Both claim shapes round-trip; a PIN token carries no uid/tv keys at all.
func TestClaimShapes(t *testing.T) {
	pin, err := CreateAccessToken(cfg(), pinClaims())
	if err != nil {
		t.Fatal(err)
	}
	body, _ := base64.RawURLEncoding.DecodeString(strings.Split(pin, ".")[1])
	if strings.Contains(string(body), `"uid"`) || strings.Contains(string(body), `"tv"`) {
		t.Errorf("a PIN token carries account claims: %s", body)
	}
	uid, tv := 7, 3
	acct, err := CreateAccessToken(cfg(), Claims{Sub: "petrov", Name: "Пётр", Role: "viewer",
		Uid: &uid, Tv: &tv})
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeAccessToken(cfg(), acct)
	if err != nil {
		t.Fatal(err)
	}
	if got.Uid == nil || *got.Uid != 7 || got.Tv == nil || *got.Tv != 3 ||
		got.Role != "viewer" || got.Name != "Пётр" || !got.UidPresent {
		t.Errorf("account claims did not round-trip: %+v", got)
	}
}

func TestResolveMode(t *testing.T) {
	cases := []struct {
		raw, backend, mode string
		reason             string // substring; "" means nil
	}{
		{"", "files", PinMode, ""},
		{"  ", "files", PinMode, ""},
		{"pin", "files", PinMode, ""},
		{" PIN ", "files", PinMode, ""},
		{"users", "files", UsersMode, ""},
		{" Users\t", "files", UsersMode, ""},
		{"bogus", "files", PinMode,
			"AUTH_MODE='bogus' is not one of pin, users — falling back to PIN authentication"},
		{"users", "sql", PinMode, "implemented for the temporary file store only"},
	}
	for _, c := range cases {
		mode, reason := ResolveMode(c.raw, c.backend)
		if mode != c.mode {
			t.Errorf("ResolveMode(%q, %q) mode %q, want %q", c.raw, c.backend, mode, c.mode)
		}
		switch {
		case c.reason == "" && reason != nil:
			t.Errorf("ResolveMode(%q) unexpected reason %q", c.raw, *reason)
		case c.reason != "" && (reason == nil || !strings.Contains(*reason, c.reason)):
			t.Errorf("ResolveMode(%q) reason %v, want %q", c.raw, reason, c.reason)
		}
	}
}

func TestPyRepr(t *testing.T) {
	for in, want := range map[string]string{
		"root":   `'root'`,
		"o'b":    `"o'b"`,
		`a'b"c`:  `'a\'b"c'`,
		`back\`:  `'back\\'`,
		"tab\t":  `'tab\t'`,
		"Пётр":   `'Пётр'`,
		"\x01":   `'\x01'`,
		"":       `''`,
		`say "x"`: `'say "x"'`,
	} {
		if got := PyRepr(in); got != want {
			t.Errorf("PyRepr(%q) = %s, want %s", in, got, want)
		}
	}
}
