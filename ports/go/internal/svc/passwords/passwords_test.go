package passwords

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

// The interop vectors from ports/AUTH.md §3, produced by argon2-cffi. The second one uses
// DIFFERENT parameters from the ones this package hashes with, which is what proves Verify reads
// m/t/p from the string instead of assuming its own.
var vectors = []struct{ password, hash string }{
	{"Vector-Pass1",
		"$argon2id$v=19$m=65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo"},
	{"Пароль-42",
		"$argon2id$v=19$m=19456,t=2,p=1$rppcAWOP4qJuFb6Dc52G3g$NoDmjBcYZrj9DJvzNb421/YYxfGj1D+TxplD/tAEero"},
}

func TestInteropVectorsVerify(t *testing.T) {
	for _, v := range vectors {
		if !Verify(v.hash, v.password) {
			t.Errorf("argon2-cffi vector for %q did not verify", v.password)
		}
		if Verify(v.hash, v.password+"x") {
			t.Errorf("vector for %q verified with a wrong password", v.password)
		}
	}
}

// The second vector's parameters differ from ours, so it must be flagged for re-hashing, and the
// first — made with exactly ours — must not.
func TestNeedsRehashReadsTheStoredParameters(t *testing.T) {
	if NeedsRehash(vectors[0].hash) {
		t.Error("a hash with the current parameters was flagged for rehash")
	}
	if !NeedsRehash(vectors[1].hash) {
		t.Error("an m=19456,t=2,p=1 hash was not flagged for rehash")
	}
	if !NeedsRehash("garbage") {
		t.Error("an unparseable hash must report needs-rehash")
	}
}

func TestFreshHashRoundTrips(t *testing.T) {
	h, err := Hash("Str0ng-Pass")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(h, "$argon2id$v=19$m=65536,t=3,p=4$") {
		t.Fatalf("hash %q does not carry the OWASP parameters", h)
	}
	// Standard alphabet without padding: no '=', and never the URL-safe '-' or '_'.
	parts := strings.Split(h, "$")
	if len(parts) != 6 || strings.ContainsAny(parts[4]+parts[5], "=-_") {
		t.Errorf("hash %q is not six fields of unpadded standard base64", h)
	}
	if !Verify(h, "Str0ng-Pass") {
		t.Error("a fresh hash did not verify")
	}
	if Verify(h, "Str0ng-pass") {
		t.Error("a fresh hash verified with a wrong password")
	}
	if NeedsRehash(h) {
		t.Error("a fresh hash was flagged for rehash")
	}
	again, err := Hash("Str0ng-Pass")
	if err != nil {
		t.Fatal(err)
	}
	if again == h {
		t.Error("two hashes of the same password are identical; the salt is not random")
	}
}

// Every malformed record must come back false WITHOUT panicking and without a 64 MiB+
// computation being attempted on it.
func TestMalformedHashesFailClosed(t *testing.T) {
	good := vectors[0].hash
	cases := map[string]string{
		"empty":           "",
		"argon2i variant": strings.Replace(good, "$argon2id$", "$argon2i$", 1),
		"argon2d variant": strings.Replace(good, "$argon2id$", "$argon2d$", 1),
		"truncated":       good[:len(good)/2],
		"no digest":       good[:strings.LastIndex(good, "$")],
		"bad base64 salt": strings.Replace(good, "PWP+BS8J", "PWP!BS8J", 1),
		"bad base64 hash": good[:len(good)-4] + "@@@@",
		"url-safe base64": strings.ReplaceAll(strings.ReplaceAll(good, "+", "-"), "/", "_"),
		"wrong version":   strings.Replace(good, "v=19", "v=16", 1),
		"missing lanes":   strings.Replace(good, ",p=4", "", 1),
		"duplicate param": strings.Replace(good, "p=4", "t=4", 1),
		"negative t":      strings.Replace(good, "t=3", "t=-3", 1),
		"zero t":          strings.Replace(good, "t=3", "t=0", 1),
		"huge t":          strings.Replace(good, "t=3", "t=100000", 1),
		"zero p":          strings.Replace(good, "p=4", "p=0", 1),
		"too many lanes":  strings.Replace(good, "p=4", "p=255", 1),
		"memory below 8p": strings.Replace(good, "m=65536", "m=16", 1),
		"not a hash":      "sha256:deadbeef",
		"non-ascii":       good + "ё",
		"short digest":    "$argon2id$v=19$m=65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$AAAA",
		"short salt":      "$argon2id$v=19$m=65536,t=3,p=4$AAAA$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo",
		"trailing dollar": good + "$",
		"number overflow": strings.Replace(good, "m=65536", "m=99999999999", 1),
	}
	for name, stored := range cases {
		if Verify(stored, vectors[0].password) {
			t.Errorf("%s: a malformed hash verified", name)
		}
	}
}

// m=4194304 is 4 GiB. It must be refused by the bounds check, not attempted — the test would
// either take many seconds or exhaust memory if it were computed.
func TestAbsurdMemoryIsRefusedWithoutAllocating(t *testing.T) {
	absurd := strings.Replace(vectors[0].hash, "m=65536", "m=4194304", 1)
	start := time.Now()
	if Verify(absurd, vectors[0].password) {
		t.Fatal("an m=4194304 hash verified")
	}
	if elapsed := time.Since(start); elapsed > 50*time.Millisecond {
		t.Errorf("refusing m=4194304 took %v; the hash was computed rather than rejected", elapsed)
	}
}

func TestRulesAreServedInOrder(t *testing.T) {
	var codes []string
	for _, r := range RulesForUI() {
		codes = append(codes, r["code"])
		if r["label"] == "" || r["pattern"] == "" {
			t.Errorf("rule %q lacks a label or pattern", r["code"])
		}
	}
	if want := []string{"length", "digit", "letter", "upper"}; !reflect.DeepEqual(codes, want) {
		t.Fatalf("rule codes %v, want %v", codes, want)
	}
}

// The patterns are served verbatim to the browser, so they are pinned byte for byte.
func TestRulePatternsAreTheContract(t *testing.T) {
	want := map[string]string{
		"length": ".{8,}", "digit": "[0-9]",
		"letter": "[a-zA-Zа-яёА-ЯЁ]", "upper": "[A-ZА-ЯЁ]",
	}
	for _, r := range RulesForUI() {
		if r["pattern"] != want[r["code"]] {
			t.Errorf("%s pattern %q, want %q", r["code"], r["pattern"], want[r["code"]])
		}
	}
}

func TestPasswordRulesIncludingCyrillic(t *testing.T) {
	cases := []struct {
		password string
		failed   []string
	}{
		// Lower-case Cyrillic letters count as letters; only the capital is missing.
		{"пароль12", []string{"upper"}},
		// Seven Cyrillic letters and a digit is eight characters (fifteen bytes): it passes
		// the length rule…
		{"парольё1", []string{"upper"}},
		// …while six letters and a digit is seven characters but thirteen bytes, so a byte
		// count would wrongly pass it. These two cases are the rune-versus-byte boundary.
		{"пароль1", []string{"length", "upper"}},
		{"ПАРОЛЬ1", []string{"length"}},
		{"Пароль12", nil},
		{"Str0ng-Pass", nil},
		{"weakpass", []string{"digit", "upper"}},
		{"1234", []string{"length", "letter", "upper"}},
		{"", []string{"length", "digit", "letter", "upper"}},
		{"ABCDEFG1", nil},
		{"Ё1234567", nil},
	}
	for _, c := range cases {
		if got := UnmetRules(c.password); !reflect.DeepEqual(got, c.failed) {
			t.Errorf("UnmetRules(%q) = %v, want %v", c.password, got, c.failed)
		}
	}
}

func TestValidateMessage(t *testing.T) {
	if got := Validate("weakpass"); got !=
		"Password needs: at least one digit, at least one capital letter" {
		t.Errorf("Validate(weakpass) = %q", got)
	}
	if got := Validate("short1A"); got != "Password needs: at least 8 characters" {
		t.Errorf("Validate(short1A) = %q", got)
	}
	if got := Validate("Str0ng-Pass"); got != "" {
		t.Errorf("a strong password was refused: %q", got)
	}
}
