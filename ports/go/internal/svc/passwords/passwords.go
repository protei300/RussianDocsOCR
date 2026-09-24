// Package passwords is password hashing — deliberately slow, unlike the API-key hashing in
// svc/auth.
//
// **Why this is not sha256, which svc/auth uses for API keys.** An API key is 32 random bytes:
// there is nothing to guess, so a fast hash is the right tool. A password is a short string a
// human chose, and a fast hash over a low-entropy input is exactly what an attacker wants — a
// consumer GPU tries billions of sha256 candidates per second against a stolen file. The
// defence is an algorithm that is INTENTIONALLY expensive in time and memory, with a
// per-password salt so one cracking run cannot amortise across accounts.
//
// **Why Argon2id.** It is memory-hard — the cost cannot be bought down with parallel hardware
// nearly as cheaply as bcrypt's — and it is the first recommendation in OWASP's password
// storage guidance.
//
// **Why the stored value is a PHC string and not (salt, hash) fields.** The data directory is
// implementation-neutral: the Python, .NET and Kotlin services read the same files as this one,
// which is how the four implementations are kept honest. A hash stored as
//
//	$argon2id$v=19$m=65536,t=3,p=4$<salt>$<digest>
//
// carries its own algorithm and parameters, so any implementation verifies it without agreeing
// on a private layout in advance. golang.org/x/crypto/argon2 computes the digest and nothing
// else, so the PHC encoder and parser below are this port's own — about as long as this
// comment, and the part the interop vectors in the tests exist to pin down.
//
// Port of service/core/passwords.py. Python guards its argon2-cffi import so a deployment
// without it still starts on the PIN path; here the library is a compile-time dependency and
// cannot be missing, so that branch does not exist (ports/AUTH.md §1).
package passwords

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/crypto/argon2"
	"golang.org/x/sync/semaphore"
)

// The OWASP baseline. Changing these is safe: the parameters travel inside each hash, Verify
// reads them from there, and NeedsRehash reports which stored hashes predate the change.
const (
	MemoryKiB  = 65536 // 64 MiB
	Iterations = 3
	Lanes      = 4
	DigestLen  = 32
	SaltLen    = 16

	// argonVersion is 0x13, the only version argon2-cffi writes and x/crypto implements.
	argonVersion = 19
	variant      = "argon2id"
)

// HashConcurrency is how many Argon2 computations may run at once, service-wide.
//
// Memory-hard hashing is a denial-of-service lever pointed at yourself: every hash costs
// 64 MiB, Go serves each request on its own goroutine, and the sign-in endpoint is reachable
// without a token. A flood of logins with a different username each time walks straight past
// the per-account lockout, and forty concurrent verifications is 2.5 GB of RAM. Four slots cap
// the peak at 256 MiB; requests beyond that wait their turn instead of all allocating at once.
//
// A weighted semaphore rather than a buffered channel, because CONVENTIONS §1 keeps channels
// for the lease pool alone — and SemaphoreSlim / kotlinx Semaphore are what the other two
// ports reach for, so this reads the same in all three.
const HashConcurrency = 4

// slots is package state on purpose: the limit is SERVICE-wide, exactly like Python's
// module-level BoundedSemaphore. A per-caller semaphore would limit nothing.
var slots = semaphore.NewWeighted(HashConcurrency)

// Bounds on what a STORED hash may ask for. Verify reads its parameters from the record, so
// without these a hostile or corrupted users.json could make one login allocate gigabytes
// (m is in KiB: 4194304 is 4 GiB) or spin for minutes. Chosen to admit every sane
// configuration — the OWASP baseline, the RFC 9106 second recommendation (m=19456,t=2,p=1),
// and parameters raised a few times over — and nothing an attacker would find useful.
const (
	minIterations = 1
	maxIterations = 10
	maxMemoryKiB  = 1 << 20 // 1 GiB
	minLanes      = 1
	maxLanes      = 16
	minDigestLen  = 16
	maxDigestLen  = 64
	minSaltLen    = 8
)

// b64 is standard base64 WITHOUT padding (alphabet +/, not URL-safe). That is what the PHC
// format specifies and what argon2-cffi writes; a URL-safe or padded encoding here would make
// every hash this port writes unreadable by the other three services.
var b64 = base64.RawStdEncoding

// phc is one parsed hash string.
type phc struct {
	memory     uint32
	iterations uint32
	lanes      uint8
	salt       []byte
	digest     []byte
}

// ErrInvalidHash is the one failure the parser reports. Its Reason says which part was wrong —
// logged, never returned to a caller (see Verify).
type ErrInvalidHash struct{ Reason string }

func (e *ErrInvalidHash) Error() string { return "invalid argon2 hash: " + e.Reason }

func invalid(format string, args ...any) error {
	return &ErrInvalidHash{Reason: fmt.Sprintf(format, args...)}
}

// Hash hashes a password for storage and returns a self-describing PHC string.
func Hash(plain string) (string, error) {
	salt := make([]byte, SaltLen)
	if _, err := rand.Read(salt); err != nil {
		return "", fmt.Errorf("passwords: salt: %w", err)
	}
	if err := slots.Acquire(context.Background(), 1); err != nil {
		return "", fmt.Errorf("passwords: acquire hash slot: %w", err)
	}
	digest := argon2.IDKey([]byte(plain), salt, Iterations, MemoryKiB, Lanes, DigestLen)
	slots.Release(1)
	return encode(phc{memory: MemoryKiB, iterations: Iterations, lanes: Lanes,
		salt: salt, digest: digest}), nil
}

func encode(h phc) string {
	return fmt.Sprintf("$%s$v=%d$m=%d,t=%d,p=%d$%s$%s", variant, argonVersion,
		h.memory, h.iterations, h.lanes, b64.EncodeToString(h.salt), b64.EncodeToString(h.digest))
}

// parse reads a PHC string, enforcing the bounds above BEFORE anything is allocated for the
// computation itself.
//
// Strict on purpose: exactly six '$'-separated fields, the argon2id variant only, version 19,
// each of m/t/p exactly once. A lenient parser is how "$argon2i$..." — a weaker variant — would
// be verified as if it were argon2id and silently accepted.
func parse(stored string) (phc, error) {
	parts := strings.Split(stored, "$")
	if len(parts) != 6 || parts[0] != "" {
		return phc{}, invalid("expected 6 '$'-separated fields, got %d", len(parts))
	}
	if parts[1] != variant {
		return phc{}, invalid("unsupported variant %q", parts[1])
	}
	if parts[2] != "v="+strconv.Itoa(argonVersion) {
		return phc{}, invalid("unsupported version %q", parts[2])
	}

	var m, t, p uint64
	seen := map[string]bool{}
	for _, pair := range strings.Split(parts[3], ",") {
		key, raw, ok := strings.Cut(pair, "=")
		if !ok || seen[key] {
			return phc{}, invalid("malformed parameter %q", pair)
		}
		seen[key] = true
		// ParseUint rejects signs, blanks and anything beyond 32 bits, so "-1", "+3" and
		// "99999999999" all fail here rather than wrapping around.
		v, err := strconv.ParseUint(raw, 10, 32)
		if err != nil {
			return phc{}, invalid("parameter %q is not a number", pair)
		}
		switch key {
		case "m":
			m = v
		case "t":
			t = v
		case "p":
			p = v
		default:
			return phc{}, invalid("unknown parameter %q", key)
		}
	}
	if len(seen) != 3 {
		return phc{}, invalid("expected m, t and p")
	}
	if t < minIterations || t > maxIterations {
		return phc{}, invalid("t=%d outside %d..%d", t, minIterations, maxIterations)
	}
	if p < minLanes || p > maxLanes {
		return phc{}, invalid("p=%d outside %d..%d", p, minLanes, maxLanes)
	}
	// Argon2 itself requires m >= 8·p; the upper bound is ours, and it is the one that stops a
	// record from asking for 4 GiB.
	if m < 8*p || m > maxMemoryKiB {
		return phc{}, invalid("m=%d outside %d..%d", m, 8*p, maxMemoryKiB)
	}

	salt, err := b64.DecodeString(parts[4])
	if err != nil {
		return phc{}, invalid("salt is not unpadded standard base64")
	}
	digest, err := b64.DecodeString(parts[5])
	if err != nil {
		return phc{}, invalid("digest is not unpadded standard base64")
	}
	if len(salt) < minSaltLen {
		return phc{}, invalid("salt is %d bytes, need at least %d", len(salt), minSaltLen)
	}
	if len(digest) < minDigestLen || len(digest) > maxDigestLen {
		return phc{}, invalid("digest is %d bytes, outside %d..%d",
			len(digest), minDigestLen, maxDigestLen)
	}
	return phc{memory: uint32(m), iterations: uint32(t), lanes: uint8(p),
		salt: salt, digest: digest}, nil
}

// Verify checks a password against a stored hash.
//
// Returns false for a wrong password AND for a malformed stored hash. The distinction is
// deliberately not exposed: an endpoint that answered differently for "wrong password" and
// "corrupt record" — a 500 on one, a 401 on the other — would let an attacker enumerate which
// accounts exist.
//
// **Fail closed on anything unexpected.** The Python version had a narrow exception list and
// it was wrong twice in five minutes (a non-ASCII byte raised UnicodeEncodeError, a missing
// hash raised AttributeError). The Go analogue of "anything unexpected" is a panic from deep in
// the library, so a deferred recover turns that into false too — the one place in this port
// where recover is used, and it is not control flow: it is the authentication primitive
// refusing to crash the request that asked it. The reason is logged so a real bug stays visible
// instead of hiding behind a silent false.
func Verify(stored, candidate string) (ok bool) {
	defer func() {
		if rec := recover(); rec != nil {
			slog.Warn("[AUTH] unreadable password hash (panic) — treating as a failed login",
				"panic", fmt.Sprint(rec))
			ok = false
		}
	}()

	h, err := parse(stored)
	if err != nil {
		var bad *ErrInvalidHash
		kind := "error"
		if errors.As(err, &bad) {
			kind = "ErrInvalidHash"
		}
		slog.Warn(fmt.Sprintf("[AUTH] unreadable password hash (%s) — treating as a failed login",
			kind), "reason", err.Error())
		return false
	}

	if err := slots.Acquire(context.Background(), 1); err != nil {
		return false
	}
	defer slots.Release(1)
	got := argon2.IDKey([]byte(candidate), h.salt, h.iterations, h.memory, h.lanes,
		uint32(len(h.digest)))
	// Constant time: a byte-by-byte == leaks how long a prefix matched.
	return subtle.ConstantTimeCompare(got, h.digest) == 1
}

// NeedsRehash reports whether a hash was made with other parameters than the current ones.
//
// Call it after a SUCCESSFUL verification — that is the only moment the plaintext is available
// to re-hash with the current cost. An unparseable hash reports true, matching argon2-cffi's
// check_needs_rehash wrapped the way the reference wraps it.
func NeedsRehash(stored string) bool {
	h, err := parse(stored)
	if err != nil {
		return true
	}
	return h.memory != MemoryKiB || h.iterations != Iterations || h.lanes != Lanes ||
		len(h.digest) != DigestLen
}

// --- composition rules --------------------------------------------------------

// MinPasswordLength is the length rule's bound, kept as a number so the label and the pattern
// cannot disagree.
const MinPasswordLength = 8

// Rule is one composition rule, as data.
type Rule struct {
	Code    string `json:"code"`
	Label   string `json:"label"`
	Pattern string `json:"pattern"`
	re      *regexp.Regexp
}

// rules are the composition rules, in declaration order.
//
// The patterns are written to mean the same thing in Python's `re`, in Go's RE2 and in a
// browser's RegExp — no lookbehind, no \p{...}, no named groups — so the server hands this
// exact list to the UI and the password page ticks rules off as you type without a second copy
// in TypeScript. Two copies of a validation rule drift, and the drift shows up as a form that
// accepts a password the server then rejects.
//
// **The length rule is matched by the regexp, not by len().** RE2 matches over RUNES, so `.`
// is one code point and `.{8,}` counts characters — "пароль12" is eight, while len() would say
// fourteen bytes and pass seven Cyrillic letters. Using the pattern rather than
// utf8.RuneCountInString keeps the one remaining edge identical to the reference too: `.` does
// not match a newline in any of the three engines, so a password with a line break is judged
// the same way everywhere.
//
// Cyrillic is in the letter classes on purpose: this is a Russian deployment, and a rule that
// silently rejected "Пароль1" as having no letters would be a bug, not a policy. The labels
// stay English — the UI is English-only by project convention.
var rules = []Rule{
	newRule("length", fmt.Sprintf("at least %d characters", MinPasswordLength),
		fmt.Sprintf(".{%d,}", MinPasswordLength)),
	newRule("digit", "at least one digit", "[0-9]"),
	newRule("letter", "at least one letter", "[a-zA-Zа-яёА-ЯЁ]"),
	newRule("upper", "at least one capital letter", "[A-ZА-ЯЁ]"),
}

func newRule(code, label, pattern string) Rule {
	return Rule{Code: code, Label: label, Pattern: pattern, re: regexp.MustCompile(pattern)}
}

// Rules returns a copy of the rule list, so no caller can edit the policy in place.
func Rules() []Rule { return append([]Rule(nil), rules...) }

// UnmetRules returns the codes of the rules this password fails, in declaration order.
//
// A rule fails when its pattern finds no match ANYWHERE in the password — re.search, not
// re.fullmatch — which is also what RegExp.test does in the browser.
func UnmetRules(candidate string) []string {
	var failed []string
	for _, rule := range Rules() {
		if !rule.re.MatchString(candidate) {
			failed = append(failed, rule.Code)
		}
	}
	return failed
}

// Validate returns a human-readable complaint, or "" when the password is acceptable.
//
// The seeded administrator's password bypasses this — it is "1234" by design, printed on the
// login page, and the account can do nothing until it is changed. Every password a PERSON
// chooses goes through it.
func Validate(candidate string) string {
	failed := UnmetRules(candidate)
	if len(failed) == 0 {
		return ""
	}
	labels := map[string]string{}
	for _, rule := range Rules() {
		labels[rule.Code] = rule.Label
	}
	names := make([]string, 0, len(failed))
	for _, code := range failed {
		names = append(names, labels[code])
	}
	return "Password needs: " + strings.Join(names, ", ")
}

// RulesForUI is the rule list as the password page consumes it — one source of truth.
func RulesForUI() []map[string]string {
	out := []map[string]string{}
	for _, rule := range Rules() {
		out = append(out, map[string]string{
			"code": rule.Code, "label": rule.Label, "pattern": rule.Pattern})
	}
	return out
}
