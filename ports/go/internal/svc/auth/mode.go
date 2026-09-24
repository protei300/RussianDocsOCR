package auth

import (
	"fmt"
	"strings"
)

// Which authentication mode is in force.
//
// Resolved HERE, by one function, and never by reading the raw AUTH_MODE elsewhere: the whole
// point is that one place turns an unusable configuration into a usable one, so a downgrade
// behaves consistently everywhere instead of half-working.

// String constants rather than an enum — they go on the wire in /auth/config and /auth/me
// (CONVENTIONS §1).
const (
	PinMode   = "pin"
	UsersMode = "users"
)

// Modes is the closed set, in the order the downgrade message names them.
var Modes = []string{PinMode, UsersMode}

// FilesBackend is the one store backend named accounts are implemented for.
const FilesBackend = "files"

// ResolveMode returns (mode, downgradeReason). downgradeReason is nil when the configured value
// was honoured.
//
// **This function never fails and never refuses to serve.** An existing deployment that pulls
// this version must keep working exactly as before, and "as before" is the PIN. So every way of
// getting it wrong — an unset value, a typo, a mode the storage backend cannot support —
// resolves to PIN with a reason attached, rather than a service that will not start.
//
// The reason is RETURNED rather than only logged so /auth/config can show it. A silent downgrade
// is the failure to avoid: somebody configured named accounts, and if they are quietly given a
// shared four-digit PIN instead, nothing in the interface would say so.
//
// Python has a fourth branch, for argon2-cffi not being installed. Here Argon2 is a compile-time
// dependency and cannot be missing, so the branch does not exist (ports/AUTH.md §1).
func ResolveMode(raw, backend string) (string, *string) {
	value := strings.ToLower(strings.TrimSpace(raw))
	if value == "" {
		return PinMode, nil // unset is not a mistake, it is the default
	}
	if value != PinMode && value != UsersMode {
		reason := fmt.Sprintf("AUTH_MODE=%s is not one of %s — falling back to PIN authentication",
			PyRepr(value), strings.Join(Modes, ", "))
		return PinMode, &reason
	}
	if value == UsersMode && backend != FilesBackend {
		reason := "AUTH_MODE=users is implemented for the temporary file store only; the " +
			"database backend's user methods are stubs you are expected to implement (see " +
			"docs/auth.md) — falling back to PIN authentication"
		return PinMode, &reason
	}
	return value, nil
}

// PyRepr quotes a string the way Python's repr() does, because several contract messages are
// built with `%r` in the reference — "Unknown role 'root'", "User 'ADMIN' already exists",
// "AUTH_MODE='bogus' is not ..." — and a client, or the contract test, compares them verbatim.
//
// Single quotes unless the text contains a single quote and no double quote, as CPython
// chooses; backslash, the chosen quote and control characters escaped. Printable non-ASCII —
// Cyrillic in a display name, say — is left as is, which is what Python 3's repr does too.
func PyRepr(s string) string {
	quote := byte('\'')
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}
	var b strings.Builder
	b.WriteByte(quote)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == rune(quote):
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte(quote)
	return b.String()
}
