package auth

import (
	"strings"
	"sync"
	"time"
)

// Failed-login throttling.
//
// In memory, and honest about what that is: a single-process service pinned to one instance, so
// a map is the whole mechanism. A deployment behind several instances needs shared state —
// Redis, or the reverse proxy's own rate limiting — and docs/auth.md says so.
//
// Why it exists at all: four digits is 10 000 guesses, and a username is a longer-lived secret
// than a PIN nobody expected to hold. Adding named accounts without throttling would have made
// the service easier to attack, not harder.
//
// Port of the throttle section of service/core/auth.py. There it is module state; here it is a
// value the Server owns, so a test gets a fresh one and an injected clock.

// AnyIdentity is the key under which failures from one address are counted regardless of the
// username tried. Not a valid username (see repo/users.go), so it can never collide with a real
// account's counter.
const AnyIdentity = "*"

// AddressLimitFactor is how many failures one address may make across ALL usernames before it
// is blocked, as a multiple of the per-account limit. Higher than the per-account limit because
// several people can share one address (an office NAT), lower than unlimited because otherwise
// rotating usernames is a free pass.
const AddressLimitFactor = 3

// sweepThreshold is the map size beyond which stale keys are dropped. Without the sweep the map
// grows once per distinct (identity, address) pair for the life of the process.
const sweepThreshold = 10_000

type throttleKey struct{ identity, client string }

// Throttle counts failed sign-ins per (identity, address) and per address.
type Throttle struct {
	mu          sync.Mutex
	attempts    map[throttleKey][]time.Duration
	maxAttempts int
	window      time.Duration
	// now is MONOTONIC time since construction. time.Since reads Go's monotonic clock, so a
	// wall-clock step — NTP, a VM resuming — can neither lift a lockout early nor extend it.
	now func() time.Duration
}

// NewThrottle builds a throttle with the configured limits.
//
// A limit below one is raised to one: zero would mean "always blocked", and the reference then
// indexes an empty list. A misconfiguration should throttle hard, not crash every sign-in.
func NewThrottle(maxAttempts, lockoutSeconds int) *Throttle {
	start := time.Now()
	return newThrottleWithClock(maxAttempts, lockoutSeconds,
		func() time.Duration { return time.Since(start) })
}

func newThrottleWithClock(maxAttempts, lockoutSeconds int, now func() time.Duration) *Throttle {
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	if lockoutSeconds < 1 {
		lockoutSeconds = 1
	}
	return &Throttle{
		attempts:    map[throttleKey][]time.Duration{},
		maxAttempts: maxAttempts,
		window:      time.Duration(lockoutSeconds) * time.Second,
		now:         now,
	}
}

// key normalises the identity: trimmed and case-folded, so "Admin " and "admin" share one
// counter. Go has no casefold; ToLower is the same for every name the username rule admits
// (ASCII only), and for the rest it only decides which counter a doomed attempt lands in.
func key(identity, client string) throttleKey {
	if client == "" {
		client = "-"
	}
	return throttleKey{identity: strings.ToLower(strings.TrimSpace(identity)), client: client}
}

// recentLocked returns the failures still inside the window, oldest first.
func (t *Throttle) recentLocked(k throttleKey, now time.Duration) []time.Duration {
	var recent []time.Duration
	for _, at := range t.attempts[k] {
		if now-at < t.window {
			recent = append(recent, at)
		}
	}
	return recent
}

func (t *Throttle) blockedLocked(k throttleKey, limit int, now time.Duration) int {
	recent := t.recentLocked(k, now)
	if len(recent) < limit {
		return 0
	}
	// int() truncates toward zero, which for this always-positive value is the floor the
	// contract specifies.
	remaining := int((t.window - (now - recent[0])).Seconds())
	if remaining < 1 {
		remaining = 1
	}
	return remaining
}

// BlockedFor returns the seconds remaining in a lockout, or 0 when the caller may try.
//
// Two counters, and both are needed. **Per (account, address)** — locking by account alone would
// let anyone lock a known user out of the service by failing on purpose from anywhere. **Per
// address, across every account** — without it the first counter is defeated by trying a
// different username each time, which is exactly what a password-spraying run does. The first
// Python version had only the first counter.
func (t *Throttle) BlockedFor(identity, client string) int {
	now := t.now()
	t.mu.Lock()
	defer t.mu.Unlock()
	own := t.blockedLocked(key(identity, client), t.maxAttempts, now)
	address := t.blockedLocked(key(AnyIdentity, client), t.maxAttempts*AddressLimitFactor, now)
	if address > own {
		return address
	}
	return own
}

// NoteFailure records one failed attempt against BOTH counters.
func (t *Throttle) NoteFailure(identity, client string) {
	now := t.now()
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, k := range []throttleKey{key(identity, client), key(AnyIdentity, client)} {
		t.attempts[k] = append(t.recentLocked(k, now), now)
	}
	if len(t.attempts) > sweepThreshold {
		for k := range t.attempts {
			if len(t.recentLocked(k, now)) == 0 {
				delete(t.attempts, k)
			}
		}
	}
}

// Clear is called after a successful sign-in, so one typo does not linger.
//
// It clears the account's counter ONLY, never the address-wide one: otherwise an attacker
// holding one valid account could reset their budget by signing in between guesses at everyone
// else's.
func (t *Throttle) Clear(identity, client string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.attempts, key(identity, client))
}
