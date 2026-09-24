package auth

import (
	"strconv"
	"testing"
	"time"
)

// A throttle on a clock the test moves by hand: max 3 failures in a 120 s window.
func testThrottle() (*Throttle, *time.Duration) {
	var now time.Duration
	return newThrottleWithClock(3, 120, func() time.Duration { return now }), &now
}

func TestThrottlePerAccount(t *testing.T) {
	th, now := testThrottle()
	for i := 0; i < 3; i++ {
		if th.BlockedFor("admin", "10.0.0.1") != 0 {
			t.Fatalf("blocked after %d failures", i)
		}
		th.NoteFailure("admin", "10.0.0.1")
		*now += 10 * time.Second
	}
	// Oldest failure at t=0, now t=30: 120 - 30 = 90 s remain.
	if got := th.BlockedFor("admin", "10.0.0.1"); got != 90 {
		t.Errorf("BlockedFor = %d, want 90", got)
	}
	// The identity is trimmed and case-folded, so the lockout cannot be dodged by retyping.
	if th.BlockedFor(" ADMIN ", "10.0.0.1") == 0 {
		t.Error("a case/space variant of the username escaped the lockout")
	}
	// Another account from the same address, and the same account from another address, are
	// unaffected — a known username cannot be locked out from everywhere by failing on purpose.
	if th.BlockedFor("petrov", "10.0.0.1") != 0 {
		t.Error("another account from the same address was locked by one account's failures")
	}
	if th.BlockedFor("admin", "10.0.0.2") != 0 {
		t.Error("the account was locked from another address")
	}
	// The window slides: once the oldest failure is 120 s old it no longer counts.
	*now = 120 * time.Second
	if got := th.BlockedFor("admin", "10.0.0.1"); got != 0 {
		t.Errorf("still blocked (%d s) after the oldest failure left the window", got)
	}
}

// Rotating usernames must not escape: the address-wide counter allows 3× the per-account budget
// across every identity, then blocks the address for all of them.
func TestThrottlePerAddressAcrossRotatedUsernames(t *testing.T) {
	th, now := testThrottle()
	for i := 0; i < 3*AddressLimitFactor; i++ {
		name := "spray" + strconv.Itoa(i)
		if th.BlockedFor(name, "10.0.0.1") != 0 {
			t.Fatalf("blocked after only %d sprayed failures", i)
		}
		th.NoteFailure(name, "10.0.0.1")
		*now += time.Second
	}
	if th.BlockedFor("fresh-name", "10.0.0.1") == 0 {
		t.Fatal("rotating usernames was never throttled")
	}
	if th.BlockedFor("fresh-name", "10.0.0.2") != 0 {
		t.Error("the address lockout leaked to another address")
	}
}

// A success clears ONLY the account's counter. Clearing the address counter too would let an
// attacker holding one valid account reset their budget between guesses at everyone else's.
func TestThrottleSuccessClearsOnlyTheAccountCounter(t *testing.T) {
	th, _ := testThrottle()
	for i := 0; i < 3; i++ {
		th.NoteFailure("mine", "10.0.0.1")
	}
	if th.BlockedFor("mine", "10.0.0.1") == 0 {
		t.Fatal("not blocked after 3 failures")
	}
	th.Clear("mine", "10.0.0.1")
	if th.BlockedFor("mine", "10.0.0.1") != 0 {
		t.Error("a successful sign-in did not clear the account counter")
	}
	// 3 address failures are still on record; 6 more across other names reach 9 = 3×3.
	for i := 0; i < 6; i++ {
		th.NoteFailure("victim"+strconv.Itoa(i), "10.0.0.1")
	}
	if th.BlockedFor("anyone", "10.0.0.1") == 0 {
		t.Error("the success reset the address-wide counter")
	}
}

// Retry-After is never 0 while blocked, even at the last instant of the window.
func TestThrottleReportsAtLeastOneSecond(t *testing.T) {
	th, now := testThrottle()
	for i := 0; i < 3; i++ {
		th.NoteFailure("admin", "a")
	}
	*now = 120*time.Second - time.Millisecond
	if got := th.BlockedFor("admin", "a"); got != 1 {
		t.Errorf("BlockedFor at the window's edge = %d, want 1", got)
	}
}

// A limit of zero would mean "always blocked" — and the reference then indexes an empty list.
func TestThrottleRaisesAZeroLimit(t *testing.T) {
	th := NewThrottle(0, 0)
	if th.BlockedFor("admin", "a") != 0 {
		t.Error("a zero limit blocked a caller with no failures")
	}
}
