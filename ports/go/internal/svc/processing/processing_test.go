package processing

import (
	"strings"
	"testing"
)

// §7.6 — the resolution table of ports/MEMORY.md §1. A wrong value fails toward privacy.
func TestResolveTable(t *testing.T) {
	for _, c := range []struct {
		raw, mode string
		reason    bool
	}{
		{"", Stored, false},
		{"   ", Stored, false},
		{"stored", Stored, false},
		{" Stored ", Stored, false},
		{"memory", Memory, false},
		{"MEMORY", Memory, false},
		{"memroy", Memory, true},
		{"store", Memory, true},
		{"off", Memory, true},
	} {
		mode, reason := Resolve(c.raw)
		if mode != c.mode || (reason != nil) != c.reason {
			t.Errorf("Resolve(%q) = %q, %v; want %q, reason %v", c.raw, mode, reason, c.mode, c.reason)
		}
	}
	_, reason := Resolve("memroy")
	if reason == nil || !strings.Contains(*reason, "'memroy'") || !strings.Contains(*reason, "memory") {
		t.Errorf("the reason must quote the bad value: %v", reason)
	}
}
