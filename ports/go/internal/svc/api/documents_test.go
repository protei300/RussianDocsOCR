package api

import "testing"

// safeFilename is a DISPLAY sanitiser, not the path-traversal defence — artifacts always use
// a fixed name. These cases pin what it does anyway, because a filename reaches the list page
// and a bad one there is a rendering bug or worse.
func TestSafeFilename(t *testing.T) {
	cases := map[string]string{
		"passport.jpg":                 "passport.jpg",
		`..\..\windows\system32\a.jpg`: "a.jpg",
		"../../etc/passwd":             "passwd",
		"":                             "upload",
		"   ":                          "upload",
		// Reserved-on-Windows characters are dropped rather than replaced, so the name stays
		// readable instead of turning into underscores.
		`bad<>:"|?*name.jpg`: "badname.jpg",
		// Cyrillic survives intact: these names are routinely Cyrillic here.
		"паспорт.jpg": "паспорт.jpg",
	}
	for in, want := range cases {
		if got := safeFilename(in); got != want {
			t.Errorf("safeFilename(%q) = %q, want %q", in, got, want)
		}
	}
}

// Truncation counts RUNES, not bytes: a Cyrillic name cut mid-rune renders as a replacement
// character.
func TestSafeFilenameTruncatesByRune(t *testing.T) {
	long := ""
	for i := 0; i < 300; i++ {
		long += "я"
	}
	got := safeFilename(long)
	if n := len([]rune(got)); n != maxFilenameLen {
		t.Fatalf("truncated to %d runes, want %d", n, maxFilenameLen)
	}
	for _, r := range got {
		if r == '�' {
			t.Fatal("truncation split a rune")
		}
	}
}

func TestSplitDocType(t *testing.T) {
	cases := []struct{ in, base, era string }{
		{"INTPASSPORT_2011", "INTPASSPORT", "2011"},
		{"INTPASSPORTADDR_ALL", "INTPASSPORTADDR", "ALL"},
		// No underscore: survivable, unlike the pipeline's own split which raises.
		{"NONE", "NONE", ""},
	}
	for _, c := range cases {
		base, era := splitDocType(c.in)
		if base != c.base || era != c.era {
			t.Errorf("splitDocType(%q) = (%q, %q), want (%q, %q)", c.in, base, era, c.base, c.era)
		}
	}
}

// A missing key must become an empty CONTAINER, not a null: the SPA iterates boxes and fields
// unconditionally, so a null there is a runtime error in the browser rather than an empty
// table.
func TestOrEmptyHelpers(t *testing.T) {
	if got := orEmptyList(nil); len(got.([]any)) != 0 {
		t.Error("orEmptyList(nil) must be an empty array")
	}
	if got := orEmptyMap(nil); len(got.(map[string]any)) != 0 {
		t.Error("orEmptyMap(nil) must be an empty object")
	}
	// A present value passes through untouched.
	in := []any{1, 2}
	if got := orEmptyList(in); len(got.([]any)) != 2 {
		t.Error("orEmptyList dropped a present value")
	}
}
