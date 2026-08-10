package modules

import "testing"

func TestCheckDdmmyyyy(t *testing.T) {
	cases := []struct{ in, want, why string }{
		{"06011985", "06.01.1985", "eight digits reformat"},
		{"06.01.1985", "06.01.1985", "already formatted, unchanged"},
		{"06-01-1985", "06.01.1985", "hyphens become dots"},
		{"O6.O1.1985", "06.01.1985", "uppercase O is a misread zero"},
		{"123", "123", "too few digits: the substituted string, not padded"},
		{"060119851", "060119851", "too many digits: the substituted string"},
		{"32011985", "32011985", "impossible DAY: strptime raises, the ORIGINAL is kept"},
		{"06131985", "06131985", "impossible MONTH: same"},
		{"29022019", "29022019", "not a leap year: the original is kept"},
		{"29022020", "29.02.2020", "a real leap day: formatted"},
		{"", "", "empty stays empty"},
		// The discriminating case, verified against the reference: eight digits but an
		// impossible month, so the substitutions are DISCARDED along with the exception --
		// the O and the hyphens both survive. A port that returned its local substituted
		// copy here would differ, and only on malformed input.
		{"O6-13-1985", "O6-13-1985", "invalid date discards the substitutions too"},
	}
	for _, c := range cases {
		if got := CheckDdmmyyyy(c.in); got != c.want {
			t.Errorf("CheckDdmmyyyy(%q) = %q, want %q (%s)", c.in, got, c.want, c.why)
		}
	}
}

// The lowercase 'o' is deliberately NOT substituted: the reference replaces 'O' only, and
// the printed models emit uppercase.
func TestCheckDdmmyyyyLeavesLowercaseOAlone(t *testing.T) {
	if got := CheckDdmmyyyy("o6.o1.1985"); got != "o6.o1.1985" {
		t.Fatalf("got %q; lowercase o must not be treated as a zero", got)
	}
}

// Both sex helpers are ASYMMETRIC: anything without the male letter becomes female,
// including the empty string. That is behaviour, not an oversight -- the field is binary
// and the box only fires on a sex field.
func TestSexHelpersAreAsymmetric(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"M", "M"}, {"m", "M"}, {".M.", "M"}, {"F", "F"}, {"X", "F"}, {"", "F"},
	} {
		if got := CheckEnSex(c.in); got != c.want {
			t.Errorf("CheckEnSex(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	for _, c := range []struct{ in, want string }{
		{"М", "М"}, {"м", "М"}, {".М", "М"}, {"Ж", "Ж"}, {"Щ", "Ж"}, {"", "Ж"},
	} {
		if got := CheckRusSex(c.in); got != c.want {
			t.Errorf("CheckRusSex(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// The Cyrillic result is CYRILLIC М (U+041C), not Latin M (U+004D). They are visually
// identical and a Latin M would silently fail every downstream comparison.
func TestCheckRusSexReturnsCyrillicLetter(t *testing.T) {
	got := []rune(CheckRusSex("М"))
	if len(got) != 1 || got[0] != 'М' {
		t.Fatalf("got %U, want U+041C (Cyrillic М)", got)
	}
	// A LATIN M in the input contains no Cyrillic М, so it reads as female. Confirms the
	// two alphabets are not being conflated.
	if CheckRusSex("M") != "Ж" {
		t.Fatal("a Latin M must not satisfy the Cyrillic test")
	}
}

func TestCheckDriverClass(t *testing.T) {
	cases := []struct{ in, want string }{
		{"B B1 C C1 D D1 CE C1E", "BB1CC1DD1CEC1E"},
		{"A,B;C", "ABC"},
		{"XYZ", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := CheckDriverClass(c.in); got != c.want {
			t.Errorf("CheckDriverClass(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// Leading only, despite the name -- Python's lstrip. A trailing dot survives.
func TestStripEdgeDotsIsLeadingOnly(t *testing.T) {
	if got := StripEdgeDots("..ИВАНОВ."); got != "ИВАНОВ." {
		t.Fatalf("got %q, want %q", got, "ИВАНОВ.")
	}
}
