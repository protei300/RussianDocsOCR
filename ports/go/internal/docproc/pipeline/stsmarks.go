package pipeline

import (
	"regexp"
	"strings"
	"unicode"
)

// Port of pipeline/sts_marks.py: the special marks of the vehicle registration certificate
// (STS) - words torn by a line break, and the leasing record.
//
// The special marks are printed by the registry's printer into a narrow area and wrapped at
// its edge WITHOUT a hyphen, so a word can be torn in two: «ЛИЗИ» at the end of one line,
// «НГА» at the start of the next. Measured on 140 real STS backs (external set, 2026-10-07):
// 4 such tears on known words - «ЛИЗИ|НГА», «ЛИЗИН|ГОДАТЕЛЬ», «КОМ|ПАНИЯ», «АВТОЛ|ИЗИНГ». The
// position of the line end does NOT tell a tear from a break between words (lines that tear
// end at 0.90-0.92 of the document's widest line, lines that do not at 0.87-1.00), so the
// tear is recognised by vocabulary: the two pieces are glued only when together they make a
// known word and neither piece is a known word on its own. Unknown words (a company name
// outside the list, a contract number) stay as read - a wrong glue would merge two real
// words, which is worse than leaving a torn one.
//
// Pure functions of strings, like dates.go.
//
// RE2 has no Unicode-aware \b, \w or lookaround, and Python's `re` has all three on a str:
// every pattern below that the reference writes with \b or \w is written here with the
// explicit Unicode class [\p{L}\p{N}_] and with the word boundary checked on the match by
// hand (see searchBounded) - the same text must give the same record in both.

// stsWords are words that stand as they are (no ending needed); stsStems only occur with an
// ending. Special-marks terms, the words seen on real backs (2026-10-07) and lessor names
// (public leasing companies). A word is known when it is a WORD, or a word or stem plus one of
// stsEndings: «ЛИЗИНГ» + «А», «РЕГИСТРАЦИ» + «И». A bare stem is not a word - «РЕГИСТРАЦИ» at a
// line end is a torn piece.
var stsWords = []string{
	"ЛИЗИНГОДАТЕЛЬ", "ЛИЗИНГОПОЛУЧАТЕЛЬ", "ЛИЗИНГ", "ДОГОВОР", "СОБСТВЕННИК",
	"ВЛАДЕЛЕЦ", "ДУБЛИКАТ", "ВЗАМЕН", "УВЭОС", "ВЫДАН", "УЧЕТ", "ТЕНТ",
	"ФУРГОН", "РЕФРИЖЕРАТОР", "ОБТЕКАТЕЛЬ", "МОЩНОСТЬ", "ДЕЙСТВИТЕЛЬНО",
	"ОБОРУДОВАНО", "УСТАНОВЛЕНО", "АДРЕС",
	"АВТОЛИЗИНГ", "ИНТЕРЛИЗИНГ", "РОСЛИЗИНГ", "ЕВРОПЛАН", "ГАЗПРОМБАНК",
	"СБЕРБАНК", "СОВКОМБАНК", "КАРКАДЕ", "АЛЬФАМОБИЛЬ", "МЭЙДЖОР", "ЭЛЕМЕНТ",
}

var stsStems = []string{
	"ЛИЗИНГОВ", "СОБСТВЕННОСТ", "ВЛАДЕЛЬЦ", "ОБЩЕСТВ", "ОГРАНИЧЕНН",
	"ОТВЕТСТВЕННОСТ", "АКЦИОНЕРН", "ПУБЛИЧН", "КОМПАНИ", "УТРАЧЕНН",
	"РЕГИСТРАЦИ", "ВРЕМЕНН", "ЗАМЕН", "СМЕН", "ИЗМЕНЕНИ", "ПЛАТФОРМ", "ВОРОТ",
	"ПОДРАЗДЕЛЕНИ", "КОНСТРУКЦИ", "БАЛТИЙСК",
}

// stsEndings are the noun and adjective endings of the case forms the marks use.
var stsEndings = []string{"А", "Я", "У", "Ю", "Е", "И", "Ы", "О", "Ь", "ОМ", "ЕМ", "ЁМ", "ОВ", "ЕВ", "АМ",
	"ЯМ", "АХ", "ЯХ", "ОЙ", "ЕЙ", "ИЙ", "ЫЙ", "АЯ", "ЯЯ", "ОЕ", "ЕЕ", "ЫЕ", "ИЕ",
	"ИЯ", "ИИ", "ИЮ", "ЬЮ", "ОЮ", "ЕЮ", "ОГО", "ЕГО", "ОМУ", "ЕМУ", "ЫМ", "ИМ",
	"ЫХ", "ИХ", "АМИ", "ЯМИ", "ЫМИ", "ИМИ"}

var stsLetters = regexp.MustCompile(`[^А-ЯЁA-Z0-9]`)

// stsClean is _clean: upper-case, keep only Cyrillic and Latin capitals and digits.
func stsClean(word string) string {
	return stsLetters.ReplaceAllString(strings.ToUpper(word), "")
}

func inList(list []string, v string) bool {
	for _, s := range list {
		if s == v {
			return true
		}
	}
	return false
}

// stsKnown is _known: a vocabulary word as is, or a word or stem with a case ending. The
// empty ending is NOT an ending (`'' in _ENDINGS` is False), so a bare stem is not known.
func stsKnown(word string) bool {
	if inList(stsWords, word) {
		return true
	}
	for _, bases := range [][]string{stsWords, stsStems} {
		for _, base := range bases {
			if strings.HasPrefix(word, base) && inList(stsEndings, word[len(base):]) {
				return true
			}
		}
	}
	return false
}

// GlueTornWords is glue_torn_words: the words of a multi-line field, line by line -> the
// words with tears glued.
//
// lines is a list of lines, each a list of the words read on it, top to bottom. The last word
// of a line and the first of the next are glued when together they are a known word and
// neither alone is: «ЛИЗИ» + «НГА» -> «ЛИЗИНГА», while «ЛИЗИНГОДАТЕЛЬ» + «АО» stay two words.
//
// Deliberately NOT glued by the length of the line, although the registry does wrap by
// character count: on 140 real backs (2026-10-07) the rule «a full line goes on into the next
// one» glued about half of its cases wrongly («КРОНШ.» + «БАЗЫ», «КВТ» + «Л.С», a date line
// onto the line above) - the width differs between documents (lines of about 26 characters on
// some, 40 and more on others) and the reading of real marks is often noisy.
func GlueTornWords(lines [][]string) []string {
	out := []string{}
	for _, line := range lines {
		var words []string
		for _, w := range line {
			if w != "" {
				words = append(words, w)
			}
		}
		if len(words) == 0 {
			continue
		}
		if len(out) > 0 {
			tail, head := stsClean(out[len(out)-1]), stsClean(words[0])
			if tail != "" && head != "" && stsKnown(tail+head) && !stsKnown(tail) && !stsKnown(head) {
				out[len(out)-1] = out[len(out)-1] + words[0]
				words = words[1:]
			}
		}
		out = append(out, words...)
	}
	return out
}

// ---------------------------------------------------------------- leasing

// Leasing is parse_leasing's record. A part that is not printed or not found is "" (None in
// the reference). Only the flag reaches Results.Leasing (LeasingReported).
type Leasing struct {
	Leasing bool
	// Role is "lessor_named" when the marks name the lessor («ЛИЗИНГОДАТЕЛЬ ...») and
	// "lessee" when they only say the owner is the lessee («ЛИЗИНГОПОЛУЧАТЕЛЬ»).
	Role                   string
	Lessor                 string
	ContractNumber         string
	ContractDate           string
	ContractDateNormalized string
	// Until is the end of the leasing term («ЛИЗИНГ ДО 31.12.2027»).
	Until           string
	UntilNormalized string
}

const wordClass = `[\p{L}\p{N}_]`

var (
	stsNumber = regexp.MustCompile(`№\s*([0-9A-ZА-ЯЁ][0-9A-ZА-ЯЁ/\-]*)`)
	// «ДОГ ЛИЗ АХ ЭЛ/УЛН-123/ДЛ», «ПО ДОГ ЛИЗИНГА 12/34-СКТ»: the number is the first token with
	// a digit after the abbreviated «договор лизинга», within three tokens. Start: \b.
	stsAbbrNumber = regexp.MustCompile(`ДОГ` + wordClass + `*\.?\s+(?:ЛИЗ` + wordClass + `*\.?\s+)?((?:\S+\s+){0,2}?\S*[0-9]\S*)`)
	stsDate       = regexp.MustCompile(`([0-9]{1,2}[.,][0-9]{1,2}[.,][0-9]{4})`)
	// the 2010 edition: «<nn> <nnnn> Л.Д <дата>». Start and end: \b.
	stsShortLease = regexp.MustCompile(`Л\s*\.\s*Д`)
	// «по договору лизинга». Start and end: \b.
	stsLeaseContract = regexp.MustCompile(`ПО\s+ДЛ`)
	// Start: \b.
	stsUntil = regexp.MustCompile(`ЛИЗИНГ` + wordClass + `*\s+(?:ДЕЙСТВ` + wordClass + `*\s+)?ДО[^\p{L}\p{N}_]{0,2}([0-9]{1,2}[.,][0-9]{1,2}[.,][0-9]{4})`)
	// Start: \b.
	stsFrom = regexp.MustCompile(`ОТ\s+([0-9]{1,2}[.,][0-9]{1,2}[.,][0-9]{4})`)
)

func isWordRune(r rune) bool { return r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r) }

// searchBounded finds the first match of re in s at or after byte offset from whose START
// stands on a word boundary (the character before it is not a word character, or it is the
// start of the text) and, when endB is set, whose END does too (the character after it is
// not a word character, or it is the end). It stands in for the `\b` the reference writes in
// front of (and behind) these patterns.
//
// A boundary is a property of the position, not of the way the match was found, so skipping
// an invalid start and searching on from the next character gives the same leftmost valid
// match a backtracking engine would.
func searchBounded(re *regexp.Regexp, s string, from int, endB bool) []int {
	for from <= len(s) {
		loc := re.FindStringSubmatchIndex(s[from:])
		if loc == nil {
			return nil
		}
		for i := range loc {
			if loc[i] >= 0 {
				loc[i] += from
			}
		}
		start, end := loc[0], loc[1]
		okStart := true
		if start > 0 {
			prev, _ := lastRune(s[:start])
			okStart = !isWordRune(prev)
		}
		okEnd := true
		if endB && end < len(s) {
			next := firstRune(s[end:])
			okEnd = !isWordRune(next)
		}
		if okStart && okEnd {
			return loc
		}
		// resume one character after this start
		_, size := firstRuneSize(s[start:])
		from = start + size
	}
	return nil
}

func firstRune(s string) rune { r, _ := firstRuneSize(s); return r }

func firstRuneSize(s string) (rune, int) {
	for _, r := range s {
		return r, len(string(r))
	}
	return 0, 0
}

func lastRune(s string) (rune, int) {
	rs := []rune(s)
	if len(rs) == 0 {
		return 0, 0
	}
	r := rs[len(rs)-1]
	return r, len(string(r))
}

// stsLessorEnd is the START of the first match of _LESSOR_END in s, or -1:
//
//	[.,;(] | № | \bДОГ\w* | \bПО\s+ДЛ\b | ЛИЗИНГОПОЛУЧАТЕЛЬ | \bВРЕМ\w* | \bДЕЙСТВ\w* |
//	\bМОЩНОСТ\w* | \bСРОК\w* | \bДО\b
//
// what follows the lessor's name on real marks: the contract, the lessee, the registration
// term, the validity, the next mark (engine power ...). Only the start of the match matters
// - the reference takes the text before it - so each alternative is looked up on its own and
// the earliest start wins.
func stsLessorEnd(s string) int {
	best := -1
	take := func(i int) {
		if i >= 0 && (best < 0 || i < best) {
			best = i
		}
	}
	take(strings.IndexAny(s, ".,;("))
	take(strings.Index(s, "№"))
	take(strings.Index(s, "ЛИЗИНГОПОЛУЧАТЕЛЬ"))
	for _, prefix := range []string{"ДОГ", "ВРЕМ", "ДЕЙСТВ", "МОЩНОСТ", "СРОК"} {
		off := 0
		for {
			i := strings.Index(s[off:], prefix)
			if i < 0 {
				break
			}
			i += off
			if atWordStart(s, i) {
				take(i)
				break
			}
			off = i + 1
		}
	}
	// \bПО\s+ДЛ\b
	if loc := searchBounded(stsLeaseContract, s, 0, true); loc != nil {
		take(loc[0])
	}
	// \bДО\b
	off := 0
	for {
		i := strings.Index(s[off:], "ДО")
		if i < 0 {
			break
		}
		i += off
		end := i + len("ДО")
		if atWordStart(s, i) && (end >= len(s) || !isWordRune(firstRune(s[end:]))) {
			take(i)
			break
		}
		off = i + 1
	}
	return best
}

func atWordStart(s string, i int) bool {
	if i == 0 {
		return true
	}
	prev, _ := lastRune(s[:i])
	return !isWordRune(prev)
}

func normalLeasingDate(dateText string) string {
	if dateText == "" {
		return ""
	}
	return ToDdmmyyyy(strings.ReplaceAll(dateText, ",", "."))
}

// ParseLeasing is parse_leasing: the leasing record in the special marks, or nil when there
// is none.
//
// The lessee's own name is never taken out: it is the owner, already read. The text is the
// reading AFTER GlueTornWords: a torn «ЛИЗИ НГ» is not found. Never guesses: no leasing word
// -> nil, and every part is taken only where the marks print it. Real marks are abbreviated
// freely and read noisily, so expect the flag far more often than the parts (140 real backs,
// 2026-10-07).
func ParseLeasing(text string) *Leasing {
	if text == "" {
		return nil
	}
	t := strings.Join(strings.Fields(strings.ReplaceAll(strings.ToUpper(text), "Ё", "Е")), " ")
	if !(strings.Contains(t, "ЛИЗИНГ") ||
		searchBounded(stsShortLease, t, 0, true) != nil ||
		searchBounded(stsLeaseContract, t, 0, true) != nil) {
		return nil
	}

	rec := &Leasing{Leasing: true}
	switch {
	case strings.Contains(t, "ЛИЗИНГОДАТЕЛЬ"):
		rec.Role = "lessor_named"
		after := strings.SplitN(t, "ЛИЗИНГОДАТЕЛЬ", 2)[1]
		if end := stsLessorEnd(after); end >= 0 {
			after = after[:end]
		}
		rec.Lessor = strings.Trim(after, " :-") // quotes stay
	case strings.Contains(t, "ЛИЗИНГОПОЛУЧАТЕЛЬ"):
		rec.Role = "lessee"
	}

	if m := stsNumber.FindStringSubmatchIndex(t); m != nil {
		rec.ContractNumber = strings.TrimSpace(t[m[2]:m[3]])
	} else if m := searchBounded(stsAbbrNumber, t, 0, false); m != nil {
		rec.ContractNumber = strings.TrimSpace(t[m[2]:m[3]])
	}

	dateText := ""
	if m := searchBounded(stsFrom, t, 0, false); m != nil {
		dateText = t[m[2]:m[3]]
	} else if short := searchBounded(stsShortLease, t, 0, true); short != nil {
		if d := stsDate.FindStringSubmatchIndex(t[short[1]:]); d != nil {
			dateText = t[short[1]+d[2] : short[1]+d[3]]
		}
	}
	rec.ContractDate = dateText
	rec.ContractDateNormalized = normalLeasingDate(dateText)

	if m := searchBounded(stsUntil, t, 0, false); m != nil {
		rec.Until = t[m[2]:m[3]]
	}
	rec.UntilNormalized = normalLeasingDate(rec.Until)
	return rec
}

// LeasingReported are the parts of ParseLeasing that reach Results.Leasing (Pipeline.
// LEASING_REPORTED): only the flag, by measurement (STS word-break synthetic, 240 shots,
// detector v10, 2026-10-07): found 54/64, false 0/176 - while the lessor was right in 7/58,
// the contract number in 3/56, its date in 2/46. The parser finds them where the reading is
// clean; the reading of the small special-marks print is not, and a wrong value is worse than
// none for an integrator. Widen this when the reading improves - the parser needs no change.
var LeasingReported = []string{"leasing"}

// reportedLeasing is `{k: v for k, v in leasing.items() if k in LEASING_REPORTED}`.
func reportedLeasing(rec *Leasing) map[string]any {
	out := map[string]any{}
	for _, key := range LeasingReported {
		if key == "leasing" {
			out[key] = rec.Leasing
		}
	}
	return out
}

// glueTorn is Pipeline._glue_torn: glue the words a line break tore apart (GlueTornWords),
// for the fields the options name. The words come flat; the line lengths recorded by
// SplitWords (FieldWords.Lines) cut them back into lines. Words are returned unchanged when
// the field is not named, or when the counts no longer add up to the words read (a word that
// reached no engine).
func glueTorn(fw FieldWords, words []string, opts OcrOptions) []string {
	if !contains(opts.GlueTorn, fw.Label) {
		return words
	}
	total := 0
	for _, n := range fw.Lines {
		total += n
	}
	if len(fw.Lines) == 0 || total != len(words) {
		return words
	}
	lines := make([][]string, 0, len(fw.Lines))
	start := 0
	for _, n := range fw.Lines {
		lines = append(lines, words[start:start+n])
		start += n
	}
	return GlueTornWords(lines)
}
