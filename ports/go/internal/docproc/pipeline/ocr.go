package pipeline

import (
	"regexp"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/modules"
)

// FieldText is one recognised field: the per-word strings and the joined value.
//
// A SLICE, not a map, because the field order is part of the output contract — it decides
// the order of the view model's `fields` array and of the service's search text. Go
// randomises map iteration, so an order that lives only in a map is not an order
// (CONVENTIONS §1).
// rulerRuns matches the dotted ruler lines the 1998 birth-certificate form prints
// under every value; they land inside the field crops and OCR emits runs of
// dots/dashes/underscores around the real words.
var rulerRuns = regexp.MustCompile(`[._\-]{2,}`)

// CleanRulerArtifacts collapses ruler-dot runs out of a joined field value.
// Port of Pipeline._clean_ruler_artifacts (pipeline.py:1061-1076). The reference
// also drops LONE separators with a lookaround pattern Go's RE2 cannot express
// ((?:^|(?<=\s))[._\-](?=\s|$)); splitting into whitespace tokens, dropping
// tokens that are exactly one separator rune and re-joining is equivalent,
// because the reference finishes by collapsing all whitespace and trimming.
// Single in-word dots - the digit birth date, abbreviations - stay untouched,
// exactly as in the reference.
func CleanRulerArtifacts(value string) string {
	t := rulerRuns.ReplaceAllString(value, " ")
	fields := strings.Fields(t)
	kept := fields[:0]
	for _, tok := range fields {
		if tok == "." || tok == "_" || tok == "-" {
			continue
		}
		kept = append(kept, tok)
	}
	return strings.Join(kept, " ")
}

type FieldText struct {
	Label string
	Words []string
	Value string
}

// OcrFields recognises every field's words and joins them.
// Port of Pipeline._ocr_serial (pipeline.py:985-1027).
//
// docType is the label with its year suffix already stripped, which matters: the routing
// below tests `docType == "SNILS"` against the bare type.
func OcrFields(fields []FieldWords, docType string, opts OcrOptions,
	cyr, lat *modules.OcrEngine) ([]FieldText, error) {

	out := make([]FieldText, 0, len(fields))
	for _, fw := range fields {
		var words []string
		for i, patch := range fw.Patches {
			// Three branches, and the FIRST one carries a precedence subtlety worth
			// spelling out: Python's `doc_type == 'SNILS' and i % 2 == 1 or field_name in
			// ru_fields` binds `and` tighter than `or`, so it reads
			// `(SNILS and odd) or (in ru_fields)`.
			//
			// SNILS is the one type where a WORD-INDEX PARITY check decides the engine
			// regardless of field name: its dates read "31 октября 1998", Russian month
			// names interleaved with digits, so odd-indexed words must go to the Cyrillic
			// engine even though the field itself is date-routed below.
			switch {
			case (docType == "SNILS" && i%2 == 1) || contains(opts.RuFields, fw.Label):
				text, err := cyr.Predict(patch)
				if err != nil {
					return nil, err
				}
				words = append(words, cyr.FixErrors(fw.Label, text))

			case strings.Contains(strings.ToLower(fw.Label), "date"):
				text, err := lat.Predict(patch)
				if err != nil {
					return nil, err
				}
				words = append(words, lat.FixErrors(fw.Label, text))

			case contains(opts.EnFields, fw.Label):
				text, err := lat.Predict(patch)
				if err != nil {
					return nil, err
				}
				words = append(words, lat.FixErrors(fw.Label, text))

				// No default: a field in neither list contributes NO word, and the field
				// still appears with an empty value. That is the reference's behaviour --
				// the loop simply skips -- and it is reachable, because SplitWords admits a
				// field only if it is in one of the lists, but the parity branch above can
				// leave an even-indexed SNILS word unmatched.
			}
		}
		out = append(out, FieldText{Label: fw.Label, Words: words})
	}

	// Joining is separate from recognition so the per-word strings survive for the
	// ocr.<Field>.words stage, which is what localises a single bad word.
	joined := map[string]string{}
	for i := range out {
		out[i].Value = joinField(joined, out[i].Label, docType, out[i].Words)
	}
	return out, nil
}

// joinField assembles a field's final string.
// Port of Pipeline._join_field (pipeline.py:1095-1107).
//
// Three separators for three cases, and the SNILS exception is not cosmetic: an ordinary
// date joins with '.' to give "01.02.1998", but a SNILS date is words ("31 октября 1998")
// and would become "31.октября.1998".
//
// `joined` accumulates across calls because a field detected twice appends with a space
// rather than replacing -- the reference relies on the dict already holding a value.
func joinField(joined map[string]string, label, docType string, words []string) string {
	isDate := strings.Contains(strings.ToLower(label), "date")

	var value string
	switch {
	case isDate && docType != "SNILS":
		value = strings.Join(words, ".")
	case isDate && docType == "SNILS":
		value = strings.Join(words, " ")
	default:
		if prev := joined[label]; prev != "" {
			value = prev + " " + strings.Join(words, " ")
		} else {
			value = strings.Join(words, " ")
		}
	}

	// A SINGLE pass of "  " -> " ", not a loop and not a regex: Python's str.replace is
	// one pass too, so three consecutive spaces leave one behind in both. Collapsing
	// fully here would produce a different string.
	value = strings.TrimSpace(strings.ReplaceAll(value, "  ", " "))
	joined[label] = value
	return value
}

// FixFms is a deliberate no-op, carried over rather than dropped.
//
// It used to rewrite Issue_organisation_code/Issue_organization_ru from the FMS
// dictionary. Disabled in the reference for two reasons worth keeping next to the code:
//
//   - COST. An exactly-read code is an O(1) lookup, but a single misread character falls
//     through to a difflib scan of the whole ~16k-entry dictionary — measured at 3.3-5.1 s
//     for ONE document. That is the entire reason INTPASSPORT_1997/15_CR_INTPASSPORT_2001
//     took 3.8 s while every other sample took ~0.4 s.
//   - SOUNDNESS. On that fall-through the dictionary does not correct the code, it
//     REPLACES it with the code of whichever authority name scored highest — so a misread
//     digit silently becomes a confident, well-formed, wrong code.
//
// Kept as a no-op so re-enabling is one line and the reasoning stays attached. The whole
// FMS dictionary is otherwise not ported (it has no live callers).
func FixFms(_ []FieldText, _ string) {}
