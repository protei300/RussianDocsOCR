package viewmodel

import "sort"

// buildFields renders the recognised fields as an ORDERED ARRAY, each linked to its boxes.
// Port of transform.py::_build_fields.
//
// An array and not a map, deliberately. Three problems are solved by it, and all three
// return the moment a port "simplifies" it (spec/viewmodel.md):
//
//  1. ASSOCIATION. Matching a field to a box by string equality of the label is ambiguous
//     — see the `ambiguous` flag in boxes.go.
//  2. ORDER. A JSON object has none, and insertion order is not document reading order.
//  3. RENDERING. `script` selects proportional versus monospace type, which the UI cannot
//     infer from the value.
func buildFields(docType string, ocr map[string]string, boxes []Box) []Field {
	byLabel := map[string][]string{}
	confByLabel := map[string]*float64{}
	for _, b := range boxes {
		byLabel[b.Label] = append(byLabel[b.Label], b.ID)
		// The confidence reported for a field is that of the box OWNING its text, which
		// is why this is gated on Text rather than taking the last or the maximum.
		if b.Text != nil {
			confByLabel[b.Label] = b.Conf
		}
	}

	names := make([]string, 0, len(ocr))
	for name := range ocr {
		names = append(names, name)
	}
	// Sorted before ordering: Go randomises map iteration, so without this the
	// alphabetical tail of OrderFields would be fed a different sequence each run. The
	// result would still be deterministic for KNOWN fields and quietly unstable for
	// unknown ones — the worst combination to debug.
	sort.Strings(names)

	ordered := OrderFields(docType, names)
	out := make([]Field, 0, len(ordered))
	for _, name := range ordered {
		var value *string
		if v, ok := ocr[name]; ok {
			value = str(v)
		}
		ids := byLabel[name]
		if ids == nil {
			// An empty ARRAY, not null: a field can legitimately have no box (the
			// synthetic Address keys have none).
			ids = []string{}
		}
		out = append(out, Field{
			Name:    name,
			Display: FieldDisplay(name),
			Value:   value,
			Script:  FieldScript(name),
			Conf:    confByLabel[name],
			BoxIds:  ids,
		})
	}
	return out
}
