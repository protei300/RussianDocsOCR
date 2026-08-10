package worker

import (
	"encoding/json"
	"sort"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/viewmodel"
)

// buildSearchText is the lowercased haystack for the list page's free-text search.
//
// Precomputed at write time so filtering never has to parse the stored result blob. In a SQL
// backend this becomes an indexed computed column.
//
// The OCR values are appended in SORTED KEY ORDER. Go randomises map iteration, and although
// the haystack is only ever substring-matched — so order cannot change a search RESULT — an
// unstable string would differ between two runs over the same document, which makes the
// stored records non-reproducible and any diff of them noise.
func buildSearchText(filename string, vm viewmodel.Payload) string {
	parts := []string{filename}
	if vm.DocType != nil {
		parts = append(parts, *vm.DocType)
	}

	keys := make([]string, 0, len(vm.Ocr))
	for k := range vm.Ocr {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		parts = append(parts, vm.Ocr[k])
	}

	if vm.Address != nil {
		for _, line := range vm.Address.Lines {
			if line.Text != nil {
				parts = append(parts, *line.Text)
			}
		}
	}
	return strings.ToLower(strings.Join(parts, " "))
}

// toMap converts the view model into the generic map the store persists.
//
// Via JSON rather than by hand, deliberately: the stored blob must be EXACTLY what the API
// serves, and a hand-written projection is a second definition of the wire format that would
// drift from the struct tags.
func toMap(vm viewmodel.Payload) (map[string]any, error) {
	data, err := json.Marshal(vm)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return out, nil
}
