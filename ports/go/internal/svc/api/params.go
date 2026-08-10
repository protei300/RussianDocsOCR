package api

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"
)

// Query-parameter validation, matching the reference byte for byte.
//
// **This is the one place where the reference does NOT use `{"detail": "<string>"}`.** Every
// hand-written error in service/api/* raises HTTPException with a string, but a query parameter
// declared as `Query(1, ge=1, le=100)` is validated by FastAPI itself, and FastAPI answers with
// pydantic's own shape: `detail` is a LIST of objects. Reproducing that means reproducing an
// inconsistency in the reference — done deliberately, because a client parses what the server
// actually sends, and the reference is the contract.
//
// Captured from the running reference rather than written from memory:
//
//	GET /documents?page_size=500
//	422 {"detail":[{"type":"less_than_equal","loc":["query","page_size"],
//	                "msg":"Input should be less than or equal to 100",
//	                "input":"500","ctx":{"le":100}}]}
//	GET /documents?page_size=abc
//	422 {"detail":[{"type":"int_parsing","loc":["query","page_size"],
//	                "msg":"Input should be a valid integer, unable to parse string as an integer",
//	                "input":"abc"}]}
//
// Note `ctx` is absent for a parse failure and present for a bound, and that the ORDER matters:
// parsing is checked before bounds.
//
// This replaced silent clamping. The old behaviour answered 200 with 100 rows for
// `page_size=500`, so a client got a successful response to a request the reference rejects —
// invisible from the server side and impossible to notice without diffing the two.
// The clamp had an argument behind it ("the list page re-requests on every keystroke, and a
// 4xx mid-typing would flash an error"), and that argument was wrong twice over: the SPA holds
// `page`/`page_size` as numbers in its reactive model and can never send a malformed one, and
// what it does re-send per keystroke is `search`, a string with no validation at all.

// paramErrorItem is one pydantic validation entry.
type paramErrorItem struct {
	Type string `json:"type"`
	Loc  []any  `json:"loc"`
	Msg  string `json:"msg"`
	// Input is the RAW query string, not the parsed value — pydantic echoes what it was given,
	// which is why an out-of-range 500 comes back as the string "500".
	Input string `json:"input"`
	// Ctx is omitted entirely for a parse failure. A nil map disappears under omitempty, which
	// is exactly the reference's shape.
	Ctx map[string]int `json:"ctx,omitempty"`
}

// paramError is a query-parameter rejection. Carried as an error so handlers keep the single
// `if err != nil { writeError }` shape they use everywhere else.
type paramError struct{ item paramErrorItem }

func (e *paramError) Error() string { return e.item.Msg }

// queryInt reads a bounded integer query parameter the way the reference declares it.
//
// An ABSENT parameter yields the default; an EMPTY one (`?page_size=`) is a parse failure, which
// is what the reference does — verified, not assumed. Pass hi <= 0 for no upper bound, matching
// `page`, which declares ge=1 and no le.
func queryInt(q url.Values, name string, def, lo, hi int) (int, error) {
	if !q.Has(name) {
		return def, nil
	}
	raw := q.Get(name)
	v, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return 0, &paramError{paramErrorItem{
			Type:  "int_parsing",
			Loc:   []any{"query", name},
			Msg:   "Input should be a valid integer, unable to parse string as an integer",
			Input: raw,
		}}
	}
	// Bounds AFTER parsing, and ge before le, so the reported error is the same one the
	// reference reports when a value violates both.
	if v < lo {
		return 0, &paramError{paramErrorItem{
			Type:  "greater_than_equal",
			Loc:   []any{"query", name},
			Msg:   fmt.Sprintf("Input should be greater than or equal to %d", lo),
			Input: raw,
			Ctx:   map[string]int{"ge": lo},
		}}
	}
	if hi > 0 && v > hi {
		return 0, &paramError{paramErrorItem{
			Type:  "less_than_equal",
			Loc:   []any{"query", name},
			Msg:   fmt.Sprintf("Input should be less than or equal to %d", hi),
			Input: raw,
			Ctx:   map[string]int{"le": hi},
		}}
	}
	return v, nil
}
