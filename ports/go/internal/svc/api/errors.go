// Package api is the HTTP surface.
//
// **This file comes first, and the order is not arbitrary.** Every constraint it fixes is
// inherited by every other handler, and each one is a real client dependency rather than a
// style choice:
//
//   - the error body is `{"detail": "<string>"}` — what FastAPI produces and what the SPA's
//     fetch wrapper reads;
//   - a missing credential is **401**, not 403. 403 means "authenticated but not allowed",
//     and the SPA redirects to the PIN screen on 401 only;
//   - DELETE returns **204 with an empty body**. A JSON body on a 204 is a protocol error
//     that some clients reject outright;
//   - POST /documents returns **202 with the full list row**, so the SPA can insert the row
//     without a second request;
//   - /progress returns **200 with a JSON `null`**, never 404 — a finished document is not a
//     missing one, and a 404 there makes the SPA drop the row;
//   - images carry `Cache-Control: private, no-cache` and are fetched with an Authorization
//     header, never a token in the query string, because a query token lands in logs and
//     browser history;
//   - the list filter parameter is named `status`.
//
// Port of the FastAPI conventions in service/api/*.
package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/errs"
	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/settingsschema"
)

// errorBody is the one error shape. `detail` is a STRING, not an object: FastAPI's
// HTTPException produces exactly this, and the SPA reads `detail` directly.
type errorBody struct {
	Detail string `json:"detail"`
}

// clientErr carries a CLIENT-FACING message alongside a sentinel.
//
// `fmt.Errorf("%w: msg", sentinel)` is the wrong tool here, and it shipped a real defect
// before this existed: Error() then returns "conflict: The default key ..." and that whole
// string went into the response body, so the client saw an internal sentinel name. This type
// keeps the two separate — Error() is the message the user reads, Unwrap() is what errors.Is
// matches on for the status code.
type clientErr struct {
	sentinel error
	msg      string
}

func (e *clientErr) Error() string { return e.msg }
func (e *clientErr) Unwrap() error { return e.sentinel }

// clientError builds one. Use it for every error whose text reaches a response.
func clientError(sentinel error, format string, args ...any) error {
	return &clientErr{sentinel: sentinel, msg: fmt.Sprintf(format, args...)}
}

// writeJSON emits a payload with the correct content type.
//
// Encoding into a buffer first would let a marshalling failure be reported properly, but the
// payloads here are all plain structs and maps; streaming keeps the memory profile flat for
// a 100 KB result blob. A late failure logs and truncates, which is visible in the client as
// a parse error — acceptable, and better than buffering every response.
func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if payload == nil {
		return
	}
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		slog.Error("[API] response encode failed", "err", err)
	}
}

// writeError maps an error to a status and the `detail` body.
//
// The mapping lives HERE and nowhere else, so a handler never picks a status code: that is
// what keeps 401-versus-403 and 409-versus-400 consistent across a dozen endpoints.
func writeError(w http.ResponseWriter, err error) {
	// A query-parameter rejection is the ONE case whose body is not `{"detail": "<string>"}`:
	// FastAPI generates it from pydantic and `detail` is a list. See params.go for the captured
	// reference responses and why the inconsistency is reproduced rather than smoothed over.
	var param *paramError
	if errors.As(err, &param) {
		writeJSON(w, http.StatusUnprocessableEntity,
			map[string]any{"detail": []paramErrorItem{param.item}})
		return
	}
	status, detail := classify(err)
	writeJSON(w, status, errorBody{Detail: detail})
}

func classify(err error) (int, string) {
	var validation *settingsschema.ValidationError
	switch {
	case errors.Is(err, errs.ErrNotFound):
		return http.StatusNotFound, "Not found"
	case errors.Is(err, errs.ErrUnauthorized):
		// 401, NOT 403 — see the package note.
		return http.StatusUnauthorized, "Not authenticated"
	case errors.Is(err, errs.ErrConflict):
		return http.StatusConflict, err.Error()
	case errors.As(err, &validation):
		// **400, not 422.** FastAPI's own validation errors are 422, which is why 422 looks
		// right here — but the reference raises HTTPException(400) for a rejected SETTING,
		// and the reference is the contract. The message passes through because it names the
		// bound that was violated, which is the only useful thing a settings form can show.
		return http.StatusBadRequest, err.Error()
	case errors.Is(err, errs.ErrBadRequest):
		return http.StatusBadRequest, err.Error()
	case errors.Is(err, errs.ErrImageUnreadable):
		return http.StatusUnprocessableEntity, err.Error()
	case errors.Is(err, errs.ErrRuntimeNotReady), errors.Is(err, errs.ErrPipelineBusy):
		return http.StatusServiceUnavailable, err.Error()
	default:
		// The message is NOT echoed for an unclassified error: it may carry a path or an
		// internal detail, and the log already has it in full.
		slog.Error("[API] unhandled error", "err", err)
		return http.StatusInternalServerError, "Internal server error"
	}
}

// writeNoContent is the DELETE response: 204 and NOTHING. Not an empty JSON object — a body
// on a 204 is a protocol violation some clients reject.
func writeNoContent(w http.ResponseWriter) { w.WriteHeader(http.StatusNoContent) }

// methodNotAllowed keeps the error shape uniform even for routing failures, which otherwise
// come back as Go's plain-text default and break a client that always parses JSON.
func methodNotAllowed(w http.ResponseWriter, allowed string) {
	w.Header().Set("Allow", allowed)
	writeJSON(w, http.StatusMethodNotAllowed, errorBody{Detail: "Method not allowed"})
}
