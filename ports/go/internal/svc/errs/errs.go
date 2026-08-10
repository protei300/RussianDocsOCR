// Package errs holds the service's error taxonomy.
//
// Deliberately few and deliberately specific. The distinction that earns its keep is
// TRANSIENT versus not: the worker has to tell "try again later" from "this document will
// never work", because retrying a corrupt JPEG forever is exactly as wrong as giving up on
// a transient CUDA hiccup.
//
// Port of service/ml/errors.py. Go has no exception hierarchy, so `transient` cannot be a
// class attribute — it is a function over the sentinel instead, and the SET of transient
// errors is written in one place rather than spread across type definitions.
package errs

import "errors"

// The sentinels. One per genuinely different caller reaction; see CONVENTIONS §D-02 for
// why the list is closed at seven and why there are no error graphs.
var (
	// ErrPipelineBusy: no pipeline became free within the lease timeout.
	//
	// Transient BY DEFINITION — the job goes back on the queue rather than being marked
	// failed. Seeing it repeatedly means a previous job wedged and never released its
	// lease, which is a real condition the status page must surface as `degraded`.
	ErrPipelineBusy = errors.New("pipeline busy")

	// ErrImageUnreadable: the uploaded bytes are not a decodable image.
	//
	// Deterministic. The same bytes fail the same way forever, so the worker must NOT
	// retry; a retry loop here burns the queue on a file that will never succeed.
	ErrImageUnreadable = errors.New("image unreadable")

	// ErrRuntimeNotReady: recognition was requested before the models finished loading.
	//
	// Transient, and an EXPECTED race rather than a fault: loading takes seconds and the
	// service accepts uploads immediately, which is the entire point of the queue.
	ErrRuntimeNotReady = errors.New("runtime not ready")

	// ErrNotFound: no such row. Distinct from an empty result, which is not an error.
	ErrNotFound = errors.New("not found")

	// ErrUnauthorized: no valid API key and no valid session.
	ErrUnauthorized = errors.New("unauthorized")

	// ErrConflict: the request is well-formed but contradicts the current state —
	// deleting the default API key is the case that exists today.
	ErrConflict = errors.New("conflict")

	// ErrBadRequest: malformed input that no retry or state change would fix.
	ErrBadRequest = errors.New("bad request")
)

// Transient reports whether retrying the same input could plausibly succeed.
//
// One function rather than a flag on each error, so the answer is auditable in one place.
// The DEFAULT IS FALSE, which is the safe direction: an unrecognised error retried
// forever is worse than one failed early, because the queue stops making progress and
// nothing in the log says why.
func Transient(err error) bool {
	switch {
	case errors.Is(err, ErrPipelineBusy), errors.Is(err, ErrRuntimeNotReady):
		return true
	default:
		return false
	}
}
