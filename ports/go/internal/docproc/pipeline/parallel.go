package pipeline

import (
	"golang.org/x/sync/errgroup"
)

// RunGroup runs a fixed set of tasks concurrently and collects their results
// POSITIONALLY.
//
// The shape of this function is mandated by CONVENTIONS §3 and must be preserved
// across ports: one launch statement per member in the reference's source order, one
// join, one deterministic collection indexed by position.
//
// Why positional collection and not a channel: Python's group collects with
// `futures[i].result()`, which is ordered by construction. A channel-collected version
// returns results in completion order, which varies with load — and that reorders
// boxes, reorders words, and changes the joined field string. It is an exact-match
// conformance failure with no float anywhere near it, and it only appears under
// concurrency, which is the worst way to find a bug.
//
// limit caps concurrency; pass 0 for unlimited. The word-splitting group uses
// min(8, n), matching the reference's ThreadPoolExecutor(max_workers=8).
//
// The first error wins and cancels the rest, which is errgroup's own semantic and maps
// onto C#'s aggregate-then-rethrow-first and Kotlin's coroutineScope.
//
// **On error the PARTIAL results are still returned**, and that is deliberate rather than
// sloppy. When T owns an unmanaged resource — the word-splitting group's T holds a list of
// crops — a task that succeeded before a sibling failed has already allocated, and swallowing
// its result makes cleanup impossible for the caller. Python did not need this because its GC
// collected the abandoned futures' results; a port that discards them leaks. Callers that
// ignore the results on error stay correct, so this cannot break the other group, whose T is
// a plain label. Ports must keep this: returning only `default` on failure is a leak by
// construction.
func RunGroup[T any](limit int, tasks []func() (T, error)) ([]T, error) {
	out := make([]T, len(tasks))
	var g errgroup.Group
	if limit > 0 {
		g.SetLimit(limit)
	}
	for i, task := range tasks {
		i, task := i, task
		g.Go(func() error {
			v, err := task()
			if err != nil {
				return err
			}
			// Written by index into a pre-sized slice: no append, no mutex, no
			// ordering question.
			out[i] = v
			return nil
		})
	}
	// `out`, not nil, on failure: whatever succeeded is the caller's to release.
	return out, g.Wait()
}

// MinLimit is min(cap, n), the idiom the reference uses when sizing a pool to the work
// available. Spelled out so the three ports read alike rather than each inlining it.
func MinLimit(cap, n int) int {
	if n < cap {
		return n
	}
	return cap
}
