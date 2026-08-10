package viewmodel

import (
	"math"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/tensor"
)

// FloatPrecision is the decimal places for every float on the wire.
//
// Rounding happens on the PRODUCING side, never in the checker. Past this precision float
// formatting differs between runtimes, so a golden comparison would fail for reasons that
// have nothing to do with recognition (spec/viewmodel.md, cross-language requirement 1).
const FloatPrecision = 4

// anglePrecision is finer because OBB angles are in radians, where 4 decimals is a
// visible rotation.
const anglePrecision = 6

// num rounds to the wire precision and returns nil for values JSON cannot carry.
//
// NaN and infinity become null rather than invalid JSON. Half-to-EVEN, matching Python's
// round() and np.round(); Go's math.Round is half-away-from-zero and would disagree on
// every tie (CONVENTIONS §6.5).
func num(v float64) *float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil
	}
	out := tensor.RoundHalfEven(v, FloatPrecision)
	return &out
}

// numAt is num with an explicit precision, for the radian angles.
func numAt(v float64, places int) *float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil
	}
	out := tensor.RoundHalfEven(v, places)
	return &out
}

// str returns a pointer so that "absent" is distinguishable from the empty string.
//
// Requirement 4 of the spec: emit null rather than omitting a key. A port with
// non-nullable fields would disagree with a Python dict that simply lacks the value, and
// the key-set comparison then fails for a reason nobody intended.
func str(s string) *string { return &s }

func intp(v int) *int { return &v }
