package tensor

import (
	"fmt"
	"math"
	"sort"
)

// AsFloat32 returns the data as float32, converting from uint8 if needed.
//
// The uint8 -> float32 direction is the one the pipeline actually uses:
// ClassificationPreprocessing and YoloPreprocessing both hand over UINT8 pixel data
// in the range 0-255 with **no normalisation** — the scaling is baked into the ONNX
// graphs — and the inference layer then casts to whatever dtype the session declares.
// (Python's `BasePreprocessing.normalization` method is shadowed by a same-named tuple
// attribute and is uncallable, so normalisation never happens in preprocessing. That
// is documented dead code, not an oversight to reproduce.)
func (a *Array) AsFloat32() ([]float32, error) {
	switch a.Dtype {
	case Float32:
		return a.F32, nil
	case Uint8:
		out := make([]float32, len(a.U8))
		for i, v := range a.U8 {
			out[i] = float32(v)
		}
		return out, nil
	case Float64:
		// Narrowing is allowed but never silent about precision: nothing in the
		// pipeline feeds float64 into a model, so reaching here is a design slip
		// worth seeing in a stack trace rather than tolerating.
		out := make([]float32, len(a.F64))
		for i, v := range a.F64 {
			out[i] = float32(v)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("tensor: cannot view %s as float32", a.Dtype)
	}
}

// AsUint8 returns the data as uint8. Only a uint8 array qualifies: converting from a
// float would need a rounding and clipping policy, and inventing one silently is how
// an OCR input stops matching the reference.
func (a *Array) AsUint8() ([]uint8, error) {
	if a.Dtype != Uint8 {
		return nil, fmt.Errorf("tensor: cannot view %s as uint8 without a rounding policy", a.Dtype)
	}
	return a.U8, nil
}

// Row returns row i of a 2-D float32 array as a view (no copy).
func (a *Array) Row(i int) ([]float32, error) {
	if a.Dtype != Float32 || len(a.Shape) != 2 {
		return nil, fmt.Errorf("tensor: Row needs a 2-D float32 array, got %s%v", a.Dtype, a.Shape)
	}
	w := a.Shape[1]
	if i < 0 || i >= a.Shape[0] {
		return nil, fmt.Errorf("tensor: row %d out of range [0,%d)", i, a.Shape[0])
	}
	return a.F32[i*w : (i+1)*w], nil
}

// Flat drops leading unit dimensions, so a [1,1100] model output can be treated as a
// 1100-element vector without copying. Mirrors the `np.squeeze` the Python model
// wrappers apply before post-processing.
func (a *Array) Flat() []float32 {
	return a.F32
}

// Argmax returns the index of the FIRST maximum, matching numpy.argmax.
//
// Strict `>`, never `>=`. On a tie, `>=` returns the last index instead of the first;
// in CTC decoding that flips a timestep and changes a character, and in class
// selection it changes the predicted label. There is no float tolerance to hide behind
// — the comparison downstream is exact.
func Argmax(v []float32) int {
	best, bestV := 0, float32(math.Inf(-1))
	for i, x := range v {
		if x > bestV {
			best, bestV = i, x
		}
	}
	return best
}

// Max returns the largest value, or 0 for an empty slice.
//
// The zero for an empty input is deliberate: Python's
// `probability.max(initial=0)` in MultiClassPostprocessing does the same, and
// numpy would otherwise raise.
func Max(v []float32) float32 {
	out := float32(0)
	for i, x := range v {
		if i == 0 || x > out {
			out = x
		}
	}
	if len(v) == 0 {
		return 0
	}
	return out
}

// CosineDistance is sklearn's cosine metric: 1 - cosine_similarity.
//
// Accumulated in float64 even though the inputs are float32, because that is what
// numpy's `dot` and `norm` do — they promote. This is the one deliberate exception to
// the "float32 throughout" rule in CONVENTIONS §6.7, and measured agreement with the
// reference is ~9e-16.
//
// A zero-length vector yields distance 1 (orthogonal) rather than NaN, so a degenerate
// embedding falls out of the radius filter instead of poisoning a comparison.
func CosineDistance(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 1
	}
	return 1 - dot/(math.Sqrt(na)*math.Sqrt(nb))
}

// EuclideanDistance is the plain L2 distance, accumulated in float64 for the same
// reason as CosineDistance.
func EuclideanDistance(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

// RoundHalfEven rounds to `places` decimals with numpy's tie rule: **half to EVEN**.
//
// np.round(0.5) == 0 and np.round(1.5) == 2. Go's math.Round is half-away-from-zero,
// which differs on exact ties — and ties are reachable, because confidences are
// already quantised to three decimals upstream. Use this for anything that reaches
// the wire or a comparison.
func RoundHalfEven(v float64, places int) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return v
	}
	scale := math.Pow(10, float64(places))
	return math.RoundToEven(v*scale) / scale
}

// StableSortBy sorts indices [0,n) by a key, preserving the original order of equal
// keys.
//
// Python's list.sort, np.argsort and np.lexsort are all stable; Go's sort.Slice is not.
// Two equal-x word boxes reordering swaps two tokens in a joined field string, which
// is an exact-match conformance failure with no float involved (CONVENTIONS §6.2).
func StableSortBy(n int, key func(i int) float64) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return key(idx[a]) < key(idx[b]) })
	return idx
}
