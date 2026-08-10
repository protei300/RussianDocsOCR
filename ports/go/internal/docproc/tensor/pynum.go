package tensor

import "math"

// FloorDiv reproduces CPython's float `//` operator.
//
// It is NOT `math.Floor(x / y)`, and assuming it is causes a real, reproducible
// divergence. CPython implements float floor division via `fmod` (see
// `float_divmod` in Objects/floatobject.c):
//
//	mod      = fmod(x, y)
//	div      = (x - mod) / y
//	floordiv = floor(div)          plus a half-ulp nudge
//
// Subtracting the remainder first removes rounding error that a plain division
// leaves behind, so the two formulations disagree in the last bit on some inputs.
//
// Measured consequence, which is how this was found: `Pipeline._prepare_image`
// computes `int(w // ratio)` to resize an image to at most `img_size`. For a
// 2999x1777 input, Python yields width **1499** while `math.Floor(2999/ratio)`
// yields **1500** — a one-pixel-different canvas, and therefore every downstream box
// shifted by a pixel. A unit test pins both this case and 1789x1083, which happens to
// agree.
//
// The same trap exists in C# (`Math.Floor(x / y)`) and on the JVM. Every port needs
// this function; none should use its language's floor-division shortcut.
func FloorDiv(x, y float64) float64 {
	if y == 0 {
		return math.NaN()
	}
	mod := math.Mod(x, y)
	div := (x - mod) / y
	if mod != 0 {
		// Signs differing means the quotient must round toward negative infinity.
		if (y < 0) != (mod < 0) {
			div -= 1.0
		}
	}
	if div != 0 {
		floordiv := math.Floor(div)
		// CPython's nudge: the subtraction above can leave `div` just under an
		// integer, and this pulls it back. Reproduced rather than simplified away.
		if div-floordiv > 0.5 {
			floordiv += 1.0
		}
		return floordiv
	}
	// Preserve the sign of zero, as CPython does.
	return math.Copysign(0, x/y)
}

// FloorDivInt is FloorDiv with the int() truncation Python applies afterwards.
//
// For the positive values this codebase deals with, truncation after a floor is a
// no-op — but writing it out keeps the correspondence with `int(w // ratio)` visible
// instead of relying on the reader to know that.
func FloorDivInt(x, y float64) int {
	return int(FloorDiv(x, y))
}
