package net.russiandocs.docproc.tensors

import kotlin.math.floor
import kotlin.math.withSign

/**
 * CPython's numeric semantics, where they differ from the JVM's.
 *
 * Ported from the Go port's `tensor/pynum.go`, which found the discrepancy the hard way.
 */
public object PyNum {

    /**
     * CPython's `//` for floats. **Not** `floor(x / y)`.
     *
     * CPython computes it through `fmod`: `mod = fmod(x, y); div = (x - mod) / y`, then floors, plus
     * a half-ulp nudge. Subtracting the remainder first removes the rounding error that plain
     * division leaves behind, so the two disagree in the last bit.
     *
     * **This is not academic.** For a 2999x1777 image, `_prepare_image`'s `int(w // ratio)` gives
     * 1499 in Python and 1500 from `floor(2999 / ratio)` — a canvas one pixel wider, which shifts
     * every box downstream and fails exact comparison at a stage far from the cause. Found by a unit
     * test in the Go port, before it could do that.
     *
     * A related consequence worth stating because it is counter-intuitive: `_prepare_image` does NOT
     * guarantee the long side ends up equal to `img_size`.
     */
    public fun floorDiv(x: Double, y: Double): Double {
        if (y == 0.0) {
            return Double.NaN
        }
        // Kotlin's `%` on Double is IEEE fmod — truncated division, remainder takes the sign of the
        // dividend — which is what CPython's implementation calls. **Not** Math.IEEEremainder, which
        // rounds the quotient to NEAREST and returns a differently-signed value; using that here
        // silently breaks the negative cases.
        val mod = x % y
        var div = (x - mod) / y
        if (mod != 0.0 && (y < 0) != (mod < 0)) {
            // Signs differing means the quotient rounds toward negative infinity.
            div -= 1.0
        }
        if (div != 0.0) {
            var f = floor(div)
            // CPython's nudge: the subtraction above can leave `div` just under an integer, and this
            // pulls it back. Reproduced rather than simplified away.
            if (div - f > 0.5) {
                f += 1.0
            }
            return f
        }
        // Preserve the sign of zero, as CPython does.
        return 0.0.withSign(x / y)
    }

    /**
     * [floorDiv] followed by Python's `int()` truncation.
     *
     * For the positive values this codebase deals with, truncating after a floor is a no-op — but
     * writing it out keeps the correspondence with `int(w // ratio)` visible instead of relying on
     * the reader to know that.
     */
    public fun floorDivInt(x: Double, y: Double): Int = floorDiv(x, y).toInt()

    /**
     * `np.round`: half to EVEN.
     *
     * `Math.rint` already does this on the JVM, unlike `Math.round` (half up, away from zero for
     * positives) and unlike Kotlin's `roundToInt` (also half up). Wrapped anyway, for two reasons:
     * the call sites should say which rounding they mean rather than depending on a default, and a
     * `roundToInt` introduced later would then be visibly wrong beside this.
     *
     * The distinction is load-bearing: `np.round(0.5) == 0`, and a different integer box coordinate
     * means a different crop, which means different text.
     */
    public fun roundHalfEven(value: Double): Double = Math.rint(value)

    public fun roundHalfEvenToInt(value: Double): Int = Math.rint(value).toInt()

    /**
     * Python's `int()` on a float: truncation toward zero.
     *
     * Kotlin's `Double.toInt()` truncates identically, so this is a naming wrapper rather than a
     * behavioural one — present so a reader of the ported line can see which Python builtin it is.
     */
    public fun toInt(value: Double): Int = value.toInt()
}
