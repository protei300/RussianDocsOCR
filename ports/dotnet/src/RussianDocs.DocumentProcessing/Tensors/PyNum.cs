namespace RussianDocs.DocumentProcessing.Tensors;

/// <summary>
/// CPython's numeric semantics, where they differ from .NET's.
///
/// <para>
/// Ported from the Go port's <c>tensor/pynum.go</c>, which found the discrepancy the hard way.
/// </para>
/// </summary>
public static class PyNum
{
    /// <summary>
    /// CPython's <c>//</c> for floats. **Not** <c>Math.Floor(x / y)</c>.
    ///
    /// <para>
    /// CPython computes it through <c>fmod</c>: <c>mod = fmod(x, y); div = (x - mod) / y</c>, then
    /// floors, plus a half-ulp nudge. Subtracting the remainder first removes the rounding error
    /// that plain division leaves behind, so the two disagree in the last bit.
    /// </para>
    ///
    /// <para>
    /// **This is not academic.** For a 2999x1777 image, <c>_prepare_image</c>'s
    /// <c>int(w // ratio)</c> gives 1499 in Python and 1500 from <c>Math.Floor(2999 / ratio)</c> —
    /// a canvas one pixel wider, which shifts every box downstream and fails exact comparison at a
    /// stage far from the cause. Found by a unit test in the Go port, before it could do that.
    /// </para>
    ///
    /// <para>
    /// A related consequence worth stating because it is counter-intuitive: <c>_prepare_image</c>
    /// does NOT guarantee the long side ends up equal to <c>img_size</c>.
    /// </para>
    /// </summary>
    public static double FloorDiv(double x, double y)
    {
        if (y == 0)
        {
            return double.NaN;
        }
        // `%` on doubles in .NET is fmod — truncated division, remainder takes the sign of the
        // dividend — which is what CPython's implementation calls. **Not** Math.IEEERemainder,
        // which rounds the quotient to NEAREST and therefore returns a differently-signed value;
        // using it here silently breaks the negative cases.
        double mod = x % y;
        double div = (x - mod) / y;
        if (mod != 0 && (y < 0) != (mod < 0))
        {
            // Signs differing means the quotient rounds toward negative infinity.
            div -= 1.0;
        }
        if (div != 0)
        {
            double floor = Math.Floor(div);
            // CPython's nudge: the subtraction above can leave `div` just under an integer, and
            // this pulls it back. Reproduced rather than simplified away.
            if (div - floor > 0.5)
            {
                floor += 1.0;
            }
            return floor;
        }
        // Preserve the sign of zero, as CPython does.
        return Math.CopySign(0, x / y);
    }

    /// <summary>
    /// <see cref="FloorDiv"/> followed by Python's <c>int()</c> truncation.
    ///
    /// <para>
    /// For the positive values this codebase deals with, truncating after a floor is a no-op — but
    /// writing it out keeps the correspondence with <c>int(w // ratio)</c> visible instead of
    /// relying on the reader to know that.
    /// </para>
    /// </summary>
    public static int FloorDivInt(double x, double y) => (int)FloorDiv(x, y);

    /// <summary>
    /// <c>np.round</c>: half to even.
    ///
    /// <para>
    /// .NET's <c>Math.Round(double)</c> already does this, unlike Go's <c>math.Round</c> which
    /// rounds away from zero. Wrapped anyway, for two reasons: the call sites should say which
    /// rounding they mean rather than depending on a default, and an <c>Math.Round(x, 0,
    /// MidpointRounding.AwayFromZero)</c> introduced later would then be visibly wrong.
    /// </para>
    /// </summary>
    public static double RoundHalfEven(double value) => Math.Round(value, MidpointRounding.ToEven);

    /// <inheritdoc cref="RoundHalfEven(double)"/>
    public static int RoundHalfEvenToInt(double value) => (int)Math.Round(value, MidpointRounding.ToEven);
}
