namespace RussianDocs.DocumentProcessing.Tensors;

/// <summary>Small numeric helpers whose exact semantics are part of the contract.</summary>
public static class Ops
{
    /// <summary>
    /// Index of the FIRST maximum — <c>np.argmax</c>'s tie rule.
    ///
    /// <para>
    /// Strict <c>&gt;</c>, never <c>&gt;=</c>. On a tie, <c>&gt;=</c> takes the LAST maximum, which
    /// flips a CTC timestep and changes a character. Because both answers are legitimate maxima the
    /// failure has no numeric signature at all — it shows up as one wrong letter in one field.
    /// </para>
    /// </summary>
    public static int Argmax(ReadOnlySpan<float> values)
    {
        if (values.IsEmpty)
        {
            throw new ArgumentException("tensor: argmax of an empty span");
        }
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }
        return best;
    }

    public static float Max(ReadOnlySpan<float> values) => values[Argmax(values)];

    /// <summary>
    /// <c>1 - cosine_similarity</c>, which is what sklearn's cosine METRIC returns.
    ///
    /// <para>
    /// Accumulated in double even though the inputs are float32. That is not a violation of "never
    /// widen float32": the reference computes this inside sklearn, which also accumulates in double,
    /// so matching it requires the same widening. The rule bans widening where the REFERENCE stays
    /// in float32 — here it does not.
    /// </para>
    /// </summary>
    public static double CosineDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        if (na == 0 || nb == 0)
        {
            return 1.0;
        }
        return 1.0 - dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    public static double EuclideanDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            sum += d * d;
        }
        return Math.Sqrt(sum);
    }

    /// <summary>
    /// <c>round(value, digits)</c> with Python's half-to-even, used for the confidences that reach
    /// the wire.
    /// </summary>
    public static double RoundHalfEven(double value, int digits) =>
        Math.Round(value, digits, MidpointRounding.ToEven);
}
