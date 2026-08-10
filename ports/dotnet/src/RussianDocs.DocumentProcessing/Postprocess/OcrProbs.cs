using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Postprocess;

public sealed record TextResult(string Text) : IResult;

/// <summary>
/// Greedy CTC decoding with per-step alphabet masking.
///
/// <para>
/// The model emits a softmax matrix <c>[1, T, C]</c>; this collapses it to a string.
/// </para>
/// </summary>
public sealed class OcrProbs : IPostprocessor
{
    /// <summary>
    /// The alphabet, as CODE POINTS.
    ///
    /// <para>
    /// **Not chars, and not bytes.** The alphabet in <c>model.json</c> is a UTF-8 string of Cyrillic
    /// and Latin letters; indexing it by byte gives mojibake, which is the trap Go had to be warned
    /// about. C# strings are UTF-16, so <c>alphabet[i]</c> would work for the shipped alphabets — every
    /// character is in the BMP — but it would break silently the first time one is not. Enumerating
    /// runes costs nothing and cannot be wrong.
    /// </para>
    /// </summary>
    private readonly string[] _alphabet;

    private readonly int _blankIndex;
    private readonly HashSet<int>? _allowed;
    private readonly HashSet<int>? _disallowed;

    /// <summary>
    /// </summary>
    /// <param name="alphabet">The model's FULL alphabet, from <c>model.json</c>.</param>
    /// <param name="allowedChars">
    /// The per-document charset, or null to disable masking. Note this is a subset of
    /// <paramref name="alphabet"/> resolved from <c>ocr_alphabets.json</c>, not the same thing.
    /// </param>
    /// <param name="blankIndex">The CTC blank. Legitimately 0, which is why the config field is nullable.</param>
    public OcrProbs(string alphabet, IReadOnlySet<string>? allowedChars, int blankIndex)
    {
        if (string.IsNullOrEmpty(alphabet))
        {
            throw new InvalidDataException("postprocess: OCRProbs needs an Alphabet");
        }

        var runes = new List<string>();
        var enumerator = System.Globalization.StringInfo.GetTextElementEnumerator(alphabet);
        while (enumerator.MoveNext())
        {
            runes.Add((string)enumerator.Current);
        }
        _alphabet = [.. runes];
        _blankIndex = blankIndex;

        if (allowedChars is null)
        {
            return;
        }

        // **Class index is alphabet index PLUS ONE**, because class 0 is the blank. Getting this
        // off by one shifts every decoded character by one position in the alphabet, which produces
        // readable-looking nonsense rather than an error.
        _allowed = [];
        _disallowed = [];
        for (int i = 0; i < _alphabet.Length; i++)
        {
            if (allowedChars.Contains(_alphabet[i]))
            {
                _allowed.Add(i + 1);
            }
            else
            {
                _disallowed.Add(i + 1);
            }
        }
    }

    public IResult Apply(NdArray output, Context context) => new TextResult(Decode(output));

    /// <summary>
    /// Greedy decode: argmax per timestep, mask, then collapse repeats and blanks.
    /// </summary>
    public string Decode(NdArray output)
    {
        ReadOnlySpan<float> data = output.AsFloat32();
        int[] shape = output.Shape;
        if (shape.Length == 3)
        {
            shape = shape[1..];
        }
        if (shape.Length != 2)
        {
            throw new InvalidDataException(
                $"postprocess: OCRProbs expects [T,C] or [1,T,C], got {NdArray.Describe(output.Shape)}");
        }

        int steps = shape[0], classes = shape[1];
        if (classes > _alphabet.Length + 1)
        {
            throw new InvalidDataException(
                $"postprocess: {classes} classes exceeds alphabet of {_alphabet.Length} plus blank");
        }

        var indices = new int[steps];
        bool masking = _disallowed is { Count: > 0 };

        for (int t = 0; t < steps; t++)
        {
            ReadOnlySpan<float> row = data.Slice(t * classes, classes);
            int best = Ops.Argmax(row);

            if (!masking || best == _blankIndex || _allowed!.Contains(best))
            {
                indices[t] = best;
                continue;
            }

            // **Masking SUBSTITUTES the best allowed non-blank; it does not zero the column.** Zeroing
            // disallowed classes lets the blank win and the character disappears entirely — the
            // reference instead swaps in the nearest permitted letter, which is how `Î` becomes `I`
            // and `І` becomes `И` rather than vanishing.
            int bestAllowed = -1;
            double bestScore = double.NegativeInfinity;
            for (int c = 0; c < classes; c++)
            {
                if (c == _blankIndex || _disallowed!.Contains(c))
                {
                    continue;
                }
                if (row[c] > bestScore)
                {
                    bestAllowed = c;
                    bestScore = row[c];
                }
            }
            indices[t] = bestAllowed >= 0 ? bestAllowed : best;
        }

        // Standard CTC collapse: drop repeats of the same class and drop blanks.
        var text = new System.Text.StringBuilder();
        int previous = -1;
        foreach (int index in indices)
        {
            if (index != previous && index != _blankIndex)
            {
                text.Append(_alphabet[index - 1]);
            }
            previous = index;
        }
        return text.ToString();
    }
}
