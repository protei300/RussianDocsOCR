using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Preprocess;

/// <summary>
/// The OCR v2 input: full-colour BGR, fixed height, DYNAMIC width.
///
/// <para>
/// The dynamic width is what made this the riskiest single thing in the whole port exercise — the
/// spike's kill-shot test — because both the input width and the output time dimension change on
/// every call inside one long-lived session. Normalisation is baked into the graph, so nothing is
/// scaled here.
/// </para>
/// </summary>
public sealed class OcrV2 : IPreprocessor
{
    /// <summary>
    /// The minimum width the model will accept.
    ///
    /// <para>
    /// 16, and it is load-bearing rather than defensive: a very narrow crop scaled by height alone
    /// can round to 1 or 2 pixels, which the graph rejects. The reference clamps here too.
    /// </para>
    /// </summary>
    private const int MinWidth = 16;

    private readonly int _height;
    private readonly string _colorOrder;

    public OcrV2(ModelInput input)
    {
        _height = input.Height is > 0 ? input.Height.Value : 32;
        _colorOrder = string.IsNullOrEmpty(input.ColorOrder) ? "BGR" : input.ColorOrder;
    }

    public (NdArray Tensor, Meta Meta) Apply(Image image)
    {
        int h = image.Height, w = image.Width;

        // A zero-sized crop returns a BLANK tensor of the minimum shape rather than throwing. The
        // reference does the same, and the degenerate crop is reachable: ClampedCrop yields it when a
        // detector box lands entirely outside the patch. Failing here would turn a rare bad box into
        // a failed document.
        if (h == 0 || w == 0)
        {
            var blank = NdArray.FromUInt8(new byte[_height * MinWidth * 3], 1, _height, MinWidth, 3);
            return (blank, new Meta { OrigH = h, OrigW = w, Ratio = 1 });
        }

        // Width scaled by the height ratio, rounded half-to-even, floored at the minimum.
        int newW = Math.Max(MinWidth,
            (int)Ops.RoundHalfEven((double)w * _height / h, 0));

        // **The model wants BGR and the pipeline works in RGB**, so the conversion is real work, not
        // a no-op like the one in the quality tiles. Driven by the config's ColorOrder rather than
        // hardcoded, because that is where the reference reads it from.
        Image source = _colorOrder == "BGR" ? Io.ToBgr(image) : image.Clone();
        using (source)
        {
            using Image resized = Io.Resize(source, newW, _height, Interpolation.Linear);
            NdArray pixels = resized.ToArray();
            var batched = new NdArray(pixels.Data,
                [1, resized.Height, resized.Width, resized.Channels], Dtype.UInt8, 1);
            return (batched, new Meta { OrigH = h, OrigW = w, Ratio = 1 });
        }
    }
}
