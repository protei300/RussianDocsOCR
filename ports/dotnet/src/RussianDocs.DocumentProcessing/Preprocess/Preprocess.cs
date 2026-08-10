using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Preprocess;

/// <summary>
/// What a preprocessor tells the postprocessor so it can undo the scaling.
///
/// <para>
/// Travels through the pipeline instead of being recomputed, because recomputing it means duplicating
/// the letterbox arithmetic in two places and having them drift.
/// </para>
/// </summary>
public sealed record Meta
{
    public double Ratio { get; init; } = 1.0;
    public int[] PadExtra { get; init; } = [0, 0];
    public double[] PadLetter { get; init; } = [0, 0];
    public int PaddedH { get; init; }
    public int PaddedW { get; init; }
    public int OrigH { get; init; }
    public int OrigW { get; init; }
}

/// <summary>One preprocessing step. An interface, never an abstract base class — see CONVENTIONS §5.</summary>
public interface IPreprocessor
{
    (NdArray Tensor, Meta Meta) Apply(Image image);
}

/// <summary>
/// A recognised-but-unimplemented input type.
///
/// <para>
/// Wired rather than omitted (D-06). An omitted case reads as an oversight and gets "helpfully"
/// filled in differently by each port; a case that exists and refuses cannot.
/// </para>
/// </summary>
public sealed class NotImplementedPreprocessor(string tag) : IPreprocessor
{
    public (NdArray, Meta) Apply(Image image) =>
        throw new NotImplementedException($"preprocess: input type \"{tag}\" is not implemented");
}

public static class Padding
{
    /// <summary>
    /// The symmetric constant border from a config's <c>PaddingSize</c>.
    ///
    /// <para>
    /// Every shipped <c>model.json</c> declares <c>[0, 0]</c>, so this is a no-op in practice — ported
    /// because it is part of the contract and because the reference returns the applied offsets for
    /// the postprocessor to undo.
    /// </para>
    ///
    /// <para>
    /// Note the halving: Python pads <c>pad_v // 2</c> top AND bottom, so a <c>PaddingSize</c> of
    /// <c>[4, 6]</c> adds 3 rows above and below, not 6 in total.
    /// </para>
    /// </summary>
    public static (Image Padded, int[] Extra) Pad(Image image, IReadOnlyList<int>? size,
        IReadOnlyList<int>? colour)
    {
        if (size is null || size.Count < 2 || (size[0] == 0 && size[1] == 0))
        {
            // Clone, so the caller owns the result either way and the disposal rule has no special
            // case. One copy of a no-op is cheaper than an ownership question at every call site.
            return (image.Clone(), [0, 0]);
        }

        int padH = size[0] / 2, padV = size[1] / 2;
        byte r = 0, g = 0, b = 0;
        if (colour is { Count: >= 3 })
        {
            r = (byte)colour[0];
            g = (byte)colour[1];
            b = (byte)colour[2];
        }
        return (Io.CopyMakeBorderConstant(image, padV, padV, padH, padH, r, g, b), [padH, padV]);
    }
}

/// <summary>
/// The classification input: pad, resize to the declared size, hand over as uint8 NHWC.
///
/// <para>
/// **No normalisation happens here.** The shipped graphs bake it in, and the reference's own
/// <c>BasePreprocessing.normalization</c> method is shadowed by an attribute of the same name and can
/// never be called — so the reference does not normalise either. Adding it would double-normalise
/// every input, which looks like a model problem rather than a preprocessing one.
/// </para>
/// </summary>
public sealed class Classification : IPreprocessor
{
    private readonly int _width;
    private readonly int _height;
    private readonly IReadOnlyList<int>? _paddingSize;
    private readonly IReadOnlyList<int>? _paddingColour;

    public Classification(ModelInput input)
    {
        List<int> shape = input.Shape
            ?? throw new InvalidDataException("preprocess: classification input has no Shape");
        if (shape.Count < 2)
        {
            throw new InvalidDataException(
                $"preprocess: classification Shape needs at least 2 entries, got {shape.Count}");
        }

        // **Shape is (h, w) and cv2.resize takes (w, h).** The shipped sizes are square, which hides
        // a swap — hence the deliberately non-square resize in the conformance suite.
        _height = shape[0];
        _width = shape[1];
        _paddingSize = input.PaddingSize;
        _paddingColour = input.PaddingColor;
    }

    public (NdArray Tensor, Meta Meta) Apply(Image image)
    {
        (Image padded, int[] extra) = Padding.Pad(image, _paddingSize, _paddingColour);
        using (padded)
        {
            using Image resized = Io.Resize(padded, _width, _height, Interpolation.Linear);
            NdArray pixels = resized.ToArray();
            // Add the batch dimension: (H, W, C) becomes (1, H, W, C).
            var batched = new NdArray(pixels.Data,
                [1, resized.Height, resized.Width, resized.Channels], Dtype.UInt8, 1);
            return (batched, new Meta
            {
                PadExtra = extra,
                OrigH = image.Height,
                OrigW = image.Width,
            });
        }
    }
}
