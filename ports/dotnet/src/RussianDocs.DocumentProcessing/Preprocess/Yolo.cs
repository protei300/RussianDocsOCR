using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Preprocess;

/// <summary>
/// The detector input: pad, letterbox to the declared square, hand over as uint8 NHWC.
/// </summary>
public sealed class Yolo : IPreprocessor
{
    /// <summary>
    /// The letterbox fill. Grey rather than black because that is what the reference uses, and the
    /// value reaches the model — a different fill shifts every score slightly.
    /// </summary>
    private static readonly byte[] Fill = [114, 114, 114];

    private readonly int _width;
    private readonly int _height;
    private readonly IReadOnlyList<int>? _paddingSize;
    private readonly IReadOnlyList<int>? _paddingColour;

    public Yolo(ModelInput input)
    {
        List<int> shape = input.Shape
            ?? throw new InvalidDataException("preprocess: YOLO input has no Shape");
        if (shape.Count < 2)
        {
            throw new InvalidDataException(
                $"preprocess: YOLO Shape needs at least 2 entries, got {shape.Count}");
        }
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
            int paddedH = padded.Height, paddedW = padded.Width;
            (Image boxed, double ratio, double[] padLetter) = Letterbox(padded, _height, _width);
            using (boxed)
            {
                NdArray pixels = boxed.ToArray();
                var batched = new NdArray(pixels.Data,
                    [1, boxed.Height, boxed.Width, boxed.Channels], Dtype.UInt8, 1);
                return (batched, new Meta
                {
                    Ratio = ratio,
                    PadExtra = extra,
                    PadLetter = padLetter,
                    PaddedH = paddedH,
                    PaddedW = paddedW,
                    OrigH = image.Height,
                    OrigW = image.Width,
                });
            }
        }
    }

    /// <summary>
    /// Scales to fit and pads the remainder — the standard YOLO letterbox.
    ///
    /// <para>
    /// **The asymmetric <c>±0.1</c> is not decoration.** With an odd amount of padding to distribute,
    /// <c>round(dh - 0.1)</c> and <c>round(dh + 0.1)</c> put the extra row at the BOTTOM and the
    /// extra column at the RIGHT. A "clean" implementation that halves the padding evenly shifts
    /// every returned box by a pixel, and the failure surfaces as a coordinate mismatch with no
    /// obvious source.
    /// </para>
    ///
    /// <para>
    /// Every rounding here is half-to-even, matching <c>np.round</c>. In .NET that is the default;
    /// the calls are explicit anyway so the intent survives a reader who does not know that.
    /// </para>
    ///
    /// <para>
    /// The resize is SKIPPED when the size already matches, rather than run with a ratio of 1.0 —
    /// running the interpolator needlessly can perturb pixels.
    /// </para>
    /// </summary>
    internal static (Image Boxed, double Ratio, double[] PadLetter) Letterbox(
        Image src, int targetH, int targetW)
    {
        int h = src.Height, w = src.Width;

        double ratio = Math.Min((double)targetH / h, (double)targetW / w);
        int newW = PyNum.RoundHalfEvenToInt(w * ratio);
        int newH = PyNum.RoundHalfEvenToInt(h * ratio);

        double dw = (targetW - newW) / 2.0;
        double dh = (targetH - newH) / 2.0;

        Image scaled = w != newW || h != newH
            ? Io.Resize(src, newW, newH, Interpolation.Linear)
            : src.Clone();

        using (scaled)
        {
            int top = PyNum.RoundHalfEvenToInt(dh - 0.1);
            int bottom = PyNum.RoundHalfEvenToInt(dh + 0.1);
            int left = PyNum.RoundHalfEvenToInt(dw - 0.1);
            int right = PyNum.RoundHalfEvenToInt(dw + 0.1);

            Image boxed = Io.CopyMakeBorderConstant(scaled, top, bottom, left, right,
                Fill[0], Fill[1], Fill[2]);
            return (boxed, ratio, [dw, dh]);
        }
    }
}

/// <summary>
/// The oriented-box detector input (<c>YOLOOBB</c>, <c>OBBPreprocessing</c>): the same pad and letterbox as
/// <see cref="Yolo"/>, but handed over as a float32 NCHW tensor scaled to [0, 1] - the standard ultralytics
/// export, which unlike the older detectors has no normalisation baked in.
///
/// <para>
/// The division is done on float32 values (<c>uint8 -> float32, / 255</c>), as numpy does it: the scaled value is
/// the nearest float32 to <c>b / 255</c>, not a double rounded afterwards.
/// </para>
/// </summary>
public sealed class YoloObb : IPreprocessor
{
    private readonly Yolo _inner;

    public YoloObb(ModelInput input) => _inner = new Yolo(input);

    public (NdArray Tensor, Meta Meta) Apply(Image image)
    {
        (NdArray nhwc, Meta meta) = _inner.Apply(image);
        int h = nhwc.Shape[1], w = nhwc.Shape[2], c = nhwc.Shape[3];
        ReadOnlySpan<byte> pixels = nhwc.AsUInt8();
        var planes = new float[c * h * w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                for (int k = 0; k < c; k++)
                {
                    planes[(k * h + y) * w + x] = pixels[(y * w + x) * c + k] / 255f;
                }
            }
        }
        return (NdArray.FromFloat32(planes, 1, c, h, w), meta);
    }
}
