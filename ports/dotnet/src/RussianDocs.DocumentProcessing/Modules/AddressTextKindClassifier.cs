using OpenCvSharp;
using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>
/// Tells a printed address line from a handwritten one. Port of <c>AddressTextKindClassifier</c>.
///
/// <para>
/// A line is wide and short, so the crop is put on a square gray canvas (aspect kept, nothing stretched) before
/// the classifier's own square resize, which keeps the proportions of the letters. The answer is the label and
/// P(handwritten) - the head's raw score, not the confidence of the winner.
/// </para>
/// </summary>
public sealed class AddressTextKindClassifier : IDisposable
{
    /// <summary>The key in <c>models_path.yaml</c>, and the folder under <c>models/</c>.</summary>
    public const string ModelKey = "AddressTextKindClassifier";

    /// <summary>The letterbox fill, as in the reference.</summary>
    private static readonly Scalar Gray = new(114, 114, 114);

    private readonly Model _model;
    private readonly string[] _labels;
    private readonly double _threshold;

    public AddressTextKindClassifier(string root, IReadOnlyDictionary<string, string> paths, Device device,
        int threads)
    {
        _model = Loader.Load(Path.Combine(ModelPaths.Resolve(root, paths, ModelKey), "ONNX"), device, threads);
        ModelOutput output = _model.Config.Outputs[0];
        _labels = output.LabelsAsStrings();
        _threshold = output.Threshold ?? 0.5;
        if (_labels.Length < 2)
        {
            throw new InvalidDataException(
                $"modules: {ModelKey} declares {_labels.Length} labels, a binary head needs 2");
        }
    }

    /// <summary>The line on a square canvas, centred (<c>letterbox_square</c>). The caller owns the result.</summary>
    public static Image LetterboxSquare(Image image)
    {
        int h = image.Height, w = image.Width;
        int size = Math.Max(h, w);
        var canvas = new Mat(size, size, MatType.CV_8UC3, Gray);
        // `(size - h) // 2` of two non-negative ints
        int y0 = (size - h) / 2, x0 = (size - w) / 2;
        using var window = new Mat(canvas, new Rect(x0, y0, w, h));
        image.Mat.CopyTo(window);
        return Image.Wrap(canvas);
    }

    /// <summary><c>('printed' | 'handwritten', P(handwritten))</c>.</summary>
    public (string Kind, double PHandwritten) Predict(Image line)
    {
        using Image square = LetterboxSquare(line);
        IResult[] result = _model.Predict(square);
        if (result[0] is not ClassResult cls)
        {
            throw new InvalidDataException(
                $"modules: {ModelKey} output is {result[0].GetType().Name}, want ClassResult");
        }
        // the reference's rule: `labels[0 if p < threshold else 1], p`
        return (cls.Raw < _threshold ? _labels[0] : _labels[1], cls.Raw);
    }

    public void Dispose() => _model.Dispose();
}
