using System.Globalization;
using System.Text.Json.Serialization;
using OpenCvSharp;
using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>
/// Document type and 90-degree orientation, from one model with two heads.
///
/// <para>
/// The field names are the wire contract — this is what the <c>doctype.label</c> stage compares.
/// </para>
/// </summary>
public sealed record DocTypeResult
{
    [JsonPropertyName("doc_type")] public string DocType { get; init; } = "NONE";
    [JsonPropertyName("doc_type_confidence")] public double DocTypeConfidence { get; init; }
    [JsonPropertyName("angle")] public int Angle { get; init; }
    [JsonPropertyName("angle_confidence")] public double AngleConfidence { get; init; }
}

/// <summary>
/// Port of <c>pipeline_modules/doctype_angles_classificator</c>.
///
/// <para>
/// One model, two outputs, in the order the config declares them: an embedding that the metric head
/// turns into a document type, and a four-way angle classifier. They are not interchangeable, so both
/// are cast to their expected result type and a mismatch is an error rather than a reinterpretation.
/// </para>
/// </summary>
public sealed class DocTypeAngles : IDisposable
{
    private const string ModuleName = "DocTypeAngles";

    private readonly Model _model;
    private readonly int[] _angleLabels;

    public DocTypeAngles(string root, IReadOnlyDictionary<string, string> paths, Device device,
        int threads)
    {
        string dir = Path.Combine(ModelPaths.Resolve(root, paths, ModuleName), "ONNX");
        _model = Loader.Load(dir, device, threads);

        if (_model.Config.Outputs.Count != 2)
        {
            _model.Dispose();
            throw new InvalidDataException(
                $"modules: {ModuleName} expects 2 outputs (embeddings, angle), got " +
                $"{_model.Config.Outputs.Count}");
        }
        _angleLabels = _model.Config.Outputs[1].LabelsAsInts();
    }

    public DocTypeResult Predict(Image image)
    {
        IResult[] outputs = _model.Predict(image);

        // Cast ONCE, here, in the module that knows what it asked for. This is the single place the
        // closed result set is narrowed — see CONVENTIONS §5.
        if (outputs[0] is not MetricResult metric)
        {
            throw new InvalidDataException(
                $"modules: {ModuleName} output 0 is {outputs[0].GetType().Name}, want MetricResult");
        }
        if (outputs[1] is not ClassResult angle)
        {
            throw new InvalidDataException(
                $"modules: {ModuleName} output 1 is {outputs[1].GetType().Name}, want ClassResult");
        }

        // The confidence the wire carries is a RATIO against the class threshold, not the raw
        // distance, and it is rounded to two places — matching the reference exactly, because this
        // value is compared as a float with a 1e-3 tolerance that leaves no room for a third digit.
        double confidence = metric.Threshold > 0
            ? Ops.RoundHalfEven(1 - metric.Distance / metric.Threshold, 2)
            : 0.0;

        return new DocTypeResult
        {
            DocType = metric.Label,
            DocTypeConfidence = confidence,
            Angle = AngleFromLabel(angle.Label),
            AngleConfidence = angle.Confidence,
        };
    }

    /// <summary>
    /// Predicts, then rotates the image upright.
    ///
    /// <para>
    /// <c>angle / 90</c> quarter-turns COUNTER-clockwise, because the angle names how far the document
    /// is rotated and the correction undoes it. The result is a new image the caller owns; the input
    /// is only borrowed.
    /// </para>
    /// </summary>
    public (DocTypeResult Meta, Image Upright) PredictTransform(Image image)
    {
        DocTypeResult meta = Predict(image);

        Image current = image.Clone();
        for (int i = 0; i < meta.Angle / 90; i++)
        {
            Image next = Rotate90Ccw(current);
            current.Dispose();
            current = next;
        }
        return (meta, current);
    }

    private static Image Rotate90Ccw(Image src)
    {
        var dst = new Mat();
        Cv2.Rotate(src.Mat, dst, RotateFlags.Rotate90Counterclockwise);
        return Image.Wrap(dst);
    }

    /// <summary>
    /// Maps the classifier's label back to degrees.
    ///
    /// <para>
    /// A lookup rather than <c>int.Parse</c>: the label comes from the config, and an angle the model
    /// does not declare must be an error rather than a number that happens to parse.
    /// </para>
    /// </summary>
    private int AngleFromLabel(string label)
    {
        foreach (int candidate in _angleLabels)
        {
            if (candidate.ToString(CultureInfo.InvariantCulture) == label)
            {
                return candidate;
            }
        }
        throw new InvalidDataException(
            $"modules: {ModuleName} angle label \"{label}\" is not one of " +
            $"[{string.Join(", ", _angleLabels)}]");
    }

    public void Dispose() => _model.Dispose();
}
