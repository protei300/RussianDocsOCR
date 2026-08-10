using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Postprocess;

/// <summary>What a postprocessor needs from the preprocessing that fed the model.</summary>
public sealed record Context
{
    public double Ratio { get; init; } = 1.0;
    public int[] PadExtra { get; init; } = [0, 0];
    public double[] PadLetter { get; init; } = [0, 0];
    public int PaddedH { get; init; }
    public int PaddedW { get; init; }
    public int OrigH { get; init; }
    public int OrigW { get; init; }
    public bool Resize { get; init; }
}

/// <summary>
/// The closed set of postprocessor results.
///
/// <para>
/// A marker interface rather than a generic <c>Model&lt;T&gt;</c>: the concrete type is not known
/// until <c>model.json</c> has been read, so a generic model would still need a runtime cast at the
/// load site. All three ports reached the same conclusion. The cast happens ONCE, in the module that
/// knows what it asked for.
/// </para>
/// </summary>
public interface IResult;

/// <summary>A single label plus its score. Angle heads and the quality classifiers.</summary>
public sealed record ClassResult(string Label, double Confidence) : IResult;

/// <summary>Nearest-centroid outcome. <c>Label</c> is <c>"NONE"</c> when nothing is close enough.</summary>
public sealed record MetricResult(string Label, double Distance, double Threshold) : IResult;

public interface IPostprocessor
{
    IResult Apply(NdArray output, Context context);
}

/// <summary>Wired rather than omitted, for the same reason as the preprocessor twin (D-06).</summary>
public sealed class NotImplementedPostprocessor(string tag) : IPostprocessor
{
    public IResult Apply(NdArray output, Context context) =>
        throw new NotImplementedException($"postprocess: output type \"{tag}\" is not implemented");
}

/// <summary>
/// Picks the nearest centroid — the head <c>DocTypeAngles</c> uses to name the document type.
///
/// <para>
/// The reference builds a sklearn <c>NearestNeighbors(metric='cosine', radius=1)</c> index for this.
/// With NINE centroids that is a linear scan with extra steps, so this is the scan: no equivalent of
/// sklearn is needed here or anywhere else in the project.
/// </para>
///
/// <para>
/// Two things about the outcome are easy to get subtly wrong. The <c>radius</c> is a hard filter —
/// a centroid further away than the radius is not a neighbour at all, not merely a bad one. And the
/// per-class <c>max_distance</c> is applied AFTER the nearest is chosen, so a document can have a
/// clear nearest centroid and still come back <c>"NONE"</c>.
/// </para>
/// </summary>
public sealed class Metric : IPostprocessor
{
    private readonly double _radius;
    private readonly string[] _labels;
    private readonly float[][] _centers;
    private readonly double[] _maxDistance;
    private readonly bool _cosine;

    public Metric(string npzPath, string metric)
    {
        (_radius, _cosine) = metric switch
        {
            "Cosine" or "cosine" => (1.0, true),
            "Euclidean" or "euclidean" => (10.0, false),
            _ => throw new InvalidDataException($"postprocess: unsupported metric \"{metric}\""),
        };

        Dictionary<string, NdArray> blob = Npz.Load(npzPath);
        NdArray labels = Require(blob, "labels", npzPath);
        NdArray centers = Require(blob, "centers", npzPath);
        NdArray maxDistance = Require(blob, "max_distance", npzPath);

        _labels = labels.AsUnicode();
        if (centers.Shape.Length != 2 || centers.Shape[0] != _labels.Length)
        {
            throw new InvalidDataException(
                $"postprocess: centers {NdArray.Describe(centers.Shape)} does not align with " +
                $"{_labels.Length} labels");
        }

        int dim = centers.Shape[1];
        ReadOnlySpan<float> flat = centers.AsFloat32();
        _centers = new float[_labels.Length][];
        for (int i = 0; i < _labels.Length; i++)
        {
            _centers[i] = flat.Slice(i * dim, dim).ToArray();
        }

        ReadOnlySpan<float> maxima = maxDistance.AsFloat32();
        _maxDistance = new double[_labels.Length];
        for (int i = 0; i < _labels.Length; i++)
        {
            _maxDistance[i] = maxima[i];
        }
    }

    private static NdArray Require(Dictionary<string, NdArray> blob, string key, string path) =>
        blob.TryGetValue(key, out NdArray? value)
            ? value
            : throw new InvalidDataException($"postprocess: {path} has no '{key}'");

    public IResult Apply(NdArray output, Context context)
    {
        ReadOnlySpan<float> embedding = output.AsFloat32();
        if (_centers.Length == 0)
        {
            throw new InvalidDataException("postprocess: no centroids loaded");
        }
        if (embedding.Length != _centers[0].Length)
        {
            throw new InvalidDataException(
                $"postprocess: embedding has {embedding.Length} dims, centroids have " +
                $"{_centers[0].Length}");
        }

        int best = -1;
        double bestDistance = double.PositiveInfinity;
        for (int i = 0; i < _centers.Length; i++)
        {
            double d = _cosine
                ? Ops.CosineDistance(embedding, _centers[i])
                : Ops.EuclideanDistance(embedding, _centers[i]);
            if (d > _radius)
            {
                continue; // outside the radius: not a neighbour at all
            }
            if (d < bestDistance)
            {
                best = i;
                bestDistance = d;
            }
        }

        if (best < 0)
        {
            return new MetricResult("NONE", double.PositiveInfinity, 0);
        }

        double threshold = _maxDistance[best];
        return bestDistance < threshold
            ? new MetricResult(_labels[best], bestDistance, threshold)
            : new MetricResult("NONE", bestDistance, threshold);
    }
}

/// <summary>
/// A single sigmoid score against a threshold — <c>BinaryClassification</c>.
///
/// <para>
/// **Not an argmax, and getting that wrong is silent.** These outputs have <c>Shape [1]</c> and two
/// declared labels: the value is P(second label), compared against <c>Threshold</c> (0.5 when the
/// config omits it). Feeding a one-element vector to an argmax returns index 0 every time, so every
/// document came back as the FIRST label — <c>FAKE</c> for the spoofing checks — with no error
/// anywhere. Found by conformance, which reported REAL vs FAKE on five of seven cases; nothing in
/// the code would have shown it.
/// </para>
///
/// <para>
/// The reported confidence is the score for the label that WON, not the raw output, so a
/// below-threshold result reports how strongly it was below.
/// </para>
/// </summary>
public sealed class BinaryClass(string[] labels, double threshold) : IPostprocessor
{
    public IResult Apply(NdArray output, Context context)
    {
        ReadOnlySpan<float> scores = output.AsFloat32();
        if (scores.IsEmpty)
        {
            throw new InvalidDataException("postprocess: empty binary score");
        }
        if (labels.Length < 2)
        {
            throw new InvalidDataException(
                $"postprocess: binary output needs 2 labels, got {labels.Length}");
        }

        double score = scores[0];
        return score > threshold
            ? new ClassResult(labels[1], score)
            : new ClassResult(labels[0], 1 - score);
    }
}

/// <summary>Argmax over a score vector, with the model's declared labels.</summary>
public sealed class MultiClass : IPostprocessor
{
    private readonly string[] _labels;

    public MultiClass(ModelOutput output)
    {
        _labels = output.LabelsAsStrings();
        if (_labels.Length == 0)
        {
            throw new InvalidDataException("postprocess: multiclass output declares no Labels");
        }
    }

    public IResult Apply(NdArray output, Context context)
    {
        ReadOnlySpan<float> scores = output.AsFloat32();
        if (scores.IsEmpty)
        {
            throw new InvalidDataException("postprocess: empty score vector");
        }
        int index = Ops.Argmax(scores);
        if (index >= _labels.Length)
        {
            throw new InvalidDataException(
                $"postprocess: class {index} has no label (only {_labels.Length} declared)");
        }
        return new ClassResult(_labels[index], Ops.Max(scores));
    }
}
