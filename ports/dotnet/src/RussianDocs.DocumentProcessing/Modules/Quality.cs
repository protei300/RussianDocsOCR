using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>One tile's classification.</summary>
internal readonly record struct TileVerdict(string Label, double Confidence);

/// <summary>Shared tiling for the two tile-based quality checks.</summary>
internal static class Tiles
{
    private const int WindowSize = 128;

    /// <summary>
    /// Resizes to a whole number of tiles and classifies each one.
    ///
    /// <para>
    /// **The colour conversion looks like a bug and is not.** The reference calls
    /// <c>cvtColor(COLOR_BGR2RGB)</c> on an image that is ALREADY RGB, so what actually happens is
    /// RGB to BGR — the quality classifiers see BGR. Reproduced exactly: "fixing" it would change
    /// every verdict these two models produce.
    /// </para>
    ///
    /// <para>
    /// The iteration order is x-outer, y-inner, matching the reference. It does not affect the
    /// aggregates below, which are order-independent, but keeping it means a future per-tile stage
    /// compares without a reordering step.
    /// </para>
    /// </summary>
    internal static List<TileVerdict> Classify(Model model, Image image, int canvasX, int canvasY)
    {
        using Image canvas = Io.Resize(image, canvasX * WindowSize, canvasY * WindowSize,
            Interpolation.Linear);
        using Image swapped = Io.ToBgr(canvas);

        var verdicts = new List<TileVerdict>(canvasX * canvasY);
        for (int xStep = 0; xStep < canvasX; xStep++)
        {
            for (int yStep = 0; yStep < canvasY; yStep++)
            {
                int x = WindowSize * xStep, y = WindowSize * yStep;
                using Image tile = Crop.ClampedCrop(swapped, x, y, x + WindowSize, y + WindowSize);
                IResult[] result = model.Predict(tile);
                if (result[0] is not ClassResult cls)
                {
                    throw new InvalidDataException(
                        $"modules: quality tile output is {result[0].GetType().Name}, want ClassResult");
                }
                verdicts.Add(new TileVerdict(cls.Label, cls.Confidence));
            }
        }
        return verdicts;
    }
}

/// <summary>Glare detection over a 7x4 tile grid.</summary>
public sealed class Glare : IDisposable
{
    private static readonly int[] Canvas = [7, 4];

    /// <summary>
    /// A tile counts as glared only ABOVE this confidence. Below it the tile is treated as clean,
    /// which is why a low-confidence GLARE verdict does not condemn the document.
    /// </summary>
    private const double ConfidenceGate = 0.85;

    private readonly Model _model;

    public Glare(string root, IReadOnlyDictionary<string, string> paths, Device device, int threads)
        => _model = Loader.Load(Path.Combine(ModelPaths.Resolve(root, paths, "Glare"), "ONNX"),
            device, threads);

    public (string Label, double Score) Predict(Image image)
    {
        List<TileVerdict> tiles = Tiles.Classify(_model, image, Canvas[0], Canvas[1]);
        if (tiles.Count == 0)
        {
            throw new InvalidDataException("modules: Glare classified no tiles");
        }

        // Counts CLEAN tiles, so `score` is the fraction that are glared. Written as the reference
        // writes it — adding 0 for a glared tile and 1 otherwise — rather than collapsed into a
        // count of glared tiles, because the two differ if a third label ever appears.
        double sum = 0;
        foreach (TileVerdict tile in tiles)
        {
            sum += tile.Label == "GLARE" && tile.Confidence > ConfidenceGate ? 0 : 1;
        }
        double score = 1 - sum / tiles.Count;
        return (score > 0 ? "bad" : "good", score);
    }

    public void Dispose() => _model.Dispose();
}

/// <summary>Blur detection over a 7x4 tile grid.</summary>
public sealed class Blur : IDisposable
{
    private static readonly int[] Canvas = [7, 4];
    private const double Gate = 0.9;

    private readonly Model _model;

    public Blur(string root, IReadOnlyDictionary<string, string> paths, Device device, int threads)
        => _model = Loader.Load(Path.Combine(ModelPaths.Resolve(root, paths, "Blur"), "ONNX"),
            device, threads);

    public (string Label, double Score) Predict(Image image)
    {
        List<TileVerdict> tiles = Tiles.Classify(_model, image, Canvas[0], Canvas[1]);

        // **Only three of the five labels count, and the others are excluded from the DENOMINATOR
        // too.** A tile the model calls something else is not a vote for sharpness, it simply does
        // not vote — which is a different aggregate from treating it as 0.
        double sum = 0;
        int counted = 0;
        foreach (TileVerdict tile in tiles)
        {
            switch (tile.Label)
            {
                case "Blur5":
                    sum += 0.5;
                    counted++;
                    break;
                case "Blur10":
                    sum += 1;
                    counted++;
                    break;
                case "NonBlur":
                    counted++;
                    break;
            }
        }

        // No countable tiles returns "sharp", not a division by zero. Rejecting a document because
        // the classifier had nothing to say about any tile would be a false negative.
        if (counted == 0)
        {
            return ("good", 1.0);
        }

        double score = 1 - sum / counted;
        return (score > Gate ? "good" : "bad", score);
    }

    public void Dispose() => _model.Dispose();
}

/// <summary>
/// Print and LCD spoofing. One type, two instances — they differ only in the model and the gate.
///
/// <para>
/// The gate is the interesting part: <c>PrintSpoofing</c> applies a 0.9 threshold ON TOP of the
/// model's own decision, so a REAL verdict below that confidence becomes FAKE. <c>LCDSpoofing</c>
/// passes 0 and takes the model's word. Both are reproduced as-is; the asymmetry is in the reference.
/// </para>
/// </summary>
public sealed class Spoofing : IDisposable
{
    private readonly Model _model;
    private readonly double _gate;

    public string Name { get; }

    private Spoofing(string name, double gate, string root,
        IReadOnlyDictionary<string, string> paths, Device device, int threads)
    {
        Name = name;
        _gate = gate;
        _model = Loader.Load(Path.Combine(ModelPaths.Resolve(root, paths, name), "ONNX"),
            device, threads);
    }

    public static Spoofing Print(string root, IReadOnlyDictionary<string, string> paths,
        Device device, int threads) =>
        new("PrintSpoofing", 0.9, root, paths, device, threads);

    public static Spoofing Lcd(string root, IReadOnlyDictionary<string, string> paths,
        Device device, int threads) =>
        new("LCDSpoofing", 0, root, paths, device, threads);

    public (string Label, double Score) Predict(Image image)
    {
        IResult[] result = _model.Predict(image);
        if (result[0] is not ClassResult cls)
        {
            throw new InvalidDataException(
                $"modules: {Name} output is {result[0].GetType().Name}, want ClassResult");
        }
        return _gate > 0 && cls.Confidence < _gate
            ? ("FAKE", cls.Confidence)
            : (cls.Label, cls.Confidence);
    }

    public void Dispose() => _model.Dispose();
}
