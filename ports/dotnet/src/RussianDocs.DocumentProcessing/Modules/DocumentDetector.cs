using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>A page of an internal passport, lying inside a document. Box is [x1, y1, x2, y2].</summary>
public sealed class DocumentPage(double[] box, double conf)
{
    public double[] Box { get; set; } = box;
    public double Conf { get; } = conf;
}

/// <summary>
/// One document found in the frame. Box is [x1, y1, x2, y2]: on the image the detector read at
/// first, and on the INPUT image once <c>Recognizer</c> has scaled it back.
/// </summary>
public sealed class DetectedDocument(double[] box, double conf)
{
    public double[] Box { get; set; } = box;
    public double Conf { get; } = conf;
    public List<DocumentPage> Pages { get; } = [];
}

/// <summary>
/// Finds every document lying in a frame, and the pages of a passport spread. Port of
/// <c>DocumentDetector</c> in <c>pipeline_modules/document_detector/document_detector.py</c>
/// (decision #142): the first stage of the pipeline — the type classifier, the border detector and
/// everything after them then work on the crop of one document, not on the whole frame.
/// </summary>
public sealed class DocumentDetector : IDisposable
{
    /// <summary>The key in <c>models_path.yaml</c>, and the folder under <c>models/</c>.</summary>
    public const string ModelKey = "DocumentDetector";

    private readonly DetectionModel _model;

    public DocumentDetector(string root, IReadOnlyDictionary<string, string> paths, Device device,
        int threads)
        => _model = new DetectionModel(
            Path.Combine(ModelPaths.Resolve(root, paths, ModelKey), "ONNX"), device, threads);

    /// <summary>
    /// True when this weight set carries the document detector. A set of models-v8 or older does
    /// not, and the reference then reads the whole frame instead of refusing to start — so the port
    /// asks BEFORE constructing, rather than catching whatever the session constructor throws.
    /// </summary>
    public static bool WeightsPresent(string root, IReadOnlyDictionary<string, string> paths)
    {
        if (!paths.ContainsKey(ModelKey))
        {
            return false;
        }
        string dir = Path.Combine(ModelPaths.Resolve(root, paths, ModelKey), "ONNX");
        return File.Exists(Path.Combine(dir, "model.json")) && File.Exists(Path.Combine(dir, "model.onnx"));
    }

    /// <summary>Documents in the image, largest first.</summary>
    public List<DetectedDocument> Predict(Image image) => GroupDocuments(_model.Predict(image));

    private static double Area(double[] box) =>
        Math.Max(0.0, box[2] - box[0]) * Math.Max(0.0, box[3] - box[1]);

    /// <summary>True when at least <paramref name="share"/> of <paramref name="inner"/> lies within <paramref name="outer"/>.</summary>
    private static bool Inside(double[] inner, double[] outer, double share = 0.7)
    {
        double w = Math.Min(inner[2], outer[2]) - Math.Max(inner[0], outer[0]);
        double h = Math.Min(inner[3], outer[3]) - Math.Max(inner[1], outer[1]);
        if (w <= 0 || h <= 0)
        {
            return false;
        }
        return w * h >= share * Math.Max(Area(inner), 1e-9);
    }

    /// <summary>
    /// Detector boxes to documents, each with the pages that lie inside it.
    ///
    /// <para>
    /// The detector has two classes: <c>document</c> — whatever lies in the frame as one piece (a
    /// passport spread, a card, a single visible page) — and <c>page</c>, every visible page of an
    /// internal passport. A page belongs to the FIRST document it lies in (in area order); a document
    /// with no page inside is a single sheet. Documents come out largest first, which is the one
    /// <c>process_img</c> reads. OrderByDescending is stable, as Python's <c>sort(reverse=True)</c>
    /// is, so two equal areas keep the detector's order.
    /// </para>
    /// </summary>
    public static List<DetectedDocument> GroupDocuments(List<Box> boxes)
    {
        List<Box> docs = [.. boxes.Where(b => b.Label == "document")
            .OrderByDescending(b => Area([b.X1, b.Y1, b.X2, b.Y2]))];
        List<Box> pages = [.. boxes.Where(b => b.Label == "page")];

        var output = docs.Select(d => new DetectedDocument([d.X1, d.Y1, d.X2, d.Y2], d.Conf)).ToList();
        foreach (Box p in pages)
        {
            double[] pbox = [p.X1, p.Y1, p.X2, p.Y2];
            DetectedDocument? owner = output.FirstOrDefault(d => Inside(pbox, d.Box));
            owner?.Pages.Add(new DocumentPage(pbox, p.Conf));
        }
        foreach (DetectedDocument d in output)
        {
            // Stable, by (y1, x1) — Python sorts on that tuple.
            List<DocumentPage> sorted = [.. d.Pages.OrderBy(p => p.Box[1]).ThenBy(p => p.Box[0])];
            d.Pages.Clear();
            d.Pages.AddRange(sorted);
        }
        return output;
    }

    public void Dispose() => _model.Dispose();
}
