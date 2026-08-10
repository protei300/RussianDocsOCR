using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>
/// Finds the document's borders and returns the perspective-corrected canvas.
/// </summary>
public sealed class DocDetector : IDisposable
{
    /// <summary>
    /// The share of the largest page's area a second segment must reach to be kept.
    ///
    /// <para>
    /// 0.6, from the reference. It is what stops a background blob being stitched onto a single-page
    /// document, and what allows the two halves of a passport spread to both survive.
    /// </para>
    /// </summary>
    private const double SecondSegmentAreaFraction = 0.6;

    private readonly SegmentationModel _model;

    public DocDetector(string root, IReadOnlyDictionary<string, string> paths, Device device,
        int threads)
        => _model = new SegmentationModel(
            Path.Combine(ModelPaths.Resolve(root, paths, "DocDetector"), "ONNX"), device, threads);

    /// <summary>
    /// Returns the corrected canvas and the SELECTED contours.
    ///
    /// <para>
    /// The contours travel out alongside the canvas so the conformance harness can compare them
    /// (<c>borders.segments</c>) and localise a divergence to the mask rather than to the warp. That
    /// distinction earned its keep immediately in the Go port: segments matched while the canvas was
    /// six pixels narrow, which placed the bug in the quadrilateral extraction and nowhere else.
    /// </para>
    ///
    /// <para>
    /// When no usable segment is found the ORIGINAL image is returned. Not a safety net bolted on —
    /// it is what the reference does, and a port that errored instead would fail every document whose
    /// borders the model cannot see.
    /// </para>
    /// </summary>
    public (Image Canvas, List<Point[]>? Segments) PredictTransform(Image image, int maxPages)
    {
        (List<Box> _, List<Point[]> segments) = _model.Predict(image);
        if (segments.Count == 0)
        {
            return (image.Clone(), null);
        }

        List<int> kept = SelectPages(segments, maxPages);
        if (kept.Count == 0)
        {
            return (image.Clone(), null);
        }

        var chosen = kept.Select(i => segments[i]).ToList();
        (Image? warped, bool ok) = Geometry.FixPerspective(image,
            chosen.Cast<IReadOnlyList<Point>>().ToList(), StackDirection.Auto,
            Geometry.DocMarginFraction);

        return ok && warped is not null
            ? (warped, chosen)
            : (image.Clone(), chosen);
    }

    /// <summary>
    /// Ranks segments by contour area and applies the area-fraction rule.
    ///
    /// <para>
    /// Returns indices in ASCENDING order, matching the reference's <c>sorted(keep)</c> — and that
    /// order then decides which page <see cref="Geometry.FixPerspective"/> treats as first when
    /// stitching a spread.
    /// </para>
    ///
    /// <para>
    /// The ranking sort is STABLE and descending: two segments of identical area keep their detection
    /// order, so the choice between them is deterministic.
    /// </para>
    /// </summary>
    private static List<int> SelectPages(List<Point[]> segments, int maxPages)
    {
        var areas = segments.Select(s => s.Length >= 3 ? Contours.ContourArea(s) : 0.0).ToList();

        // OrderByDescending is stable in LINQ; List.Sort is not.
        var order = Enumerable.Range(0, areas.Count).OrderByDescending(i => areas[i]).ToList();
        if (order.Count == 0 || areas[order[0]] <= 0)
        {
            return [];
        }

        int limit = Math.Max(1, maxPages);
        double maxArea = areas[order[0]];
        var keep = new List<int> { order[0] };
        foreach (int index in order.Skip(1))
        {
            if (keep.Count >= limit)
            {
                break;
            }
            if (areas[index] >= SecondSegmentAreaFraction * maxArea)
            {
                keep.Add(index);
            }
        }
        keep.Sort();
        return keep;
    }

    public void Dispose() => _model.Dispose();
}
