using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Tensors;

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

    /// <summary>
    /// A segment with (almost) no ink inside it is not a document page, however confident the model
    /// is: the white lid of a flatbed scanner next to a passport scores 0.90-0.96 as "Document" and,
    /// being 3x larger than a page, used to win <see cref="SelectPages"/>'s area rule and push both
    /// real pages out (~20 of 100 scans in the Damir set came out as an empty canvas). Ink is the
    /// mean gradient magnitude inside the eroded mask at <see cref="InkScalePx"/>: measured 1.5-5.4
    /// on lids, 19-101 on passport pages, 24-43 on the mostly bare registration page
    /// (doc_detector.py:8-26). A blank segment is dropped only when another segment with real ink
    /// exists, so a lone blank sheet still goes through as before.
    /// </summary>
    private const double BlankInk = 8.0;

    private const int InkScalePx = 400;

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

        // First drop segments without ink (a scanner lid, a blank sheet next to the document), THEN
        // apply the area rule below — matching the reference's order (doc_detector.py:125-131).
        segments = DropBlankSegments(image, segments);

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

    /// <summary>
    /// Indices to keep: blank segments (ink &lt; <see cref="BlankInk"/>) are dropped when at least one
    /// inked segment exists. Port of <c>doc_detector.drop_blank_segments</c> (doc_detector.py:48-58).
    /// </summary>
    private static List<Point[]> DropBlankSegments(Image image, List<Point[]> segments)
    {
        using Image gray = Io.ToGray(image);
        List<double> ink = [.. segments.Select(s => s.Length >= 3 ? SegmentInk(gray.Mat, s) : 0.0)];
        if (!ink.Any(v => v >= BlankInk))
        {
            return segments;
        }
        return [.. segments.Where((_, i) => ink[i] >= BlankInk)];
    }

    /// <summary>
    /// Mean gradient magnitude inside <paramref name="contour"/> (eroded so the segment's own edge
    /// does not count), on the image downscaled to <see cref="InkScalePx"/>. Port of
    /// <c>doc_detector.segment_ink</c> (doc_detector.py:29-45).
    ///
    /// <para>
    /// OpenCvSharp types are fully qualified rather than brought in with <c>using OpenCvSharp;</c>,
    /// because this file's own <see cref="Point"/> (double-precision, from
    /// <c>RussianDocs.DocumentProcessing.Imaging</c>) would otherwise collide with
    /// <c>OpenCvSharp.Point</c> at every one of this file's many existing bare uses of the name.
    /// </para>
    /// </summary>
    private static double SegmentInk(OpenCvSharp.Mat gray, Point[] contour)
    {
        int h = gray.Rows, w = gray.Cols;
        double s = InkScalePx / (double)Math.Max(h, w);
        // `max(1, int(w * s))` in the reference — TRUNCATION, not rounding.
        int newW = Math.Max(1, (int)(w * s));
        int newH = Math.Max(1, (int)(h * s));

        using var small = new OpenCvSharp.Mat();
        OpenCvSharp.Cv2.Resize(gray, small, new OpenCvSharp.Size(newW, newH),
            interpolation: OpenCvSharp.InterpolationFlags.Area);

        OpenCvSharp.Point[] scaled =
        [
            .. contour.Select(p => new OpenCvSharp.Point(
                PyNum.RoundHalfEvenToInt(p.X * s), PyNum.RoundHalfEvenToInt(p.Y * s))),
        ];

        using var mask = new OpenCvSharp.Mat(small.Size(), OpenCvSharp.MatType.CV_8UC1,
            OpenCvSharp.Scalar.All(0));
        OpenCvSharp.Cv2.FillPoly(mask, new[] { scaled }, OpenCvSharp.Scalar.All(255));
        using var kernel = new OpenCvSharp.Mat(5, 5, OpenCvSharp.MatType.CV_8UC1, OpenCvSharp.Scalar.All(1));
        OpenCvSharp.Cv2.Erode(mask, mask, kernel);

        if (OpenCvSharp.Cv2.CountNonZero(mask) == 0)
        {
            return 0.0;
        }

        using var gx = new OpenCvSharp.Mat();
        using var gy = new OpenCvSharp.Mat();
        OpenCvSharp.Cv2.Sobel(small, gx, OpenCvSharp.MatType.CV_32F, 1, 0, ksize: 3);
        OpenCvSharp.Cv2.Sobel(small, gy, OpenCvSharp.MatType.CV_32F, 0, 1, ksize: 3);

        using var magnitude = new OpenCvSharp.Mat();
        OpenCvSharp.Cv2.Magnitude(gx, gy, magnitude);

        return OpenCvSharp.Cv2.Mean(magnitude, mask).Val0;
    }

    public void Dispose() => _model.Dispose();
}
