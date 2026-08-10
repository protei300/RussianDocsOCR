using OpenCvSharp;

namespace RussianDocs.DocumentProcessing.Imaging;

/// <summary>A point in image coordinates. Double, because contours carry sub-pixel positions.</summary>
public readonly record struct Point(double X, double Y);

/// <summary>
/// Contour extraction and the OpenCV calls around it.
///
/// <para>
/// **Every OpenCV default argument the reference relies on is passed EXPLICITLY here.** That is
/// CONVENTIONS trap 15 and it is not pedantry: <c>cv2.convexHull(cnt)</c> defaults to
/// <c>clockwise=False</c>, while OpenCvSharp, gocv and the JVM bindings all make the parameter
/// required — so a port CHOOSES a value instead of inheriting one. The Go port chose <c>true</c> and
/// lost six pixels of canvas width on an internal-passport spread, because the hull's ORIENTATION
/// decides which vertices Douglas-Peucker keeps. Audit against the Python call site, never against
/// the binding's own defaults.
/// </para>
/// </summary>
public static class Contours
{
    public static Point[] ToPoints(IEnumerable<Point2f> points) =>
        [.. points.Select(p => new Point(p.X, p.Y))];

    private static Point2f[] ToPoint2f(IReadOnlyList<Point> points) =>
        [.. points.Select(p => new Point2f((float)p.X, (float)p.Y))];

    private static OpenCvSharp.Point[] ToCvPoints(IReadOnlyList<Point> points) =>
        [.. points.Select(p => new OpenCvSharp.Point((int)p.X, (int)p.Y))];

    /// <summary>Otsu threshold. Returns the binary image and the chosen threshold value.</summary>
    public static (Image Binary, double Threshold) ThresholdOtsu(Image src, bool invert)
    {
        using Image gray = src.Channels == 1 ? src.Clone() : Io.ToGray(src);
        var binary = new Mat();
        ThresholdTypes type = (invert ? ThresholdTypes.BinaryInv : ThresholdTypes.Binary)
            | ThresholdTypes.Otsu;
        double value = Cv2.Threshold(gray.Mat, binary, 0, 255, type);
        return (Image.Wrap(binary), value);
    }

    /// <summary>
    /// External contours only, no hierarchy — <c>RETR_EXTERNAL</c> with <c>CHAIN_APPROX_SIMPLE</c>.
    /// </summary>
    public static List<Point[]> FindExternalContours(Image src)
    {
        Cv2.FindContours(src.Mat, out OpenCvSharp.Point[][] contours, out _,
            RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        return [.. contours.Select(c => c.Select(p => new Point(p.X, p.Y)).ToArray())];
    }

    /// <summary>Contour area, via the shoelace formula OpenCV uses.</summary>
    public static double ContourArea(IReadOnlyList<Point> points) =>
        points.Count < 3 ? 0 : Math.Abs(Cv2.ContourArea(ToPoint2f(points)));

    /// <summary>
    /// The convex hull. <c>clockwise: false</c> — the reference's default, and NOT the binding's.
    /// </summary>
    public static Point[] ConvexHull(IReadOnlyList<Point> points)
    {
        if (points.Count == 0)
        {
            return [];
        }
        // The Point2f overload always returns points, so only the orientation needs saying — and it
        // is the one that matters. OpenCvSharp has no default for it, unlike cv2.
        Point2f[] hull = Cv2.ConvexHull(ToPoint2f(points), clockwise: false);
        return ToPoints(hull);
    }

    /// <summary>Perimeter of a CLOSED contour — <c>cv2.arcLength(pts, True)</c>.</summary>
    public static double ArcLength(IReadOnlyList<Point> points) =>
        points.Count < 2 ? 0 : Cv2.ArcLength(ToPoint2f(points), closed: true);

    /// <summary>Douglas-Peucker simplification of a CLOSED contour.</summary>
    public static Point[] ApproxPolyDp(IReadOnlyList<Point> points, double epsilon) =>
        ToPoints(Cv2.ApproxPolyDP(ToPoint2f(points), epsilon, closed: true));

    /// <summary>
    /// The four corners of the minimum-area rectangle.
    ///
    /// <para>
    /// Taken from <c>BoxPoints</c> rather than reconstructed from centre/size/angle: OpenCV changed
    /// the <c>minAreaRect</c> angle convention around 4.5, and a hand-rolled version silently
    /// produces the corners in a different ORDER — which then feeds a perspective transform.
    /// </para>
    /// </summary>
    public static Point[] MinAreaRectPoints(IReadOnlyList<Point> points)
    {
        RotatedRect rect = Cv2.MinAreaRect(ToPoint2f(points));
        return ToPoints(Cv2.BoxPoints(rect));
    }

    /// <summary>Warps a quadrilateral onto an axis-aligned rectangle of the given size.</summary>
    public static Image WarpPerspectiveQuad(Image src, IReadOnlyList<Point> quad, int width, int height)
    {
        if (quad.Count != 4)
        {
            throw new ArgumentException($"imaging: warp needs 4 points, got {quad.Count}");
        }

        Point2f[] source = ToPoint2f(quad);
        Point2f[] destination =
        [
            new(0, 0),
            new(width - 1, 0),
            new(width - 1, height - 1),
            new(0, height - 1),
        ];

        using Mat transform = Cv2.GetPerspectiveTransform(source, destination);
        var dst = new Mat();
        Cv2.WarpPerspective(src.Mat, dst, transform, new Size(width, height));
        return Image.Wrap(dst);
    }

    /// <summary>Horizontal concatenation. Heights must match.</summary>
    public static Image HStack(Image a, Image b)
    {
        if (a.Height != b.Height)
        {
            throw new ArgumentException($"imaging: hstack heights differ: {a.Height} vs {b.Height}");
        }
        var dst = new Mat();
        Cv2.HConcat([a.Mat, b.Mat], dst);
        return Image.Wrap(dst);
    }

    /// <summary>Vertical concatenation. Widths must match.</summary>
    public static Image VStack(Image a, Image b)
    {
        if (a.Width != b.Width)
        {
            throw new ArgumentException($"imaging: vstack widths differ: {a.Width} vs {b.Width}");
        }
        var dst = new Mat();
        Cv2.VConcat([a.Mat, b.Mat], dst);
        return Image.Wrap(dst);
    }
}
