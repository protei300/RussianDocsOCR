using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Imaging;

/// <summary>How a multi-page spread is joined back together.</summary>
public enum StackDirection
{
    /// <summary>Decide from the page geometry, which is what the reference does.</summary>
    Auto,
    Horizontal,
    Vertical,
}

/// <summary>
/// Quadrilateral geometry: ordering corners, expanding a margin, and the perspective correction.
/// </summary>
public static class Geometry
{
    /// <summary>
    /// The outward cushion applied to a detected document quad.
    ///
    /// <para>
    /// 0.01, from DOC_MARGIN_FRAC in the reference. It is a fraction of the document's OWN size that
    /// each EDGE moves out by, so the applied scale is 1 + 2*margin. I first wrote 0.005 from memory
    /// and every single-page canvas came out about 1 % large — 910 columns against the golden 901 —
    /// which is the whole error, visible only because the shape is compared exactly.
    /// </para>
    /// </summary>
    public const double DocMarginFraction = 0.01;

    /// <summary>
    /// Orders four points as top-left, top-right, bottom-right, bottom-left.
    ///
    /// <para>
    /// By coordinate SUM and DIFFERENCE, exactly as the reference: the smallest x+y is top-left, the
    /// largest is bottom-right, and the extremes of y-x give the other two. It is not a sort by angle
    /// and it does not generalise — but it is what produced the goldens.
    /// </para>
    ///
    /// <para>
    /// Ties resolve to the FIRST index reaching the extreme, because the comparisons are strict. On a
    /// perfectly axis-aligned rectangle two corners can share a sum, and picking the later one
    /// rotates the whole quad.
    /// </para>
    /// </summary>
    public static Point[]? OrderPoints(IReadOnlyList<Point> points)
    {
        if (points.Count != 4)
        {
            return null;
        }

        int minSum = 0, maxSum = 0, minDiff = 0, maxDiff = 0;
        for (int i = 0; i < points.Count; i++)
        {
            double sum = points[i].X + points[i].Y;
            double diff = points[i].Y - points[i].X;
            if (sum < points[minSum].X + points[minSum].Y)
            {
                minSum = i;
            }
            if (sum > points[maxSum].X + points[maxSum].Y)
            {
                maxSum = i;
            }
            if (diff < points[minDiff].Y - points[minDiff].X)
            {
                minDiff = i;
            }
            if (diff > points[maxDiff].Y - points[maxDiff].X)
            {
                maxDiff = i;
            }
        }
        return [points[minSum], points[minDiff], points[maxSum], points[maxDiff]];
    }

    /// <summary>
    /// Reduces a contour to four corners.
    ///
    /// <para>
    /// Tries increasing Douglas-Peucker tolerances until one yields exactly four points, and falls
    /// back to the minimum-area rectangle of the ORIGINAL contour — not of the hull. The fraction
    /// ladder is the reference's and the order matters: a coarser tolerance can also produce four
    /// points, but different ones.
    /// </para>
    /// </summary>
    public static Point[]? ExtractQuad(IReadOnlyList<Point> contour)
    {
        if (contour.Count < 4)
        {
            return null;
        }

        Point[] hull = Contours.ConvexHull(contour);
        if (hull.Length == 0)
        {
            return null;
        }

        double perimeter = Contours.ArcLength(hull);
        foreach (double fraction in new[] { 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15 })
        {
            Point[] approx = Contours.ApproxPolyDp(hull, fraction * perimeter);
            if (approx.Length == 4)
            {
                return approx;
            }
        }
        return Contours.MinAreaRectPoints(contour);
    }

    /// <summary>Scales a quadrilateral outward from its centroid by a fraction of its size.</summary>
    public static Point[] ExpandQuad(IReadOnlyList<Point> quad, double margin)
    {
        if (margin <= 0)
        {
            return [.. quad];
        }

        double cx = 0, cy = 0;
        foreach (Point p in quad)
        {
            cx += p.X;
            cy += p.Y;
        }
        cx /= quad.Count;
        cy /= quad.Count;

        double scale = 1.0 + 2.0 * margin;
        return [.. quad.Select(p => new Point(cx + (p.X - cx) * scale, cy + (p.Y - cy) * scale))];
    }

    /// <summary>
    /// Warps a quadrilateral to an axis-aligned image.
    ///
    /// <para>
    /// The output size comes from the LONGER of each opposing pair of edges, rounded HALF TO EVEN.
    /// Rounding away from zero here — Go's default, and easy to reach for in any language — gives a
    /// canvas one pixel different in each dimension, and every box downstream is then compared
    /// against a golden made on a differently-sized canvas.
    /// </para>
    /// </summary>
    public static (Image? Warped, bool Ok) FourPointTransform(Image image, IReadOnlyList<Point> quad)
    {
        Point[]? rect = OrderPoints(quad);
        if (rect is null)
        {
            return (null, false);
        }

        Point tl = rect[0], tr = rect[1], br = rect[2], bl = rect[3];
        int width = PyNum.RoundHalfEvenToInt(Math.Max(Distance(br, bl), Distance(tr, tl)));
        int height = PyNum.RoundHalfEvenToInt(Math.Max(Distance(tr, br), Distance(tl, bl)));
        if (width < 2 || height < 2)
        {
            return (null, false);
        }

        try
        {
            return (Contours.WarpPerspectiveQuad(image, rect, width, height), true);
        }
        catch (Exception)
        {
            // A degenerate quad makes GetPerspectiveTransform throw. The reference returns the
            // original image in that case rather than failing the document, so the caller needs a
            // false here, not an exception.
            return (null, false);
        }
    }

    public static double Distance(Point a, Point b) =>
        Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

    /// <summary>
    /// Corrects perspective for one or more detected pages and stitches them together.
    ///
    /// <para>
    /// Single page: order, expand by the margin, warp. Two pages: warp each, then join — HORIZONTALLY
    /// when the pages sit side by side and VERTICALLY when they are stacked, decided from the
    /// centroids. The join resizes nothing, so the shared dimension must already match; when it does
    /// not, the smaller value wins and the canvas is that much narrower. That is where the Go port's
    /// six-pixel discrepancy showed up, and the cause was upstream in the hull orientation.
    /// </para>
    /// </summary>
    public static (Image? Canvas, bool Ok) FixPerspective(Image image,
        IReadOnlyList<IReadOnlyList<Point>> segments, StackDirection direction, double margin)
    {
        var pages = new List<(Point[] Quad, Image Warped)>();
        try
        {
            foreach (IReadOnlyList<Point> segment in segments)
            {
                Point[]? quad = ExtractQuad(segment);
                if (quad is null)
                {
                    continue;
                }

                // ORDER FIRST, then expand, then CLAMP to the image. All three steps and their order
                // are the reference's: expanding an unordered quad moves the corners about its
                // centroid correctly but hands FourPointTransform points it will reorder anyway, and
                // skipping the clamp lets the cushion push a corner outside the image, where the warp
                // samples the border colour and widens the canvas.
                Point[]? rect = OrderPoints(quad);
                if (rect is null)
                {
                    continue;
                }
                rect = ExpandQuad(rect, margin);
                for (int i = 0; i < rect.Length; i++)
                {
                    rect[i] = new Point(
                        Math.Clamp(rect[i].X, 0, image.Width),
                        Math.Clamp(rect[i].Y, 0, image.Height));
                }

                (Image? warped, bool ok) = FourPointTransform(image, rect);
                if (!ok || warped is null)
                {
                    continue;
                }
                pages.Add((rect, warped));
            }

            if (pages.Count == 0)
            {
                return (null, false);
            }
            if (pages.Count == 1)
            {
                Image only = pages[0].Warped;
                pages.Clear(); // ownership moves to the caller
                return (only, true);
            }

            // Direction from the FIRST TWO pages' centroids only, matching the reference. A wider
            // horizontal separation means the pages sit side by side.
            StackDirection resolved = direction;
            if (direction == StackDirection.Auto)
            {
                Point c0 = Centroid(pages[0].Quad), c1 = Centroid(pages[1].Quad);
                resolved = Math.Abs(c0.X - c1.X) >= Math.Abs(c0.Y - c1.Y)
                    ? StackDirection.Horizontal
                    : StackDirection.Vertical;
            }

            // Ordered by the quad's MINIMUM coordinate, not its centroid: two pages of different
            // sizes can have centroids in the opposite order to their left edges.
            bool horizontal = resolved == StackDirection.Horizontal;
            var ordered = horizontal
                ? pages.OrderBy(pg => pg.Quad.Min(pt => pt.X)).ToList()
                : pages.OrderBy(pg => pg.Quad.Min(pt => pt.Y)).ToList();

            // **The pages are RESIZED to a common dimension before joining.** This is the step whose
            // absence produced a 727x528 canvas against the golden's 701x505: hconcat and vconcat
            // require the shared dimension to match exactly, so the reference scales every page to
            // the SMALLEST of them and scales the other axis proportionally, rounding half to even.
            int common = horizontal
                ? ordered.Min(pg => pg.Warped.Height)
                : ordered.Min(pg => pg.Warped.Width);

            var scaled = new List<Image>(ordered.Count);
            try
            {
                foreach ((Point[] _, Image warped) in ordered)
                {
                    int other = horizontal
                        ? Math.Max(1, PyNum.RoundHalfEvenToInt(
                            (double)warped.Width * common / warped.Height))
                        : Math.Max(1, PyNum.RoundHalfEvenToInt(
                            (double)warped.Height * common / warped.Width));
                    scaled.Add(horizontal
                        ? Io.Resize(warped, other, common, Interpolation.Linear)
                        : Io.Resize(warped, common, other, Interpolation.Linear));
                }

                Image joined = scaled[0].Clone();
                for (int k = 1; k < scaled.Count; k++)
                {
                    Image combined = horizontal
                        ? Contours.HStack(joined, scaled[k])
                        : Contours.VStack(joined, scaled[k]);
                    joined.Dispose();
                    joined = combined;
                }
                return (joined, true);
            }
            finally
            {
                foreach (Image part in scaled)
                {
                    part.Dispose();
                }
            }
        }
        catch (ArgumentException)
        {
            return (null, false);
        }
        finally
        {
            foreach ((Point[] _, Image warped) in pages)
            {
                warped.Dispose();
            }
        }
    }

    private static Point Centroid(IReadOnlyList<Point> quad)
    {
        double cx = 0, cy = 0;
        foreach (Point p in quad)
        {
            cx += p.X;
            cy += p.Y;
        }
        return new Point(cx / quad.Count, cy / quad.Count);
    }

    private static double SpreadX(List<Point[]> quads) =>
        quads.Max(q => Centroid(q).X) - quads.Min(q => Centroid(q).X);

    private static double SpreadY(List<Point[]> quads) =>
        quads.Max(q => Centroid(q).Y) - quads.Min(q => Centroid(q).Y);
}
