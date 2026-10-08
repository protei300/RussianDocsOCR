using RussianDocs.DocumentProcessing.Maps;
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
    /// <see cref="ExpandQuad"/> in FLOAT32 arithmetic. <c>expand_quad</c> is applied to the float32
    /// array that <c>order_points</c> (and the page registrar's <c>page_quads</c>) returns, and NumPy
    /// then does every step in float32: the centroid <c>mean(axis=0)</c> (rows added in order, then
    /// divided by 4), <c>quad - centre</c>, the product with the margin scale (a Python float is weak
    /// next to a float32 array, so it is cast to float32 first) and the final sum — each step rounds.
    /// In double the corners land up to ~6e-5 px elsewhere, and <c>warpPerspective</c> quantises its
    /// interpolation weights to 1/32 px, so that is enough to move a few hundred pixels of a 600x900
    /// canvas by one grey level or more (measured on BIRTHCERT_1998 once the document crop shifted the
    /// quad: 393 pixels, max 8 — a digest mismatch on <c>borders.canvas</c> and, through the field
    /// detector, a 0.007 confidence step downstream).
    /// </summary>
    public static Point[] ExpandQuadF32(IReadOnlyList<Point> quad, double margin)
    {
        if (margin <= 0)
        {
            return [.. quad];
        }

        float cx = 0f, cy = 0f;
        foreach (Point p in quad)
        {
            cx += (float)p.X;
            cy += (float)p.Y;
        }
        cx /= quad.Count;
        cy /= quad.Count;

        float scale = (float)(1.0 + 2.0 * margin);
        return [.. quad.Select(p =>
        {
            float x = (float)p.X, y = (float)p.Y;
            float nx = cx + (x - cx) * scale;
            float ny = cy + (y - cy) * scale;
            return new Point(nx, ny);
        })];
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
        (Image? warped, bool ok, _) = FourPointTransformWithMatrix(image, quad);
        return (warped, ok);
    }

    /// <summary><see cref="FourPointTransform"/>, also giving the matrix of the warp (input to output).</summary>
    public static (Image? Warped, bool Ok, double[,]? Matrix) FourPointTransformWithMatrix(Image image,
        IReadOnlyList<Point> quad)
    {
        Point[]? rect = OrderPoints(quad);
        if (rect is null)
        {
            return (null, false, null);
        }

        Point tl = rect[0], tr = rect[1], br = rect[2], bl = rect[3];
        int width = PyNum.RoundHalfEvenToInt(Math.Max(Distance(br, bl), Distance(tr, tl)));
        int height = PyNum.RoundHalfEvenToInt(Math.Max(Distance(tr, br), Distance(tl, bl)));
        if (width < 2 || height < 2)
        {
            return (null, false, null);
        }

        try
        {
            Image warped = Contours.WarpPerspectiveQuad(image, rect, width, height, out double[,] matrix);
            return (warped, true, matrix);
        }
        catch (Exception)
        {
            // A degenerate quad makes GetPerspectiveTransform throw. The reference returns the
            // original image in that case rather than failing the document, so the caller needs a
            // false here, not an exception.
            return (null, false, null);
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
    public static (Image? Canvas, bool Ok, IMap? Map) FixPerspective(Image image,
        IReadOnlyList<IReadOnlyList<Point>> segments, StackDirection direction, double margin)
    {
        var pages = new List<(Point[] Quad, Image Warped, double[,] Matrix)>();
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
                rect = ExpandQuadF32(rect, margin);
                for (int i = 0; i < rect.Length; i++)
                {
                    rect[i] = new Point(
                        Math.Clamp(rect[i].X, 0, image.Width),
                        Math.Clamp(rect[i].Y, 0, image.Height));
                }

                (Image? warped, bool ok, double[,]? matrix) = FourPointTransformWithMatrix(image, rect);
                if (!ok || warped is null)
                {
                    continue;
                }
                pages.Add((rect, warped, matrix!));
            }

            if (pages.Count == 0)
            {
                return (null, false, null);
            }
            IMap[] pageMaps = [.. pages.Select(pg => (IMap)new HomographyMap(pg.Matrix))];
            (int W, int H)[] sizes = [.. pages.Select(pg => (pg.Warped.Width, pg.Warped.Height))];
            if (pages.Count == 1)
            {
                Image only = pages[0].Warped;
                pages.Clear(); // ownership moves to the caller
                return (only, true, Stitched.Geometry(sizes, [new Placement(1.0, 0.0, 0.0)], pageMaps));
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
            List<int> order = horizontal
                ? [.. Enumerable.Range(0, pages.Count).OrderBy(i => pages[i].Quad.Min(pt => pt.X))]
                : [.. Enumerable.Range(0, pages.Count).OrderBy(i => pages[i].Quad.Min(pt => pt.Y))];

            // **The pages are RESIZED to a common dimension before joining.** This is the step whose
            // absence produced a 727x528 canvas against the golden's 701x505: hconcat and vconcat
            // require the shared dimension to match exactly, so the reference scales every page to
            // the SMALLEST of them and scales the other axis proportionally, rounding half to even.
            (Placement[] placements, int[] newW, int[] newH) = Stitched.Place(sizes, order, horizontal);

            var scaled = new List<Image>(order.Count);
            try
            {
                foreach (int i in order)
                {
                    scaled.Add(Io.Resize(pages[i].Warped, newW[i], newH[i], Interpolation.Linear));
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
                return (joined, true, Stitched.Geometry(sizes, placements, pageMaps));
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
            return (null, false, null);
        }
        finally
        {
            foreach ((Point[] _, Image warped, double[,] _) in pages)
            {
                warped.Dispose();
            }
        }
    }

    /// <summary>
    /// Merges already-rectified pages into one canvas. Port of
    /// <c>image_transformation.stitch_pages</c> — split out from <see cref="FixPerspective"/>'s own
    /// join logic (rather than having <c>FixPerspective</c> call this) so that logic, already
    /// verified against the 8 non-INTPASSPORT conformance cases, stays untouched; this is a second,
    /// independent caller for <see cref="PageRegistration.PageRegistrar"/>'s <c>_register_pages</c>
    /// port, which needs to stitch pages that were NOT produced by <c>FixPerspective</c>'s own
    /// per-segment warp (they come from template registration or a Borders quad warped separately).
    /// </summary>
    /// <param name="pages">Rectified pages, in detection order. Not disposed here — the caller owns
    /// them before and after this call.</param>
    /// <param name="quads">The photo quad each page came from, same order — used only to pick the
    /// stitch direction (on <see cref="StackDirection.Auto"/>) and the page order.</param>
    public static Image? StitchPages(IReadOnlyList<Image> pages, IReadOnlyList<Point[]> quads,
        StackDirection direction) =>
        StitchPagesPlaced(pages, quads, direction).Canvas;

    /// <summary>
    /// <see cref="StitchPages"/>, also giving where each page landed (<c>stitch_pages</c>' second result):
    /// one <see cref="Placement"/> per INPUT page, in the order of <paramref name="pages"/>.
    /// </summary>
    public static (Image? Canvas, Placement[] Placements) StitchPagesPlaced(IReadOnlyList<Image> pages,
        IReadOnlyList<Point[]> quads, StackDirection direction)
    {
        if (pages.Count == 0)
        {
            return (null, []);
        }
        if (pages.Count == 1)
        {
            return (pages[0].Clone(), [new Placement(1.0, 0.0, 0.0)]);
        }

        StackDirection resolved = direction;
        if (direction == StackDirection.Auto)
        {
            Point c0 = Centroid(quads[0]), c1 = Centroid(quads[1]);
            resolved = Math.Abs(c0.X - c1.X) >= Math.Abs(c0.Y - c1.Y)
                ? StackDirection.Horizontal
                : StackDirection.Vertical;
        }
        bool horizontal = resolved == StackDirection.Horizontal;
        List<int> ordered = horizontal
            ? [.. Enumerable.Range(0, pages.Count).OrderBy(i => quads[i].Min(pt => pt.X))]
            : [.. Enumerable.Range(0, pages.Count).OrderBy(i => quads[i].Min(pt => pt.Y))];

        (int W, int H)[] sizes = [.. pages.Select(p => (p.Width, p.Height))];
        (Placement[] placements, int[] newW, int[] newH) = Stitched.Place(sizes, ordered, horizontal);

        var scaled = new List<Image>(ordered.Count);
        try
        {
            foreach (int i in ordered)
            {
                scaled.Add(Io.Resize(pages[i], newW[i], newH[i], Interpolation.Linear));
            }

            Image joined = scaled[0].Clone();
            for (int k = 1; k < scaled.Count; k++)
            {
                Image combined = horizontal ? Contours.HStack(joined, scaled[k]) : Contours.VStack(joined, scaled[k]);
                joined.Dispose();
                joined = combined;
            }
            return (joined, placements);
        }
        finally
        {
            foreach (Image part in scaled)
            {
                part.Dispose();
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
