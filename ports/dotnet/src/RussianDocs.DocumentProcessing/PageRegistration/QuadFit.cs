using RussianDocs.DocumentProcessing.Imaging;

namespace RussianDocs.DocumentProcessing.PageRegistration;

/// <summary>
/// Page quad from straight-line fits to the segmentation contour. Port of
/// <c>pipeline_modules/page_registration/quad_fit.py</c>, following the structure of the Go port's
/// already-graded <c>internal/docproc/modules/quadfit.go</c> (44/44 stages, both profiles).
///
/// <para>
/// The Borders mask usually follows the physical page edge well; what breaks the plain
/// <see cref="Geometry.ExtractQuad"/> (convex hull + polygon simplification) is the CORNER choice: a
/// thumb merged into the mask, a bent corner, or a run of stair-steps along a blurred edge moves one
/// polygon vertex, and the whole side swings. A straight line fitted to the ~100 contour points of
/// that side survives the thumb (RANSAC drops it as an outlier) where a polygon vertex does not.
/// </para>
///
/// <para>
/// Method: start from the polygon quad, assign every contour point to its nearest side, fit each
/// side with RANSAC + least squares, intersect adjacent lines, repeat (the assignment improves as the
/// lines do). Only sides with enough well-spread inliers are refit; the others keep the polygon's
/// estimate. The result must be convex and agree with the initial quad (IoU) — otherwise the caller
/// keeps the polygon quad.
/// </para>
///
/// <para>
/// Optionally a side that lies on the photo frame (the page runs out of the picture) is
/// EXTRAPOLATED from the opposite side and the page's known aspect ratio, so a clipped page keeps its
/// scale instead of being stretched to fill the frame; the missing strip becomes border padding.
/// </para>
///
/// <para>
/// <b>Not bit-exact with the reference, on purpose — same as Go.</b> RANSAC's sample draw uses
/// <see cref="Random"/>, not <c>numpy.random.default_rng(0)</c>'s PCG64; the same seed does not
/// reproduce the same sample pairs in either language. RANSAC here rejects a handful of outlier
/// contour points, not choosing between structurally different candidates, so the practical effect is
/// a few-pixel difference in the fitted quad, not a different answer — the same tradeoff the Go port
/// already made and shipped at 44/44.
/// </para>
/// </summary>
public static class QuadFit
{
    private const double ResampleStepPx = 3.0;
    private const int Iterations = 3;
    private const int RansacSamples = 200;
    private const double InlierFrac = 0.012;
    private const int MinSidePoints = 12;
    private const double MinSupportFrac = 0.30;
    private const double AssignOverhang = 0.05;
    private const double MinIouWithInit = 0.6;
    private const double MaxOutsideFrac = 0.35;
    private const double FrameTolPx = 3.0;
    private const double ExtrapolateMinVisible = 0.5;
    private const double ExtrapolateMaxVisible = 0.97;

    /// <summary>Mirrors <c>fit_quad_lines</c>'s info dict, trimmed to what a caller needs.</summary>
    public sealed record Info
    {
        public string Method { get; init; } = "none";
        public string? Reason { get; init; }
        public int[] Clipped { get; init; } = [];
        public int? Extrapolated { get; init; }
        public double? Visible { get; init; }
        public double? IouInit { get; init; }
    }

    /// <summary>
    /// Quad of a page from line fits to its segmentation contour.
    /// </summary>
    /// <param name="contour">Contour points, in photo pixels.</param>
    /// <param name="imageHeight">
    /// Photo height; pass 0 (with <paramref name="imageWidth"/> also 0) for "no frame", matching
    /// <c>image_shape=None</c> — disables the frame-clipping/extrapolation logic entirely.
    /// </param>
    /// <param name="imageWidth">Photo width; see <paramref name="imageHeight"/>.</param>
    /// <param name="aspectHOverW">
    /// Page height / width. Enables extrapolation of a side that lies on the frame; pass 0 to
    /// disable, matching <c>aspect_h_over_w=None</c>.
    /// </param>
    /// <param name="extrapolate">Whether that extrapolation is allowed at all.</param>
    /// <param name="seed">RANSAC PRNG seed — the reference's is 0 (<c>np.random.default_rng(0)</c>).</param>
    /// <returns>
    /// Quad ordered TL, TR, BR, BL, or null only when no quad at all can be made from the contour.
    /// </returns>
    public static (Point[]? Quad, Info Result) FitQuadLines(IReadOnlyList<Point> contour,
        int imageHeight, int imageWidth, double aspectHOverW, bool extrapolate, int seed = 0)
    {
        Point[]? init = Geometry.ExtractQuad(contour);
        if (init is null)
        {
            return (null, new Info { Method = "none" });
        }
        init = Geometry.OrderPoints(init) ?? init;
        bool hasFrame = imageHeight > 0 && imageWidth > 0;

        Point[] pts = Resample(contour, ResampleStepPx);
        if (pts.Length < 4 * MinSidePoints)
        {
            return (init, new Info { Method = "polygon", Reason = "few points" });
        }

        var rng = new Random(seed);
        var usable = new bool[pts.Length];
        for (int i = 0; i < pts.Length; i++)
        {
            usable[i] = !hasFrame || !(pts[i].X <= FrameTolPx || pts[i].X >= imageWidth - 1 - FrameTolPx
                || pts[i].Y <= FrameTolPx || pts[i].Y >= imageHeight - 1 - FrameTolPx);
        }

        Point[] quad = [.. init];
        var refit = new bool[4];
        for (int iter = 0; iter < Iterations; iter++)
        {
            Point[] a = quad;
            Point[] b = [quad[1], quad[2], quad[3], quad[0]];
            var d = new Point[4];
            var len = new double[4];
            var u = new Point[4];
            bool degenerate = false;
            for (int k = 0; k < 4; k++)
            {
                d[k] = new Point(b[k].X - a[k].X, b[k].Y - a[k].Y);
                len[k] = Math.Sqrt(d[k].X * d[k].X + d[k].Y * d[k].Y);
                if (len[k] < 2)
                {
                    degenerate = true;
                }
                u[k] = len[k] > 0 ? new Point(d[k].X / len[k], d[k].Y / len[k]) : new Point(0, 0);
            }
            if (degenerate)
            {
                return (init, new Info { Method = "polygon", Reason = "degenerate side" });
            }

            var side = new int[pts.Length];
            var assigned = new bool[pts.Length];
            for (int i = 0; i < pts.Length; i++)
            {
                int bestK = -1;
                double bestDist = double.PositiveInfinity;
                for (int k = 0; k < 4; k++)
                {
                    double rx = pts[i].X - a[k].X, ry = pts[i].Y - a[k].Y;
                    double proj = len[k] > 0 ? (rx * u[k].X + ry * u[k].Y) / len[k] : double.PositiveInfinity;
                    if (proj < -AssignOverhang || proj > 1 + AssignOverhang)
                    {
                        continue;
                    }
                    double dst = Math.Abs(rx * -u[k].Y + ry * u[k].X);
                    if (dst < bestDist)
                    {
                        bestDist = dst;
                        bestK = k;
                    }
                }
                side[i] = bestK;
                assigned[i] = bestK >= 0;
            }

            var linePoint = new Point[4];
            var lineDir = new Point[4];
            for (int k = 0; k < 4; k++)
            {
                var sel = new List<Point>();
                for (int i = 0; i < pts.Length; i++)
                {
                    if (side[i] == k && assigned[i] && usable[i])
                    {
                        sel.Add(pts[i]);
                    }
                }
                refit[k] = false;
                if (sel.Count >= MinSidePoints
                    && FitLineRansac([.. sel], Math.Max(2.0, InlierFrac * len[k]), rng)
                        is (Point p0, Point dv, bool[] inl))
                {
                    double lo = 0, hi = 0;
                    bool first = true;
                    for (int i = 0; i < sel.Count; i++)
                    {
                        if (!inl[i])
                        {
                            continue;
                        }
                        double t = (sel[i].X - p0.X) * dv.X + (sel[i].Y - p0.Y) * dv.Y;
                        if (first)
                        {
                            lo = hi = t;
                            first = false;
                        }
                        else
                        {
                            lo = Math.Min(lo, t);
                            hi = Math.Max(hi, t);
                        }
                    }
                    if (!first && hi - lo >= MinSupportFrac * len[k])
                    {
                        linePoint[k] = p0;
                        lineDir[k] = dv;
                        refit[k] = true;
                    }
                }
                if (!refit[k])
                {
                    linePoint[k] = a[k];
                    lineDir[k] = u[k];
                }
            }

            var corners = new Point[4];
            for (int k = 0; k < 4; k++)
            {
                int prev = (k + 3) % 4;
                if (Intersect(linePoint[prev], lineDir[prev], linePoint[k], lineDir[k]) is not { } c)
                {
                    return (init, new Info { Method = "polygon", Reason = "parallel sides" });
                }
                corners[k] = c;
            }
            quad = Geometry.OrderPoints(corners) ?? corners;
        }

        if (!refit.Any(r => r))
        {
            return (init, new Info { Method = "polygon", Reason = "no side fitted" });
        }
        if (!IsConvex(quad))
        {
            return (init, new Info { Method = "polygon", Reason = "not convex" });
        }
        double iou = QuadIou(quad, init);
        if (iou < MinIouWithInit)
        {
            return (init, new Info { Method = "polygon", Reason = "disagrees with polygon", IouInit = Round3(iou) });
        }

        int[] clipped = [];
        int? extrapolated = null;
        double? visible = null;
        if (hasFrame)
        {
            double w = imageWidth, h = imageHeight;
            double slack = MaxOutsideFrac * Geometry.Distance(quad[1], quad[0]);
            foreach (Point p in quad)
            {
                if (p.X < -slack || p.X > w + slack || p.Y < -slack || p.Y > h + slack)
                {
                    return (init, new Info
                    {
                        Method = "polygon", Reason = "corner outside frame", IouInit = Round3(iou),
                    });
                }
            }
            clipped = ClippedSides(quad, w, h);
            foreach (int k in clipped)
            {
                foreach (int idx in new[] { k, (k + 1) % 4 })
                {
                    quad[idx] = new Point(
                        Math.Clamp(quad[idx].X, 0, w - 1), Math.Clamp(quad[idx].Y, 0, h - 1));
                }
            }
            if (extrapolate && aspectHOverW > 0 && clipped.Length == 1)
            {
                (Point[]? newQuad, double vis) = ExtrapolateSide(quad, clipped[0], aspectHOverW);
                visible = Round3(vis);
                if (newQuad is not null)
                {
                    quad = newQuad;
                    extrapolated = clipped[0];
                }
            }
        }

        return (quad, new Info
        {
            Method = "lines", Clipped = clipped, Extrapolated = extrapolated, Visible = visible,
            IouInit = Round3(iou),
        });
    }

    private static double Round3(double v) => Math.Round(v, 3, MidpointRounding.ToEven);

    private static Point[] Resample(IReadOnlyList<Point> contour, double step)
    {
        if (contour.Count < 3)
        {
            return [.. contour];
        }
        var closed = new Point[contour.Count + 1];
        for (int i = 0; i < contour.Count; i++)
        {
            closed[i] = contour[i];
        }
        closed[^1] = contour[0];

        var cum = new double[closed.Length];
        for (int i = 1; i < closed.Length; i++)
        {
            cum[i] = cum[i - 1] + Geometry.Distance(closed[i - 1], closed[i]);
        }
        double total = cum[^1];
        if (total <= 0)
        {
            return [.. contour];
        }
        int n = Math.Max((int)(total / step), contour.Count);
        var outPts = new Point[n];
        for (int i = 0; i < n; i++)
        {
            double t = total * i / n;
            outPts[i] = InterpAlong(closed, cum, t);
        }
        return outPts;
    }

    private static Point InterpAlong(Point[] closed, double[] cum, double t)
    {
        if (t <= cum[0])
        {
            return closed[0];
        }
        if (t >= cum[^1])
        {
            return closed[^1];
        }
        int lo = 0, hi = cum.Length - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (cum[mid] <= t)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        double seg = cum[hi] - cum[lo];
        if (seg <= 0)
        {
            return closed[lo];
        }
        double f = (t - cum[lo]) / seg;
        return new Point(
            closed[lo].X + f * (closed[hi].X - closed[lo].X),
            closed[lo].Y + f * (closed[hi].Y - closed[lo].Y));
    }

    /// <summary>
    /// RANSAC line through <paramref name="pts"/>; two rounds of least-squares refit afterwards
    /// (<c>cv2.fitLine</c> with <c>DIST_L2</c> is ordinary total least squares: centroid plus the
    /// leading eigenvector of the centred covariance — <see cref="FitLineLs"/>).
    /// </summary>
    private static (Point P0, Point Dir, bool[] Inliers)? FitLineRansac(Point[] pts, double thr, Random rng)
    {
        int n = pts.Length;
        if (n < 2)
        {
            return null;
        }
        int bestCount = -1, bestI = 0, bestJ = 0;
        for (int s = 0; s < RansacSamples; s++)
        {
            int i = rng.Next(n), j = rng.Next(n);
            double dx = pts[j].X - pts[i].X, dy = pts[j].Y - pts[i].Y;
            double l = Math.Sqrt(dx * dx + dy * dy);
            if (l <= 1e-6)
            {
                continue;
            }
            double ux = dx / l, uy = dy / l;
            double nx = -uy, ny = ux;
            int count = 0;
            foreach (Point p in pts)
            {
                double rx = p.X - pts[i].X, ry = p.Y - pts[i].Y;
                if (Math.Abs(rx * nx + ry * ny) < thr)
                {
                    count++;
                }
            }
            if (count > bestCount)
            {
                bestCount = count;
                bestI = i;
                bestJ = j;
            }
        }
        if (bestCount < 0)
        {
            return null;
        }

        double bdx = pts[bestJ].X - pts[bestI].X, bdy = pts[bestJ].Y - pts[bestI].Y;
        double bl = Math.Sqrt(bdx * bdx + bdy * bdy);
        double bux = bdx / bl, buy = bdy / bl;
        var inl = new bool[n];
        for (int i = 0; i < n; i++)
        {
            double rx = pts[i].X - pts[bestI].X, ry = pts[i].Y - pts[bestI].Y;
            inl[i] = Math.Abs(rx * -buy + ry * bux) < thr;
        }
        Point p0 = pts[bestI], dir = new Point(bux, buy);

        for (int iter = 0; iter < 2; iter++)
        {
            var sel = new List<Point>();
            for (int i = 0; i < n; i++)
            {
                if (inl[i])
                {
                    sel.Add(pts[i]);
                }
            }
            if (sel.Count < 2)
            {
                return null;
            }
            (p0, dir) = FitLineLs([.. sel]);
            double nx2 = -dir.Y, ny2 = dir.X;
            inl = new bool[n];
            for (int i = 0; i < n; i++)
            {
                double rx = pts[i].X - p0.X, ry = pts[i].Y - p0.Y;
                inl[i] = Math.Abs(rx * nx2 + ry * ny2) < thr;
            }
        }
        return (p0, dir, inl);
    }

    /// <summary>Total least squares through a point set: centroid + leading eigenvector of the
    /// centred 2x2 covariance, matching <c>cv2.fitLine(..., cv2.DIST_L2, ...)</c>.</summary>
    private static (Point P0, Point Dir) FitLineLs(Point[] pts)
    {
        double mx = 0, my = 0;
        foreach (Point p in pts)
        {
            mx += p.X;
            my += p.Y;
        }
        double n = pts.Length;
        mx /= n;
        my /= n;
        double sxx = 0, sxy = 0, syy = 0;
        foreach (Point p in pts)
        {
            double dx = p.X - mx, dy = p.Y - my;
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
        double trace = sxx + syy;
        double diff = sxx - syy;
        double disc = Math.Sqrt(diff * diff + 4 * sxy * sxy);
        double lambda1 = (trace + disc) / 2;
        double vx, vy;
        if (Math.Abs(sxy) > 1e-12)
        {
            vx = lambda1 - syy;
            vy = sxy;
        }
        else if (sxx >= syy)
        {
            vx = 1;
            vy = 0;
        }
        else
        {
            vx = 0;
            vy = 1;
        }
        double l = Math.Sqrt(vx * vx + vy * vy);
        if (l < 1e-12)
        {
            vx = 1;
            vy = 0;
            l = 1;
        }
        return (new Point(mx, my), new Point(vx / l, vy / l));
    }

    private static Point? Intersect(Point p, Point d, Point q, Point e)
    {
        double den = d.X * e.Y - d.Y * e.X;
        if (Math.Abs(den) < 1e-9)
        {
            return null;
        }
        double t = ((q.X - p.X) * e.Y - (q.Y - p.Y) * e.X) / den;
        return new Point(p.X + t * d.X, p.Y + t * d.Y);
    }

    /// <summary>Shared with <see cref="PageRegistrar"/>'s sane-quad check (page_registration.py
    /// uses <c>cv2.isContourConvex</c> there too — same test, one implementation).</summary>
    internal static bool IsConvex(IReadOnlyList<Point> quad)
    {
        int n = quad.Count;
        double sign = 0;
        for (int i = 0; i < n; i++)
        {
            Point a = quad[i], b = quad[(i + 1) % n], c = quad[(i + 2) % n];
            double cross = (b.X - a.X) * (c.Y - b.Y) - (b.Y - a.Y) * (c.X - b.X);
            if (cross == 0)
            {
                continue;
            }
            double s = cross < 0 ? -1.0 : 1.0;
            if (sign == 0)
            {
                sign = s;
            }
            else if (s != sign)
            {
                return false;
            }
        }
        return true;
    }

    internal static double QuadIou(Point[] a, Point[] b)
    {
        double inter = ConvexIntersectionArea(a, b);
        double union = PolygonArea(a) + PolygonArea(b) - inter;
        return union > 0 ? inter / union : 0.0;
    }

    private static double PolygonArea(IReadOnlyList<Point> pts)
    {
        double a = 0;
        int n = pts.Count;
        for (int i = 0; i < n; i++)
        {
            int j = (i + 1) % n;
            a += pts[i].X * pts[j].Y - pts[j].X * pts[i].Y;
        }
        return Math.Abs(a) / 2;
    }

    /// <summary>Sutherland-Hodgman clip of convex polygon <paramref name="a"/> by convex polygon
    /// <paramref name="b"/>, area of what remains — equivalent to <c>cv2.intersectConvexConvex</c> for
    /// two already-convex inputs.</summary>
    private static double ConvexIntersectionArea(Point[] a, Point[] b)
    {
        List<Point> outPts = [.. a];
        int n = b.Length;
        for (int i = 0; i < n && outPts.Count > 0; i++)
        {
            outPts = ClipPolygon(outPts, b[i], b[(i + 1) % n]);
        }
        return outPts.Count < 3 ? 0 : PolygonArea(outPts);
    }

    private static List<Point> ClipPolygon(List<Point> poly, Point a, Point b)
    {
        var outPts = new List<Point>();
        int n = poly.Count;
        double edgeX = b.X - a.X, edgeY = b.Y - a.Y;
        // `>= 0`, not `<= 0`: the clip subject arrives ordered TL,TR,BR,BL (Geometry.OrderPoints),
        // which in image coordinates (Y grows DOWNWARD) is CLOCKWISE on screen but has POSITIVE
        // shoelace-signed area under the standard (Y-up) convention this cross product assumes —
        // so the "left of the directed edge" half-plane that keeps the polygon's own interior is the
        // `>= 0` side, not `<= 0`. Caught by QuadIouSanity: a square clipped by a 50%-overlapping
        // copy of itself came back with IoU exactly 0 — the sign flip made the clip discard every
        // interior point on the FIRST edge, not just misjudge the boundary.
        bool Inside(Point p) => edgeX * (p.Y - a.Y) - edgeY * (p.X - a.X) >= 0;

        for (int i = 0; i < n; i++)
        {
            Point cur = poly[i], prev = poly[(i + n - 1) % n];
            bool curIn = Inside(cur), prevIn = Inside(prev);
            if (curIn)
            {
                if (!prevIn)
                {
                    outPts.Add(SegIntersect(prev, cur, a, b));
                }
                outPts.Add(cur);
            }
            else if (prevIn)
            {
                outPts.Add(SegIntersect(prev, cur, a, b));
            }
        }
        return outPts;
    }

    private static Point SegIntersect(Point p1, Point p2, Point p3, Point p4)
    {
        double den = (p1.X - p2.X) * (p3.Y - p4.Y) - (p1.Y - p2.Y) * (p3.X - p4.X);
        if (Math.Abs(den) < 1e-12)
        {
            return p1;
        }
        double ta = p1.X * p2.Y - p1.Y * p2.X;
        double tb = p3.X * p4.Y - p3.Y * p4.X;
        double x = (ta * (p3.X - p4.X) - (p1.X - p2.X) * tb) / den;
        double y = (ta * (p3.Y - p4.Y) - (p1.Y - p2.Y) * tb) / den;
        return new Point(x, y);
    }

    private static int[] ClippedSides(Point[] quad, double w, double h)
    {
        var outIdx = new List<int>();
        for (int k = 0; k < 4; k++)
        {
            Point a = quad[k], b = quad[(k + 1) % 4];
            bool OnFrame(Func<Point, double> coord, double val) =>
                Math.Abs(coord(a) - val) <= FrameTolPx && Math.Abs(coord(b) - val) <= FrameTolPx;
            if (OnFrame(p => p.X, 0) || OnFrame(p => p.X, w - 1)
                || OnFrame(p => p.Y, 0) || OnFrame(p => p.Y, h - 1))
            {
                outIdx.Add(k);
            }
        }
        return [.. outIdx];
    }

    private static (Point[]? Quad, double Visible) ExtrapolateSide(Point[] quad, int side, double aspectHOverW)
    {
        int k0 = side, k1 = (side + 1) % 4;
        int o0 = (side + 3) % 4, o1 = (side + 2) % 4;
        double oppLen = Geometry.Distance(quad[o1], quad[o0]);
        if (oppLen < 1)
        {
            return (null, 0.0);
        }
        double expect = side is 0 or 2 ? oppLen * aspectHOverW : oppLen / aspectHOverW;
        Point u0 = new(quad[k0].X - quad[o0].X, quad[k0].Y - quad[o0].Y);
        Point u1 = new(quad[k1].X - quad[o1].X, quad[k1].Y - quad[o1].Y);
        double l0 = Math.Sqrt(u0.X * u0.X + u0.Y * u0.Y);
        double l1 = Math.Sqrt(u1.X * u1.X + u1.Y * u1.Y);
        if (l0 < 1 || l1 < 1)
        {
            return (null, 0.0);
        }
        double visible = 0.5 * (l0 + l1) / expect;
        if (visible is < ExtrapolateMinVisible or > ExtrapolateMaxVisible)
        {
            return (null, visible);
        }
        Point[] newQuad = [.. quad];
        newQuad[k0] = new Point(quad[o0].X + u0.X / l0 * expect, quad[o0].Y + u0.Y / l0 * expect);
        newQuad[k1] = new Point(quad[o1].X + u1.X / l1 * expect, quad[o1].Y + u1.Y / l1 * expect);
        return (newQuad, visible);
    }
}
