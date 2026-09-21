using System.Text.Json.Serialization;
using OpenCvSharp;
using OpenCvSharp.Features2D;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Tensors;
using Point = RussianDocs.DocumentProcessing.Imaging.Point;

namespace RussianDocs.DocumentProcessing.PageRegistration;

/// <summary>
/// Template-based page registration for booklet documents (internal passport). Port of
/// <c>pipeline_modules/page_registration/page_registration.py</c>, following the structure of the
/// Go port's already-graded <c>internal/docproc/modules/pageregistration.go</c> (44/44 stages, both
/// profiles) — this file mirrors <c>PageTemplate</c>/<c>PageRegistrar</c> the same way that file does.
///
/// <para>
/// The Borders path rectifies each page from its segmentation contour: a quad is fitted to the mask
/// and warped to a rectangle from the quad's own side lengths. Everything downstream inherits the
/// mask's errors. This module instead aligns the PRINTED BLANK of the page to a canonical template
/// (<c>templates/</c>, person-free): SIFT features of the photo are matched against the template's
/// static print (ratio test + MAGSAC homography) for a coarse candidate, then refined in the
/// template's own canonical frame (re-detect, re-match with a shrinking search window). See the
/// Python module docstring for the full four-step method.
/// </para>
///
/// <para>
/// <b>Not ported, matching the Go port's own documented gap:</b> the dense ECC polish (<c>_ecc</c>,
/// <c>_gradient</c>) and anything feeding only it. The pipeline never exercises it at the defaults
/// this port targets (<c>PageRegistrar(use_ecc=False)</c> is what <c>pipeline.py</c> constructs); a
/// future caller wanting <c>use_ecc=True</c> would need to add it.
/// </para>
/// </summary>
public sealed class PageRegistrar : IDisposable
{
    private const int MinCoarseInliers = 8;
    private const int MinRefinedInliers = 18;
    private const int MinRefinedInliersChain = 10;
    private const double MinSpreadPxChain = 140.0;
    private const double MinSpreadPx = 120.0;
    private const double RatioTest = 0.8;
    private const double CoarseReprojPx = 4.0;
    private const double ChainGapFrac = 0.03;
    private const double QuadDilateFrac = 0.10;
    private const int MaxCoarseCandidates = 2;
    private const int GoodCoarseInliers = 20;
    private const double PageMarginFrac = 0.03;

    private static readonly double[] RefineReprojPx = [3.0, 2.0];
    private static readonly double[] RefineRadiusPx = [45.0, 15.0];
    private static readonly double[] ChainRadiusPx = [90.0, 15.0];

    private readonly SIFT _sift;
    private readonly BFMatcher _matcher;
    private readonly List<(string Name, List<PageTemplate> Refs)> _pages = [];

    public int PageW { get; }
    public int PageH { get; }
    private readonly int _margin;
    private readonly int _outW;
    private readonly int _outH;

    public bool LineQuads { get; init; } = true;
    public bool LineRefine { get; init; } = true;
    public bool LineDewarp { get; init; } = true;

    public IReadOnlyList<string> PageNames => [.. _pages.Select(p => p.Name)];

    /// <summary>
    /// Loads <c>templates/&lt;docType&gt;.json</c> (lowercase) from
    /// <c>document_processing/pipeline_modules/page_registration/templates</c> under
    /// <paramref name="root"/>, and builds SIFT features for every reference page. Matches
    /// <c>PageRegistrar()</c>'s own defaults (<c>nfeatures=6000, use_ecc=False, line_quads=True,
    /// line_refine=True, line_dewarp=True</c>).
    /// </summary>
    public PageRegistrar(string root, string docType, int nfeatures = 6000)
    {
        string dir = Path.Combine(root, "document_processing", "pipeline_modules",
            "page_registration", "templates");
        string metaPath = Path.Combine(dir, docType.ToLowerInvariant() + ".json");
        TemplateMeta meta = System.Text.Json.JsonSerializer.Deserialize<TemplateMeta>(
            File.ReadAllText(metaPath))
            ?? throw new InvalidDataException($"page_registration: {metaPath} did not parse");
        if (meta.Pages.Length == 0)
        {
            throw new InvalidDataException($"page_registration: {metaPath} has no pages");
        }

        _sift = SIFT.Create(nFeatures: nfeatures);
        _matcher = new BFMatcher(NormTypes.L2, crossCheck: false);

        try
        {
            foreach (TemplatePage p in meta.Pages)
            {
                TemplateRef[] refs = p.Refs.Length > 0
                    ? p.Refs
                    : [new TemplateRef { Image = p.Image!, Mask = p.Mask! }];
                var loaded = new List<PageTemplate>(refs.Length);
                foreach (TemplateRef r in refs)
                {
                    loaded.Add(PageTemplate.Load(p.Name,
                        Path.Combine(dir, r.Image), Path.Combine(dir, r.Mask), _sift));
                }
                _pages.Add((p.Name, loaded));
            }
        }
        catch
        {
            foreach ((string _, List<PageTemplate> refs) in _pages)
            {
                foreach (PageTemplate t in refs)
                {
                    t.Dispose();
                }
            }
            _sift.Dispose();
            _matcher.Dispose();
            throw;
        }

        PageTemplate first = _pages[0].Refs[0];
        PageW = first.Width;
        PageH = first.Height;
        _margin = PyNum.RoundHalfEvenToInt(PageMarginFrac * PageW);
        _outW = PageW + 2 * _margin;
        _outH = PageH + 2 * _margin;
    }

    public void Dispose()
    {
        foreach ((string _, List<PageTemplate> refs) in _pages)
        {
            foreach (PageTemplate t in refs)
            {
                t.Dispose();
            }
        }
        _sift.Dispose();
        _matcher.Dispose();
    }

    // ------------------------------------------------------------------ matching

    private sealed record MatchResult(double[,]? H, int Inliers, Point[]? Pts, bool Ok);

    /// <summary>
    /// Template -> photo/page matches, MAGSAC homography. Port of <c>PageRegistrar._match</c>
    /// (page_registration.py:202-237). <paramref name="kp"/>/<paramref name="desc"/> are the SEARCH
    /// SPACE — photo keypoints for a coarse match, or a page-frame re-detection for a guided one.
    /// With <paramref name="prior"/> and <paramref name="radius"/>, only matches whose photo point
    /// lands within <paramref name="radius"/> px of the template point after the prior are kept.
    /// </summary>
    private MatchResult Match(PageTemplate tpl, Point[] kp, Mat? desc, double reproj,
        double? radius, double[,]? prior)
    {
        if (desc is null || desc.Rows < 8)
        {
            return new MatchResult(null, desc?.Rows ?? 0, null, false);
        }
        DMatch[][] knn = _matcher.KnnMatch(tpl.Descriptors, desc, 2);
        var srcPhoto = new List<Point>();
        var dstTpl = new List<Point>();
        for (int i = 0; i < knn.Length; i++)
        {
            if (knn[i].Length != 2)
            {
                continue;
            }
            if (knn[i][0].Distance < RatioTest * knn[i][1].Distance)
            {
                srcPhoto.Add(kp[knn[i][0].TrainIdx]);
                dstTpl.Add(tpl.KeyPoints[i]); // query index i == template row i
            }
        }
        if (srcPhoto.Count < 8)
        {
            return new MatchResult(null, srcPhoto.Count, null, false);
        }

        if (prior is not null && radius is double r)
        {
            Point[] pred = Homography.TransformPoints(prior, srcPhoto);
            var disp = new Point[pred.Length];
            for (int i = 0; i < pred.Length; i++)
            {
                disp[i] = new Point(pred[i].X - dstTpl[i].X, pred[i].Y - dstTpl[i].Y);
            }
            // A prior from the other page or a Borders quad is typically off by a translation
            // (spine gap, mask margin) larger than the window. Centre the window on the MEDIAN
            // displacement of the roughly-near matches first; the homography itself is still fitted
            // photo->page (page_registration.py:220-227).
            var roughX = new List<double>();
            var roughY = new List<double>();
            foreach (Point d in disp)
            {
                if (Math.Sqrt(d.X * d.X + d.Y * d.Y) < 2.5 * r)
                {
                    roughX.Add(d.X);
                    roughY.Add(d.Y);
                }
            }
            if (roughX.Count >= 6)
            {
                double mx = Median(roughX), my = Median(roughY);
                for (int i = 0; i < disp.Length; i++)
                {
                    disp[i] = new Point(disp[i].X - mx, disp[i].Y - my);
                }
            }
            var keptSrc = new List<Point>();
            var keptDst = new List<Point>();
            for (int i = 0; i < disp.Length; i++)
            {
                if (Math.Sqrt(disp[i].X * disp[i].X + disp[i].Y * disp[i].Y) < r)
                {
                    keptSrc.Add(srcPhoto[i]);
                    keptDst.Add(dstTpl[i]);
                }
            }
            srcPhoto = keptSrc;
            dstTpl = keptDst;
            if (srcPhoto.Count < 8)
            {
                return new MatchResult(null, srcPhoto.Count, null, false);
            }
        }

        (double[,]? h, bool[] inlierMask, bool ok) = Homography.FindMagsac(srcPhoto, dstTpl, reproj);
        if (!ok || h is null)
        {
            return new MatchResult(null, 0, null, false);
        }
        var inl = new List<Point>();
        for (int i = 0; i < inlierMask.Length; i++)
        {
            if (inlierMask[i])
            {
                inl.Add(dstTpl[i]);
            }
        }
        return new MatchResult(h, inl.Count, [.. inl], true);
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0)
        {
            return 0;
        }
        double[] s = [.. values];
        Array.Sort(s);
        int n = s.Length;
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }

    /// <summary>
    /// Restricts keypoints/descriptors to those inside <paramref name="quad"/> grown by
    /// <see cref="QuadDilateFrac"/>. Port of <c>_features_in_quad</c> (page_registration.py:239-250):
    /// each keypoint is ROUNDED then clamped to the image before the inside test, exactly as the
    /// reference does before indexing its mask array.
    /// </summary>
    private static (Point[] Kp, Mat? Desc) FeaturesInQuad(Point[] kp, Mat desc, Point[] quad, int w, int h)
    {
        double cx = quad.Average(p => p.X), cy = quad.Average(p => p.Y);
        Point[] grown = [.. quad.Select(p =>
            new Point(cx + (p.X - cx) * (1 + 2 * QuadDilateFrac), cy + (p.Y - cy) * (1 + 2 * QuadDilateFrac)))];

        var keepIdx = new List<int>();
        for (int i = 0; i < kp.Length; i++)
        {
            double px = Math.Clamp(Math.Round(kp[i].X, MidpointRounding.ToEven), 0, w - 1);
            double py = Math.Clamp(Math.Round(kp[i].Y, MidpointRounding.ToEven), 0, h - 1);
            if (PointInConvex(new Point(px, py), grown))
            {
                keepIdx.Add(i);
            }
        }
        if (keepIdx.Count == 0)
        {
            return ([], null);
        }
        var outKp = new Point[keepIdx.Count];
        using var rows = new Mat(keepIdx.Count, desc.Cols, desc.Type());
        for (int i = 0; i < keepIdx.Count; i++)
        {
            outKp[i] = kp[keepIdx[i]];
            desc.Row(keepIdx[i]).CopyTo(rows.Row(i));
        }
        return (outKp, rows.Clone());
    }

    private static bool PointInConvex(Point p, IReadOnlyList<Point> quad)
    {
        int n = quad.Count;
        double sign = 0;
        for (int i = 0; i < n; i++)
        {
            Point a = quad[i], b = quad[(i + 1) % n];
            double cross = (b.X - a.X) * (p.Y - a.Y) - (b.Y - a.Y) * (p.X - a.X);
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

    // ---------------------------------------------------------------- refinement

    private static bool SpreadOk(Point[]? pts, double minPx)
    {
        if (pts is null || pts.Length < 4)
        {
            return false;
        }
        double minX = pts.Min(p => p.X), maxX = pts.Max(p => p.X);
        double minY = pts.Min(p => p.Y), maxY = pts.Max(p => p.Y);
        return maxX - minX >= minPx && maxY - minY >= minPx;
    }

    /// <summary>
    /// Guided re-match in the canonical frame. Port of <c>PageRegistrar._refine</c>
    /// (page_registration.py:259-285), ECC branch dropped (<c>use_ecc</c> is always false here — see
    /// the class docstring). Features are detected ONCE on the page warped with the coarse H; the
    /// second pass re-matches those SAME features under the first pass's result with a tighter window
    /// — no second warp/detection, which is where the reference's own profiling put the time.
    /// </summary>
    private (double[,] H, int Inliers, Point[]? Pts, bool Ok) Refine(Mat gray, PageTemplate tpl,
        double[,] h, double[] radii, int minInliers)
    {
        using Mat page = new();
        using (Mat hm = Homography.ToMat(h))
        {
            Cv2.WarpPerspective(gray, page, hm, new Size(tpl.Width, tpl.Height),
                InterpolationFlags.Linear, BorderTypes.Replicate);
        }
        using var desc = new Mat();
        _sift.DetectAndCompute(page, null, out KeyPoint[] kps, desc);
        Point[] kp = [.. kps.Select(k => new Point(k.Pt.X, k.Pt.Y))];

        double[,] prior = Homography.Identity();
        int inliers = 0;
        Point[]? pts = null;
        for (int i = 0; i < 2; i++)
        {
            MatchResult m = Match(tpl, kp, desc, RefineReprojPx[i], radii[i], prior);
            if (!m.Ok || m.Inliers < minInliers || m.H is null)
            {
                return (h, 0, null, false);
            }
            prior = m.H;
            inliers = m.Inliers;
            pts = m.Pts;
        }
        return (Homography.Multiply(prior, h), inliers, pts, true);
    }

    // ---------------------------------------------------------------- validation

    /// <summary>Port of <c>PageRegistrar._sane_quad</c> (page_registration.py:304-320).</summary>
    private bool SaneQuad(Point[]? quad, int imgW, int imgH, double? refArea)
    {
        if (quad is null || quad.Any(p => double.IsNaN(p.X) || double.IsInfinity(p.X)
            || double.IsNaN(p.Y) || double.IsInfinity(p.Y)))
        {
            return false;
        }
        if (!QuadFit.IsConvex(quad))
        {
            return false;
        }
        double area = Contours.ContourArea(quad);
        double hw = (double)imgH * imgW;
        if (area < 0.02 * hw || area > 1.5 * hw)
        {
            return false;
        }
        Point[] q = Geometry.OrderPoints(quad) ?? quad;
        double wt = 0.5 * (Geometry.Distance(q[1], q[0]) + Geometry.Distance(q[2], q[3]));
        double ht = 0.5 * (Geometry.Distance(q[3], q[0]) + Geometry.Distance(q[2], q[1]));
        if (ht < 1e-3)
        {
            return false;
        }
        double ratio = (wt / ht) / ((double)PageW / PageH);
        if (!(ratio > 0.6 && ratio < 1.7))
        {
            return false;
        }
        if (refArea is double ra && ra > 0)
        {
            double f = area / ra;
            if (!(f > 0.4 && f < 2.5))
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>Refines a candidate and validates it. Port of <c>PageRegistrar._accept</c>
    /// (page_registration.py:322-334).</summary>
    private PageRegistrationResult? Accept(Mat gray, PageTemplate tpl, double[,] h, double[] radii,
        double? refArea, bool chain)
    {
        if (!SaneQuad(tpl.QuadInImage(h), gray.Cols, gray.Rows, refArea))
        {
            return null;
        }
        int minInl = chain ? MinRefinedInliersChain : MinRefinedInliers;
        double minSpread = chain ? MinSpreadPxChain : MinSpreadPx;
        (double[,] hr, int inl, Point[]? pts, bool ok) = Refine(gray, tpl, h, radii, minInl);
        if (!ok || inl < minInl || !SpreadOk(pts, minSpread))
        {
            return null;
        }
        Point[]? quad = tpl.QuadInImage(hr);
        if (!SaneQuad(quad, gray.Cols, gray.Rows, refArea))
        {
            return null;
        }
        return new PageRegistrationResult(tpl.Name, hr, inl, "x", quad);
    }

    // ------------------------------------------------------------------- driver

    /// <summary>
    /// Registers every template page in <paramref name="imgRgb"/> (upright RGB photo). Port of
    /// <c>PageRegistrar.register</c> (page_registration.py:337-436).
    /// </summary>
    /// <param name="quads">Optional Borders page quads (photo pixels).</param>
    public List<PageRegistrationResult> Register(Image imgRgb, IReadOnlyList<Point[]>? quads = null)
    {
        using Image gray = Io.ToGray(imgRgb);
        using var desc = new Mat();
        _sift.DetectAndCompute(gray.Mat, null, out KeyPoint[] kps, desc);
        Point[] kp = [.. kps.Select(k => new Point(k.Pt.X, k.Pt.Y))];

        Point[][] quadArr = [.. (quads ?? [])];
        var quadFeats = new (Point[] Kp, Mat? Desc)[quadArr.Length];
        for (int i = 0; i < quadArr.Length; i++)
        {
            quadFeats[i] = FeaturesInQuad(kp, desc, quadArr[i], gray.Width, gray.Height);
        }

        var results = new PageRegistrationResult?[_pages.Count];
        for (int i = 0; i < _pages.Count; i++)
        {
            results[i] = new PageRegistrationResult(_pages[i].Name, null, 0, "none", null);
        }
        var taken = new HashSet<int>();
        var pending = Enumerable.Range(0, _pages.Count).ToList();

        try
        {
            for (int round = 0; round < 2; round++)
            {
                if (round == 1 && !results.Any(r => r!.Ok))
                {
                    break;
                }
                foreach (int ti in pending.ToList())
                {
                    List<PageTemplate> refs = _pages[ti].Refs;
                    PageRegistrationResult? found = null;

                    var cands = new List<(int Inl, double[,] H, string Method, double? RefArea, PageTemplate Tpl)>();
                    if (round == 0)
                    {
                        foreach (PageTemplate tpl in refs)
                        {
                            for (int qi = 0; qi < quadFeats.Length; qi++)
                            {
                                if (taken.Contains(qi))
                                {
                                    continue;
                                }
                                MatchResult m = Match(tpl, quadFeats[qi].Kp, quadFeats[qi].Desc, CoarseReprojPx, null, null);
                                if (m.Ok && m.Inliers >= MinCoarseInliers && m.H is not null)
                                {
                                    cands.Add((m.Inliers, m.H, $"quad{qi}", Contours.ContourArea(quadArr[qi]), tpl));
                                }
                            }
                            int best = cands.Count > 0 ? cands.Max(c => c.Inl) : -1;
                            if (best < GoodCoarseInliers)
                            {
                                MatchResult m = Match(tpl, kp, desc, CoarseReprojPx, null, null);
                                if (m.Ok && m.Inliers >= MinCoarseInliers && m.H is not null)
                                {
                                    cands.Add((m.Inliers, m.H, "global", null, tpl));
                                }
                            }
                            best = cands.Count > 0 ? cands.Max(c => c.Inl) : -1;
                            if (cands.Count > 0 && best >= GoodCoarseInliers)
                            {
                                break; // strong enough; do not pay for the next reference
                            }
                        }
                        foreach (var c in cands.OrderByDescending(c => c.Inl).Take(MaxCoarseCandidates))
                        {
                            PageRegistrationResult? f = Accept(gray.Mat, c.Tpl, c.H, RefineRadiusPx, c.RefArea, false);
                            if (f is not null)
                            {
                                f = f with { Method = c.Method, Ref = refs.IndexOf(c.Tpl) };
                                found = f;
                                break;
                            }
                        }
                    }

                    if (found is null)
                    {
                        for (int tj = 0; tj < results.Length; tj++)
                        {
                            PageRegistrationResult? other = results[tj];
                            if (tj == ti || other is null || !other.Ok)
                            {
                                continue;
                            }
                            double dy = (ti - tj) * (double)PageH * (1 + ChainGapFrac);
                            double[,] shift = { { 1, 0, 0 }, { 0, 1, -dy }, { 0, 0, 1 } };
                            double[,] prior = Homography.Multiply(shift, other.H!);
                            foreach (PageTemplate tpl in refs)
                            {
                                PageRegistrationResult? f = Accept(gray.Mat, tpl, prior, ChainRadiusPx, null, true);
                                if (f is not null)
                                {
                                    found = f with { Method = $"chain:{other.Name}", Ref = refs.IndexOf(tpl) };
                                    break;
                                }
                            }
                            if (found is not null)
                            {
                                break;
                            }
                        }
                    }

                    if (found is null && round == 0)
                    {
                        for (int qi = 0; qi < quadArr.Length; qi++)
                        {
                            if (taken.Contains(qi))
                            {
                                continue;
                            }
                            Point[] ordered = Geometry.OrderPoints(quadArr[qi]) ?? quadArr[qi];
                            double[,]? prior = Homography.Solve4(ordered, refs[0].Corners);
                            if (prior is null)
                            {
                                continue;
                            }
                            foreach (PageTemplate tpl in refs)
                            {
                                PageRegistrationResult? f = Accept(gray.Mat, tpl, prior, ChainRadiusPx,
                                    Contours.ContourArea(quadArr[qi]), true);
                                if (f is not null)
                                {
                                    found = f with { Method = $"quadprior{qi}", Ref = refs.IndexOf(tpl) };
                                    break;
                                }
                            }
                            if (found is not null)
                            {
                                break;
                            }
                        }
                    }

                    if (found is null)
                    {
                        continue;
                    }
                    found = found with { Name = _pages[ti].Name };
                    results[ti] = found;
                    pending.Remove(ti);
                    if (found.Method.StartsWith("quadprior", StringComparison.Ordinal))
                    {
                        taken.Add(int.Parse(found.Method["quadprior".Length..]));
                    }
                    else if (found.Method.StartsWith("quad", StringComparison.Ordinal))
                    {
                        taken.Add(int.Parse(found.Method["quad".Length..]));
                    }
                }
                if (pending.Count == 0)
                {
                    break;
                }
            }
        }
        finally
        {
            foreach ((Point[] _, Mat? d) in quadFeats)
            {
                d?.Dispose();
            }
        }
        return [.. results.Select(r => r!)];
    }

    /// <summary>
    /// Output scale (&lt;= 1) at which no registered page is upsampled, floored at 0.25. 1.0 when
    /// nothing is registered. Port of <c>PageRegistrar.native_scale</c> (page_registration.py:438-455).
    /// </summary>
    public double NativeScale(IReadOnlyList<PageRegistrationResult> regs)
    {
        double scale = 1.0;
        foreach (PageRegistrationResult r in regs)
        {
            if (!r.Ok)
            {
                continue;
            }
            Point[] q = Geometry.OrderPoints(r.Quad!) ?? r.Quad!;
            double wNative = 0.5 * (Geometry.Distance(q[1], q[0]) + Geometry.Distance(q[2], q[3]));
            scale = Math.Min(scale, wNative / PageW);
        }
        return Math.Max(scale, 0.25);
    }

    public (int W, int H) OutSize(double scale = 1.0) =>
        (Math.Max(1, PyNum.RoundHalfEvenToInt(_outW * scale)), Math.Max(1, PyNum.RoundHalfEvenToInt(_outH * scale)));

    /// <summary>Warps a photo quad (e.g. from Borders) into the canonical page frame — same output
    /// size/cushion as <see cref="WarpPage"/>, so the two stack consistently. Port of
    /// <c>PageRegistrar.warp_quad</c> (page_registration.py:461-472).</summary>
    public Image WarpQuad(Image imgRgb, Point[] quad, double scale = 1.0)
    {
        double m = _margin;
        Point[] src = Geometry.OrderPoints(quad) ?? quad;
        Point[] dst =
        [
            new(m * scale, m * scale),
            new((m + PageW) * scale, m * scale),
            new((m + PageW) * scale, (m + PageH) * scale),
            new(m * scale, (m + PageH) * scale),
        ];
        double[,]? h = Homography.Solve4(src, dst);
        (int w, int hgt) = OutSize(scale);
        if (h is null)
        {
            using Mat identity = Mat.Eye(3, 3, MatType.CV_64FC1);
            var fallback = new Mat();
            Cv2.WarpPerspective(imgRgb.Mat, fallback, identity, new Size(w, hgt),
                InterpolationFlags.Linear, BorderTypes.Replicate);
            return Image.Wrap(fallback);
        }
        return Homography.WarpByHomography(imgRgb, h, w, hgt, replicate: true);
    }

    /// <summary>Warps the photo to the canonical page with the margin cushion on every side, at
    /// <paramref name="scale"/>. Port of <c>PageRegistrar.warp_page</c> (page_registration.py:518-526).</summary>
    public Image WarpPage(Image imgRgb, PageRegistrationResult reg, double scale = 1.0)
    {
        double m = _margin;
        double[,] shift = { { 1, 0, m }, { 0, 1, m }, { 0, 0, 1 } };
        double[,] s = { { scale, 0, 0 }, { 0, scale, 0 }, { 0, 0, 1 } };
        double[,] full = Homography.Multiply(s, Homography.Multiply(shift, reg.H!));
        (int w, int h) = OutSize(scale);
        return Homography.WarpByHomography(imgRgb, full, w, h, replicate: true);
    }

    /// <summary>Page quads (ordered TL, TR, BR, BL) from Borders contours, plus the per-quad fit
    /// info. Port of <c>PageRegistrar.page_quads</c> (page_registration.py:474-488).</summary>
    public (List<Point[]> Quads, List<QuadFit.Info> Infos) PageQuads(
        IReadOnlyList<Point[]>? segments, int imgH, int imgW)
    {
        var quadsOut = new List<Point[]>();
        var infos = new List<QuadFit.Info>();
        foreach (Point[] s in segments ?? [])
        {
            Point[]? q;
            QuadFit.Info info;
            if (LineQuads)
            {
                (q, info) = QuadFit.FitQuadLines(s, imgH, imgW, (double)PageH / PageW, extrapolate: true);
            }
            else
            {
                q = Geometry.ExtractQuad(s);
                info = new QuadFit.Info { Method = "polygon" };
            }
            if (q is not null)
            {
                quadsOut.Add(Geometry.OrderPoints(q) ?? q);
                infos.Add(info);
            }
        }
        return (quadsOut, infos);
    }

    /// <summary>
    /// Intersection over union of two convex photo quads. Port of <c>PageRegistrar.quad_iou</c>
    /// (page_registration.py:509-516) — a static method there too.
    ///
    /// <para>
    /// <b>Both inputs are reordered (TL, TR, BR, BL) before the intersection.</b> The reference does
    /// this explicitly (<c>_order_points</c> on both <c>a</c> and <c>b</c>) because
    /// <see cref="QuadFit.QuadIou"/>'s Sutherland-Hodgman clip assumes consistent winding on both
    /// polygons — a quad from <see cref="PageTemplate.QuadInImage"/> (corners carried through an
    /// inverse homography) is not guaranteed to come out in the same order a Borders quad already
    /// is. Skipping this reorder is exactly the bug this comment is warning against: found by
    /// measurement (a registered page's best-matching Borders quad resolved to the WRONG page,
    /// because the unordered IoU against the correct quad came out smaller than against the wrong
    /// one) — not by code review.
    /// </para>
    /// </summary>
    public static double QuadIou(Point[] a, Point[] b) =>
        QuadFit.QuadIou(Geometry.OrderPoints(a) ?? a, Geometry.OrderPoints(b) ?? b);

    /// <summary>
    /// Straightens a warped page by its own lines: first the homography (<see cref="LineRefine"/>),
    /// then the bend map (<see cref="LineDewarp"/>) on the result. Port of
    /// <c>PageRegistrar.straighten</c> (page_registration.py:490-507). CONSUMES
    /// <paramref name="page"/> (disposes it if a correction replaces it, returns it unchanged
    /// otherwise) — the caller must not touch <paramref name="page"/> again, only the returned
    /// <see cref="Image"/>.
    /// </summary>
    /// <remarks>
    /// The type names <see cref="global::RussianDocs.DocumentProcessing.PageRegistration.LineRefine"/>
    /// and <see cref="global::RussianDocs.DocumentProcessing.PageRegistration.LineDewarp"/> are
    /// referenced fully qualified throughout this method: this class also has boolean properties
    /// named <see cref="LineRefine"/>/<see cref="LineDewarp"/> (matching the reference's own
    /// <c>line_refine</c>/<c>line_dewarp</c> constructor flags), and an unqualified reference inside
    /// an instance member resolves to the property, not the type.
    /// </remarks>
    public (Image Page, global::RussianDocs.DocumentProcessing.PageRegistration.LineRefine.StraightenInfo Info)
        Straighten(Image page, double scale = 1.0)
    {
        int inset = PyNum.RoundHalfEvenToInt(_margin * scale);
        var info = new global::RussianDocs.DocumentProcessing.PageRegistration.LineRefine.StraightenInfo(
            false, "disabled");
        Image cur = page;

        if (LineRefine)
        {
            using Image gray = Io.ToGray(cur);
            (double[,]? hm, var refInfo) =
                global::RussianDocs.DocumentProcessing.PageRegistration.LineRefine.RefineByLines(gray.Mat, inset);
            info = refInfo;
            if (hm is not null)
            {
                Image next =
                    global::RussianDocs.DocumentProcessing.PageRegistration.LineRefine.ApplyRefinement(cur, hm);
                cur.Dispose();
                cur = next;
            }
        }
        if (LineDewarp)
        {
            using Image gray = Io.ToGray(cur);
            (float[]? v, var dInfo) =
                global::RussianDocs.DocumentProcessing.PageRegistration.LineDewarp.DewarpByLines(gray.Mat, inset);
            if (v is not null)
            {
                Image next = global::RussianDocs.DocumentProcessing.PageRegistration.LineDewarp.ApplyDewarp(cur, v);
                cur.Dispose();
                cur = next;
            }
            info = info with { Applied = info.Applied || dInfo.Applied };
        }
        return (cur, info);
    }

    private sealed class TemplateMeta
    {
        [JsonPropertyName("doc_type")] public string DocType { get; set; } = "";
        [JsonPropertyName("pages")] public TemplatePage[] Pages { get; set; } = [];
    }

    private sealed class TemplatePage
    {
        [JsonPropertyName("name")] public string Name { get; set; } = "";
        [JsonPropertyName("image")] public string? Image { get; set; }
        [JsonPropertyName("mask")] public string? Mask { get; set; }
        [JsonPropertyName("refs")] public TemplateRef[] Refs { get; set; } = [];
    }

    private sealed class TemplateRef
    {
        [JsonPropertyName("image")] public string Image { get; set; } = "";
        [JsonPropertyName("mask")] public string Mask { get; set; } = "";
    }
}

/// <summary>One reference of a canonical page: erased print, static mask, SIFT features. Port of
/// <c>PageTemplate</c> (page_registration.py:106-129).</summary>
public sealed class PageTemplate : IDisposable
{
    public string Name { get; }
    public int Width { get; }
    public int Height { get; }
    public Point[] KeyPoints { get; }

    /// <summary>SIFT descriptors, one row per <see cref="KeyPoints"/> entry, CV_32FC1 x128. Owned.</summary>
    public Mat Descriptors { get; }

    /// <summary>{0,0}, {w,0}, {w,h}, {0,h} — the page's own corners.</summary>
    public Point[] Corners { get; }

    private Mat? _gray;

    private PageTemplate(string name, Mat gray, Point[] kp, Mat desc)
    {
        Name = name;
        _gray = gray;
        Width = gray.Cols;
        Height = gray.Rows;
        KeyPoints = kp;
        Descriptors = desc;
        Corners = [new Point(0, 0), new Point(Width, 0), new Point(Width, Height), new Point(0, Height)];
    }

    public static PageTemplate Load(string name, string imagePath, string maskPath, SIFT sift)
    {
        using Image bgr = Io.LoadRgb(imagePath);
        Mat gray = Io.ToGray(bgr).Take();

        using Mat maskGray = Cv2.ImRead(maskPath, ImreadModes.Grayscale);
        if (maskGray.Empty())
        {
            gray.Dispose();
            throw new FileNotFoundException("page_registration: template mask not found", maskPath);
        }
        using var mask = new Mat();
        Cv2.Threshold(maskGray, mask, 127, 255, ThresholdTypes.Binary);

        using var desc = new Mat();
        sift.DetectAndCompute(gray, mask, out KeyPoint[] kps, desc);
        Point[] kp = [.. kps.Select(k => new Point(k.Pt.X, k.Pt.Y))];
        return new PageTemplate(name, gray, kp, desc.Clone());
    }

    /// <summary>Page corners mapped back into the photo. Port of <c>PageTemplate.quad_in_image</c>
    /// (page_registration.py:126-129). Null when <paramref name="hImgToPage"/> is singular.</summary>
    public Point[]? QuadInImage(double[,] hImgToPage)
    {
        double[,]? inv = Homography.Invert(hImgToPage);
        return inv is null ? null : Homography.TransformPoints(inv, Corners);
    }

    public void Dispose()
    {
        _gray?.Dispose();
        _gray = null;
        Descriptors.Dispose();
    }
}

/// <summary>Result for one page. <see cref="H"/> maps photo pixels to canonical page pixels. Port of
/// <c>PageRegistration</c> (page_registration.py:132-152) — named <c>PageRegistrationResult</c> here
/// to avoid colliding with this file's own namespace.</summary>
public sealed record PageRegistrationResult(
    string Name, double[,]? H, int Inliers, string Method, Point[]? Quad, int Ref = 0)
{
    public bool Ok => H is not null;
}
