using OpenCvSharp;
using RussianDocs.DocumentProcessing.Imaging;
using Point = RussianDocs.DocumentProcessing.Imaging.Point;

namespace RussianDocs.DocumentProcessing.PageRegistration;

/// <summary>
/// Blur-tolerant refinement of a rectified page by its own straight structures. Port of
/// <c>pipeline_modules/page_registration/line_refine.py</c>, following the structure of the Go
/// port's already-graded <c>internal/docproc/modules/linerefine.go</c> (44/44 stages, both profiles).
///
/// <para>
/// After a page is warped into its frame, every horizontal structure of a flat page (field rules,
/// text baselines, the MRZ, the red band) must be exactly horizontal, and every vertical one (photo
/// frame, series/number column, page edges) exactly vertical. Evidence comes from two sources so it
/// survives blur, which kills keypoint matching first: LSD line segments and projection-profile tilts
/// of horizontal bands (text lines make a sharp profile even when badly blurred). A 4-parameter model
/// (rotation, shear, two perspective terms) is fitted with a robust (Cauchy-weighted IRLS) loss;
/// parameters without evidence are pinned toward zero by a weak prior. The correction is applied only
/// if it is small and actually reduces the residual.
/// </para>
///
/// <para>
/// <b>Two substitutions from the reference, both forced by the binding rather than chosen for
/// convenience — matching the Go port's own documented gaps, verified independently here:</b>
/// </para>
/// <list type="bullet">
/// <item>
/// <b>Canny + probabilistic Hough (<c>HoughLinesP</c>) stands in for <c>cv2.createLineSegmentDetector</c>
/// (LSD).</b> Checked directly: OpenCvSharp4 4.13.0.20260627 has no <c>CreateLineSegmentDetector</c> on
/// <c>Cv2</c> at all (build error, not a missing overload) — the same ecosystem gap Go hit with gocv
/// v0.43.0. Both are standard OpenCV primitives finding the SAME KIND of evidence (straight
/// edges/strokes) that the rest of this module treats as an undifferentiated pool of (angle, weight)
/// measurements, but they are not bit-exact substitutes for each other — a different detector finds a
/// different segment SET on the same page. See the task log for the measured effect on the two
/// INTPASSPORT conformance cases.
/// </item>
/// <item>
/// The 4-parameter fit uses <see cref="LevenbergMarquardt"/> (this port's own, finite-difference
/// Jacobian) rather than <c>scipy.optimize.least_squares</c>'s default Trust-Region-Reflective — see
/// that type's own docstring for why.
/// </item>
/// </list>
/// </summary>
public static class LineRefine
{
    private const double LsdMinLenFrac = 0.04;
    private const double LsdMaxWeight = 3.0;
    private const double AngleTolDeg = 12.0;
    private const double ProfileScale = 0.5;
    private const double ProfileFineStep = 0.25;
    private const double ProfileMinPeak = 1.15;
    private const double ProfileWeight = 3.0;
    private const double MinHorizWeight = 4.0;
    private const double MaxRotDeg = 5.0;
    private const double MaxCornerShiftFrac = 0.08;
    private const double MinGain = 0.25;
    private const double MinResidualDeg = 0.3;
    private const int IrlsIters = 3;
    private const double IrlsScaleDeg = 1.0;
    private const double VertMinLenFrac = 0.15;
    private const double BlobMaxFrac = 0.2;
    private const int NBands = 5;

    private static readonly double[] PriorScale = [0.03, 0.05, 0.03];
    private static readonly double[] ProfileCoarse = BuildRange(-6.0, 6.0, 1.0);

    private static double[] BuildRange(double lo, double hi, double step)
    {
        var list = new List<double>();
        for (double a = lo; a <= hi + 1e-9; a += step)
        {
            list.Add(a);
        }
        return [.. list];
    }

    /// <summary>Diagnostic only — not part of the conformance contract (no stage reads it).</summary>
    public sealed record StraightenInfo(bool Applied, string? Reason);

    /// <summary>One row of <c>measure()</c>'s (x1,y1,x2,y2,kind,weight,source). kind 0 = should be
    /// horizontal, 1 = should be vertical.</summary>
    private readonly record struct MeasureRow(double X1, double Y1, double X2, double Y2, int Kind, double Weight);

    private static double PyMod(double a, double b)
    {
        double m = a % b;
        return m < 0 ? m + b : m;
    }

    internal static double SegAngle(double x1, double y1, double x2, double y2) =>
        PyMod(Math.Atan2(y2 - y1, x2 - x1) * 180.0 / Math.PI + 90.0, 180.0) - 90.0;

    /// <summary>Canny + HoughLinesP — see the class docstring for why this stands in for LSD.</summary>
    internal static (double X1, double Y1, double X2, double Y2)[] DetectSegments(Mat gray, double minLen)
    {
        using var edges = new Mat();
        Cv2.Canny(gray, edges, 40, 120);
        LineSegmentPoint[] lines = Cv2.HoughLinesP(edges, 1, Math.PI / 180.0, 30, (float)minLen, 6);
        var outSegs = new List<(double, double, double, double)>(lines.Length);
        foreach (LineSegmentPoint l in lines)
        {
            double dx = l.P2.X - l.P1.X, dy = l.P2.Y - l.P1.Y;
            if (dx * dx + dy * dy >= minLen * minLen)
            {
                outSegs.Add((l.P1.X, l.P1.Y, l.P2.X, l.P2.Y));
            }
        }
        return [.. outSegs];
    }

    /// <summary>
    /// Otsu-inverted binarisation with tall connected components dropped (ink components taller than
    /// <paramref name="blobMaxHeightPx"/> are not text — a photo, a stamp, a thumb). Axis is always
    /// "row" (horizontal bands) — the only one <c>measure()</c> ever calls (Python's vertical-strip
    /// path was tried and dropped, per that module's own comment).
    /// </summary>
    private static (byte[] Ink, int W, int H) OtsuInvDropTallBlobs(Mat gray, double blobMaxHeightPx)
    {
        using var binary = new Mat();
        Cv2.Threshold(gray, binary, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
        int w = binary.Cols, h = binary.Rows;

        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        int n = Cv2.ConnectedComponentsWithStats(binary, labels, stats, centroids, PixelConnectivity.Connectivity8);

        var ink = new byte[h * w];
        var indexer = binary.GetGenericIndexer<byte>();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                ink[y * w + x] = indexer[y, x];
            }
        }
        if (n > 1)
        {
            var big = new bool[n];
            for (int i = 1; i < n; i++) // index 0 is background, never dropped
            {
                if (stats.At<int>(i, 3) > blobMaxHeightPx) // CC_STAT_HEIGHT
                {
                    big[i] = true;
                }
            }
            if (big.Any(b => b))
            {
                var labelIdx = labels.GetGenericIndexer<int>();
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int lab = labelIdx[y, x];
                        if (lab >= 0 && lab < n && big[lab])
                        {
                            ink[y * w + x] = 0;
                        }
                    }
                }
            }
        }
        return (ink, w, h);
    }

    /// <summary>
    /// Tilt (deg) that makes the row-sum projection profile sharpest, and the peak ratio (best score
    /// / median score; 0 means no evidence). Port of <c>_profile_tilt</c> (axis=1 only).
    /// </summary>
    /// <summary>Shared with <see cref="LineDewarp"/>'s per-cell tilt measurement (that module calls
    /// this once per grid cell — see its own docstring for why that is a deliberate simplification
    /// of the reference's shared-rotation optimisation, not an oversight).</summary>
    internal static (double Tilt, double Ratio) ProfileTilt(Mat region)
    {
        int rw = Math.Max(1, (int)(region.Cols * ProfileScale));
        int rh = Math.Max(1, (int)(region.Rows * ProfileScale));
        using var small = new Mat();
        Cv2.Resize(region, small, new Size(rw, rh), 0, 0, InterpolationFlags.Area);

        (byte[] ink, int w, int h) = OtsuInvDropTallBlobs(small, BlobMaxFrac * rh);
        if (w != rw || h != rh)
        {
            return (0, 0);
        }

        Point2f centre = new(w / 2.0f, h / 2.0f);

        double[] Score(double[] angles)
        {
            var scores = new double[angles.Length];
            using var inkMat = Mat.FromPixelData(h, w, MatType.CV_8UC1, ink);
            using var validMat = new Mat(h, w, MatType.CV_8UC1, Scalar.All(255));
            for (int i = 0; i < angles.Length; i++)
            {
                using Mat m = Cv2.GetRotationMatrix2D(centre, angles[i], 1.0);
                using var rot = new Mat();
                using var cnt = new Mat();
                Cv2.WarpAffine(inkMat, rot, m, new Size(w, h), InterpolationFlags.Nearest,
                    BorderTypes.Constant, Scalar.All(0));
                Cv2.WarpAffine(validMat, cnt, m, new Size(w, h), InterpolationFlags.Nearest,
                    BorderTypes.Constant, Scalar.All(0));

                var inkSum = new long[h];
                var cntSum = new long[h];
                var rotIdx = rot.GetGenericIndexer<byte>();
                var cntIdx = cnt.GetGenericIndexer<byte>();
                for (int y = 0; y < h; y++)
                {
                    long si = 0, sc = 0;
                    for (int x = 0; x < w; x++)
                    {
                        si += rotIdx[y, x];
                        sc += cntIdx[y, x];
                    }
                    inkSum[y] = si;
                    cntSum[y] = sc;
                }
                long cmax = cntSum.Length > 0 ? cntSum.Max() : 0;
                var prof = new List<double>();
                for (int y = 0; y < h; y++)
                {
                    if (cntSum[y] >= 0.6 * cmax)
                    {
                        prof.Add(inkSum[y] / (double)Math.Max(cntSum[y], 1));
                    }
                }
                scores[i] = prof.Count > 2 ? Variance(prof) : 0.0;
            }
            return scores;
        }

        double[] coarse = Score(ProfileCoarse);
        int ib = ArgMax(coarse);
        if (ib == 0 || ib == ProfileCoarse.Length - 1 || coarse[ib] <= 0)
        {
            return (0, 0);
        }
        double[] fine = BuildRange(ProfileCoarse[ib] - 1.0, ProfileCoarse[ib] + 1.0, ProfileFineStep);
        double[] fs = Score(fine);
        int jb = ArgMax(fs);
        double best = fine[jb];
        if (jb > 0 && jb < fine.Length - 1)
        {
            double y0 = fs[jb - 1], y1 = fs[jb], y2 = fs[jb + 1];
            double den = y0 - 2 * y1 + y2;
            if (den < 0)
            {
                best += ProfileFineStep * 0.5 * (y0 - y2) / den;
            }
        }
        double med = Median([.. coarse]);
        double ratio = med > 0 ? fs[jb] / med : 0.0;
        return (best, ratio);
    }

    private static double Variance(List<double> v)
    {
        double mean = v.Average();
        return v.Sum(x => (x - mean) * (x - mean)) / v.Count;
    }

    private static int ArgMax(double[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best])
            {
                best = i;
            }
        }
        return best;
    }

    private static double Median(double[] v)
    {
        if (v.Length == 0)
        {
            return 0;
        }
        double[] s = [.. v];
        Array.Sort(s);
        int n = s.Length;
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }

    internal static double[] Linspace(double a, double b, int n)
    {
        if (n == 1)
        {
            return [a];
        }
        var outArr = new double[n];
        for (int i = 0; i < n; i++)
        {
            outArr[i] = a + (b - a) * i / (n - 1);
        }
        return outArr;
    }

    /// <summary>Straightness evidence of a rectified page. Port of <c>measure()</c>
    /// (line_refine.py:138-174).</summary>
    private static MeasureRow[] Measure(Mat gray, int inset)
    {
        int h = gray.Rows, w = gray.Cols;
        var rows = new List<MeasureRow>();

        foreach ((double x1, double y1, double x2, double y2) in DetectSegments(gray, LsdMinLenFrac * w))
        {
            double l = Math.Sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            double a = SegAngle(x1, y1, x2, y2);
            double weight = Math.Min(l / (0.1 * w), LsdMaxWeight);
            if (Math.Abs(a) < AngleTolDeg)
            {
                rows.Add(new MeasureRow(x1, y1, x2, y2, 0, weight));
            }
            else if (Math.Abs(Math.Abs(a) - 90.0) < AngleTolDeg && l >= VertMinLenFrac * h)
            {
                rows.Add(new MeasureRow(x1, y1, x2, y2, 1, weight));
            }
        }

        double x0d = inset, y0d = inset, x1d = w - inset, y1d = h - inset;
        if (x1d - x0d < 60 || y1d - y0d < 60)
        {
            return [.. rows];
        }
        double bh = (y1d - y0d) / 3.0;
        foreach (double yb in Linspace(y0d, y1d - bh, NBands))
        {
            int top = (int)yb, bottom = (int)(yb + bh);
            int left = (int)x0d, right = (int)x1d;
            if (bottom <= top || right <= left)
            {
                continue;
            }
            using Mat band = new(gray, new Rect(left, top, right - left, bottom - top));
            (double tilt, double ratio) = ProfileTilt(band);
            if (ratio >= ProfileMinPeak)
            {
                double yc = yb + 0.5 * bh;
                double half = 0.5 * (x1d - x0d);
                double dy = Math.Tan(tilt * Math.PI / 180.0) * half;
                double weight = ProfileWeight * Math.Min(1.0, ratio - 1.0);
                rows.Add(new MeasureRow(x0d, yc - dy, x1d, yc + dy, 0, weight));
            }
        }
        return [.. rows];
    }

    /// <summary>Pixel homography for (rotation rad, shear, p1, p2) about the page centre, perspective
    /// terms in units of the half page width. Port of <c>_model</c> (line_refine.py:177-186).</summary>
    private static double[,] Model(double[] p, double w, double h)
    {
        double th = p[0], s = p[1], p1 = p[2], p2 = p[3];
        double k = 2.0 / w;
        double[,] t = { { k, 0, -1.0 }, { 0, k, -h / w }, { 0, 0, 1.0 } };
        double[,] r = { { Math.Cos(th), -Math.Sin(th), 0 }, { Math.Sin(th), Math.Cos(th), 0 }, { 0, 0, 1.0 } };
        double[,] sh = { { 1.0, s, 0 }, { 0, 1.0, 0 }, { 0, 0, 1.0 } };
        double[,] pm = { { 1.0, 0, 0 }, { 0, 1.0, 0 }, { p1, p2, 1.0 } };
        double[,]? invT = Homography.Invert(t);
        if (invT is null)
        {
            return Homography.Identity();
        }
        return Homography.Multiply(invT, Homography.Multiply(pm, Homography.Multiply(r, Homography.Multiply(sh, t))));
    }

    /// <summary>Port of <c>_angles_after</c> (line_refine.py:189-195).</summary>
    private static double[] AnglesAfter(double[,] hm, MeasureRow[] meas)
    {
        var outArr = new double[meas.Length];
        for (int i = 0; i < meas.Length; i++)
        {
            Point p0 = Homography.TransformPoint(hm, new Point(meas[i].X1, meas[i].Y1));
            Point p1 = Homography.TransformPoint(hm, new Point(meas[i].X2, meas[i].Y2));
            double ang = PyMod(Math.Atan2(p1.Y - p0.Y, p1.X - p0.X) * 180.0 / Math.PI + 90.0, 180.0) - 90.0;
            outArr[i] = meas[i].Kind == 0 ? ang : Math.Abs(ang) - 90.0;
        }
        return outArr;
    }

    private static double[] Residuals(double[] p, MeasureRow[] meas, double w, double h, double[] wgt)
    {
        double[] dev = AnglesAfter(Model(p, w, h), meas);
        var outArr = new double[meas.Length + 3];
        for (int i = 0; i < meas.Length; i++)
        {
            outArr[i] = dev[i] * Math.Sqrt(wgt[i]);
        }
        outArr[meas.Length] = p[1] / PriorScale[0];
        outArr[meas.Length + 1] = p[2] / PriorScale[1];
        outArr[meas.Length + 2] = p[3] / PriorScale[2];
        return outArr;
    }

    /// <summary>Robust fit of the 4 parameters: IRLS with a Cauchy weight on every measurement while
    /// the prior stays quadratic. Port of <c>_fit</c> (line_refine.py:204-217).</summary>
    private static double[] FitParams(MeasureRow[] meas, double w, double h)
    {
        double[] x = [0, 0, 0, 0];
        double[] wgt = [.. meas.Select(m => m.Weight)];
        for (int iter = 0; iter < IrlsIters; iter++)
        {
            double[] wgtSnapshot = [.. wgt];
            LevenbergMarquardt.Result sol = LevenbergMarquardt.Solve(
                p => Residuals(p, meas, w, h, wgtSnapshot), x, maxIterations: 100);
            x = sol.X;
            double[] r = AnglesAfter(Model(x, w, h), meas);
            for (int i = 0; i < meas.Length; i++)
            {
                wgt[i] = meas[i].Weight / (1.0 + (r[i] / IrlsScaleDeg) * (r[i] / IrlsScaleDeg));
            }
        }
        return x;
    }

    /// <summary>
    /// The homography (pixel, src->dst for <c>WarpPerspective</c>) that straightens a rectified page
    /// by its own lines, or null when there is not enough evidence or the correction is not
    /// warranted. Port of <c>refine_by_lines</c> (line_refine.py:229-274).
    /// </summary>
    public static (double[,]? H, StraightenInfo Info) RefineByLines(Mat gray, int inset)
    {
        double h = gray.Rows, w = gray.Cols;
        MeasureRow[] meas = Measure(gray, inset);
        if (meas.Length == 0)
        {
            return (null, new StraightenInfo(false, "no evidence"));
        }
        double horizW = meas.Where(m => m.Kind == 0).Sum(m => m.Weight);
        if (horizW < MinHorizWeight)
        {
            return (null, new StraightenInfo(false, "too little horizontal evidence"));
        }
        double[] weights = [.. meas.Select(m => m.Weight)];
        double before = WMedian([.. AnglesAfter(Homography.Identity(), meas).Select(Math.Abs)], weights);

        double[] x = FitParams(meas, w, h);
        double[,] hm = Model(x, w, h);
        double after = WMedian([.. AnglesAfter(hm, meas).Select(Math.Abs)], weights);

        if (before < MinResidualDeg)
        {
            return (null, new StraightenInfo(false, "already straight"));
        }
        if (after > (1.0 - MinGain) * before)
        {
            return (null, new StraightenInfo(false, "no gain"));
        }
        double rotDeg = x[0] * 180.0 / Math.PI;
        if (Math.Abs(rotDeg) > MaxRotDeg)
        {
            return (null, new StraightenInfo(false, "rotation too large"));
        }
        Point[] corners =
        [
            new(inset, inset), new(w - inset, inset), new(w - inset, h - inset), new(inset, h - inset),
        ];
        Point[] moved = Homography.TransformPoints(hm, corners);
        double shift = 0;
        for (int i = 0; i < corners.Length; i++)
        {
            shift = Math.Max(shift, Geometry.Distance(corners[i], moved[i]));
        }
        if (shift > MaxCornerShiftFrac * w)
        {
            return (null, new StraightenInfo(false, "correction too large"));
        }
        return (hm, new StraightenInfo(true, null));
    }

    internal static double WMedian(double[] values, double[] weights)
    {
        if (values.Length == 0)
        {
            return 0.0;
        }
        int n = values.Length;
        int[] order = [.. Enumerable.Range(0, n).OrderBy(i => values[i])];
        double running = 0;
        var cum = new double[n];
        for (int i = 0; i < n; i++)
        {
            running += weights[order[i]];
            cum[i] = running;
        }
        double target = 0.5 * cum[^1];
        int idx = 0;
        while (idx < n - 1 && cum[idx] < target)
        {
            idx++;
        }
        return values[order[idx]];
    }

    /// <summary>Warps a page through the straightening homography. Port of <c>apply_refinement</c>
    /// (line_refine.py:277-279).</summary>
    public static Image ApplyRefinement(Image page, double[,] hm) =>
        Homography.WarpByHomography(page, hm, page.Width, page.Height, replicate: true);
}
