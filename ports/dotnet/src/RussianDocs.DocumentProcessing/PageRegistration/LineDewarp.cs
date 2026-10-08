using OpenCvSharp;
using RussianDocs.DocumentProcessing.Imaging;

namespace RussianDocs.DocumentProcessing.PageRegistration;

/// <summary>
/// Bend correction of a rectified page from the local tilt of its text lines. Port of
/// <c>pipeline_modules/page_registration/line_dewarp.py</c>, following the structure of the Go
/// port's already-graded <c>internal/docproc/modules/linedewarp.go</c> (44/44 stages, both profiles).
///
/// <para>
/// A homography (see <see cref="LineRefine"/>) makes a FLAT page straight. A booklet page bent near
/// the spine is not flat: its text lines curve, and their local tilt changes along the line (with
/// x) — a homography cannot express that. This module measures the local tilt <c>t(x, y)</c> of the
/// horizontal structures on a grid of overlapping cells (projection profiles, blur-tolerant) plus
/// LSD segments, fits a low-order polynomial <c>t(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y</c>
/// (x, y normalised), and integrates it along x into a vertical displacement map — zero on the
/// page's centre column, growing outward. The page is then remapped by <c>y' = y + v(x, y)</c>.
/// </para>
///
/// <para>
/// <b>Cells are measured the reference way</b>: the whole region rotated once per angle, every cell row
/// sums cut from a shared cumulative sum (<see cref="CellTilts"/>).
/// </para>
/// </summary>
public static class LineDewarp
{
    private const int Grid = 5;
    private const int MinCells = 8;
    private const double MinDispPx = 3.0;
    private const double MaxDispFrac = 0.04;
    private const double MinGain = 0.25;
    private const int IrlsIters = 3;
    private const double IrlsScaleDeg = 1.0;
    private const double Ridge = 1e-3;

    // line_refine.py's own constants this module imports directly (line_dewarp.py:29-31).
    private const double LsdMinLenFrac = 0.04;
    private const double LsdMaxWeight = 3.0;
    private const double AngleTolDeg = 12.0;
    private const double ProfileMinPeak = 1.15;
    private const double ProfileWeight = 3.0;

    /// <summary>One row of <c>measure_cells()</c>'s (x, y, tilt_deg, weight, source). source 0 = LSD
    /// horizontal segment (at its centre), 1 = profile cell (at its centre).</summary>
    private readonly record struct CellRow(double X, double Y, double Tilt, double Weight, int Source);

    /// <summary>Local tilt evidence. Port of <c>measure_cells</c> (line_dewarp.py:43-68).</summary>
    private static CellRow[] MeasureCells(Mat gray, int inset)
    {
        double h = gray.Rows, w = gray.Cols;
        var rows = new List<CellRow>();

        // LSD at half scale: segment angles and positions scale, and the detector is the slowest
        // step of the page.
        using var half = new Mat();
        Cv2.Resize(gray, half, new Size(0, 0), 0.5, 0.5, InterpolationFlags.Area);
        foreach ((double x1h, double y1h, double x2h, double y2h) in
            LineRefine.DetectSegments(half, LsdMinLenFrac * w * 0.5))
        {
            double x1 = x1h * 2, y1 = y1h * 2, x2 = x2h * 2, y2 = y2h * 2;
            double l = Math.Sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            double a = LineRefine.SegAngle(x1, y1, x2, y2);
            double weight = Math.Min(l / (0.1 * w), LsdMaxWeight);
            if (Math.Abs(a) < AngleTolDeg)
            {
                rows.Add(new CellRow(0.5 * (x1 + x2), 0.5 * (y1 + y2), a, weight, 0));
            }
        }

        double x0 = inset, y0 = inset, x1b = w - inset, y1b = h - inset;
        if (x1b - x0 < 90 || y1b - y0 < 90)
        {
            return [.. rows];
        }
        double cw = (x1b - x0) / 3.0, ch = (y1b - y0) / 3.0;
        var cellOrigins = new List<(double Xb, double Yb)>();
        foreach (double yb in LineRefine.Linspace(y0, y1b - ch, Grid))
        {
            foreach (double xb in LineRefine.Linspace(x0, x1b - cw, Grid))
            {
                cellOrigins.Add((xb, yb));
            }
        }
        int ix0 = (int)x0, iy0 = (int)y0, ix1 = (int)x1b, iy1 = (int)y1b;
        using Mat region = new(gray, new Rect(ix0, iy0, ix1 - ix0, iy1 - iy0));
        (double Tilt, double Ratio)[] tilts = CellTilts(region,
            [.. cellOrigins.Select(o => (o.Xb - x0, o.Yb - y0, cw, ch))]);
        for (int k = 0; k < cellOrigins.Count; k++)
        {
            (double tilt, double ratio) = tilts[k];
            if (ratio >= ProfileMinPeak)
            {
                double weight = ProfileWeight * Math.Min(1.0, ratio - 1.0);
                rows.Add(new CellRow(cellOrigins[k].Xb + 0.5 * cw, cellOrigins[k].Yb + 0.5 * ch, tilt, weight, 1));
            }
        }
        return [.. rows];
    }

    private const double ProfileScale = 0.5;
    private const double BlobMaxFrac = 0.2;
    private static readonly double[] ProfileCoarse = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6];
    private static readonly double[] FineOffsets = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
    private const double ProfileFineStep = 0.25;

    /// <summary>
    /// Tilt and peak ratio of the row profile in every cell (x, y, w, h of <paramref name="region"/>).
    /// Port of <c>_cell_tilts</c> (line_dewarp.py:71-146): like <see cref="LineRefine.ProfileTilt"/> but
    /// the WHOLE region is rotated once per angle and every cell's row sums are cut out of a cumulative
    /// sum along x. Rotation about the region centre instead of the cell centre only shifts a cell's
    /// content by a few pixels at these angles, which the profile does not mind - but it IS a different
    /// picture from rotating each cell on its own, and the number of cells with a peak decides whether
    /// the page is dewarped at all (MIN_CELLS), so the shortcut of the first port is not taken. The blob
    /// filter keeps the reference's use of the FIRST cell's height for every cell.
    ///
    /// <para>Data is float32 as in the reference (the ink image, the cumulative sums, the profile); the
    /// variance is accumulated in double, which differs from NumPy's float32 pairwise sum in the last
    /// bits only.</para>
    /// </summary>
    private static (double Tilt, double Ratio)[] CellTilts(Mat region,
        (double X, double Y, double W, double H)[] cells)
    {
        using var small = new Mat();
        Cv2.Resize(region, small, new Size(0, 0), ProfileScale, ProfileScale, InterpolationFlags.Area);
        using var binary = new Mat();
        Cv2.Threshold(small, binary, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
        int h = binary.Rows, w = binary.Cols;

        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        int n = Cv2.ConnectedComponentsWithStats(binary, labels, stats, centroids, PixelConnectivity.Connectivity8);
        binary.GetArray(out byte[] bytes);
        if (n > 1)
        {
            double limit = BlobMaxFrac * (cells[0].H * ProfileScale);
            var big = new bool[n];
            bool any = false;
            for (int i = 1; i < n; i++)
            {
                big[i] = stats.At<int>(i, 3) > limit;   // CC_STAT_HEIGHT
                any |= big[i];
            }
            if (any)
            {
                labels.GetArray(out int[] lab);
                for (int i = 0; i < bytes.Length; i++)
                {
                    if (big[lab[i]])
                    {
                        bytes[i] = 0;
                    }
                }
            }
        }
        var ink = new float[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            ink[i] = bytes[i] / 255.0f;
        }
        var ones = new float[bytes.Length];
        Array.Fill(ones, 1.0f);

        Point2f centre = new(w / 2.0f, h / 2.0f);
        (int X, int Y, int W, int H)[] boxes = [.. cells.Select(c =>
            ((int)(c.X * ProfileScale), (int)(c.Y * ProfileScale),
             Math.Max(2, (int)(c.W * ProfileScale)), Math.Max(2, (int)(c.H * ProfileScale))))];

        using var inkMat = Mat.FromPixelData(h, w, MatType.CV_32FC1, ink);
        using var validMat = Mat.FromPixelData(h, w, MatType.CV_32FC1, ones);

        double[][] Score(double[] angles)
        {
            var output = new double[angles.Length][];
            for (int a = 0; a < angles.Length; a++)
            {
                output[a] = new double[boxes.Length];
                using Mat m = Cv2.GetRotationMatrix2D(centre, angles[a], 1.0);
                using var rot = new Mat();
                using var cnt = new Mat();
                Cv2.WarpAffine(inkMat, rot, m, new Size(w, h), InterpolationFlags.Nearest,
                    BorderTypes.Constant, Scalar.All(0));
                Cv2.WarpAffine(validMat, cnt, m, new Size(w, h), InterpolationFlags.Nearest,
                    BorderTypes.Constant, Scalar.All(0));
                rot.GetArray(out float[] r);
                cnt.GetArray(out float[] c);
                // cumulative sums along x, float32 (np.cumsum(..., dtype=np.float32))
                for (int y = 0; y < h; y++)
                {
                    for (int x = 1; x < w; x++)
                    {
                        r[y * w + x] += r[y * w + x - 1];
                        c[y * w + x] += c[y * w + x - 1];
                    }
                }
                for (int j = 0; j < boxes.Length; j++)
                {
                    (int bx, int by, int bw, int bh) = boxes[j];
                    int x2 = Math.Min(bx + bw, w) - 1;
                    int yEnd = Math.Min(by + bh, h);
                    int rows = yEnd - by;
                    if (rows <= 0)
                    {
                        continue;
                    }
                    var inkRow = new float[rows];
                    var cntRow = new float[rows];
                    float cmax = float.NegativeInfinity;
                    for (int y = by; y < yEnd; y++)
                    {
                        inkRow[y - by] = r[y * w + x2] - (bx > 0 ? r[y * w + bx - 1] : 0f);
                        cntRow[y - by] = c[y * w + x2] - (bx > 0 ? c[y * w + bx - 1] : 0f);
                        cmax = Math.Max(cmax, cntRow[y - by]);
                    }
                    if (cmax <= 0)
                    {
                        continue;
                    }
                    float threshold = 0.6f * cmax;
                    var prof = new List<float>();
                    for (int y = 0; y < rows; y++)
                    {
                        if (cntRow[y] >= threshold)
                        {
                            prof.Add(inkRow[y] / Math.Max(cntRow[y], 1.0f));
                        }
                    }
                    if (prof.Count > 2)
                    {
                        double mean = 0;
                        foreach (float v in prof)
                        {
                            mean += v;
                        }
                        mean /= prof.Count;
                        double variance = 0;
                        foreach (float v in prof)
                        {
                            variance += (v - mean) * (v - mean);
                        }
                        output[a][j] = (float)(variance / prof.Count);
                    }
                }
            }
            return output;
        }

        double[][] coarse = Score(ProfileCoarse);
        var peaks = new int?[boxes.Length];
        for (int j = 0; j < boxes.Length; j++)
        {
            int ib = 0;
            for (int a = 1; a < coarse.Length; a++)
            {
                if (coarse[a][j] > coarse[ib][j])
                {
                    ib = a;
                }
            }
            peaks[j] = ib == 0 || ib == ProfileCoarse.Length - 1 || coarse[ib][j] <= 0 ? null : ib;
        }
        // one fine pass per distinct coarse peak (usually one or two per page)
        var fineCache = new Dictionary<int, double[][]>();
        foreach (int ib in peaks.Where(p => p is not null).Select(p => p!.Value).Distinct())
        {
            fineCache[ib] = Score([.. FineOffsets.Select(o => ProfileCoarse[ib] + o)]);
        }

        var results = new (double, double)[boxes.Length];
        for (int j = 0; j < boxes.Length; j++)
        {
            if (peaks[j] is not int pk)
            {
                results[j] = (0.0, 0.0);
                continue;
            }
            double[][] fine = fineCache[pk];
            int jb = 0;
            for (int a = 1; a < fine.Length; a++)
            {
                if (fine[a][j] > fine[jb][j])
                {
                    jb = a;
                }
            }
            double best = ProfileCoarse[pk] + FineOffsets[jb];
            if (jb > 0 && jb < fine.Length - 1)
            {
                double y0 = fine[jb - 1][j], y1 = fine[jb][j], y2 = fine[jb + 1][j];
                double den = y0 - 2 * y1 + y2;
                if (den < 0)
                {
                    best += ProfileFineStep * 0.5 * (y0 - y2) / den;
                }
            }
            double med = LineRefine.Median([.. coarse.Select(row => row[j])]);
            results[j] = (best, med > 0 ? fine[jb][j] / med : 0.0);
        }
        return results;
    }

    private static double[] Basis5(double xn, double yn) => [1, xn, yn, xn * xn, xn * yn];

    /// <summary>Robust (Cauchy IRLS) weighted ridge regression of the tilt polynomial. Port of
    /// <c>_fit_tilt</c> (line_dewarp.py:151-163).</summary>
    private static double[] FitTilt(CellRow[] meas, double w, double h)
    {
        int n = meas.Length;
        var a = new double[n][];
        var t = new double[n];
        var wgt = new double[n];
        for (int i = 0; i < n; i++)
        {
            double xn = (meas[i].X - w / 2) / (w / 2);
            double yn = (meas[i].Y - h / 2) / (w / 2);
            a[i] = Basis5(xn, yn);
            t[i] = meas[i].Tilt;
            wgt[i] = meas[i].Weight;
        }
        var c = new double[5];
        for (int iter = 0; iter < IrlsIters; iter++)
        {
            var ata = new double[5, 5];
            var atwt = new double[5];
            for (int i = 0; i < n; i++)
            {
                for (int p = 0; p < 5; p++)
                {
                    atwt[p] += a[i][p] * wgt[i] * t[i];
                    for (int q = 0; q < 5; q++)
                    {
                        ata[p, q] += a[i][p] * wgt[i] * a[i][q];
                    }
                }
            }
            for (int d = 0; d < 5; d++)
            {
                ata[d, d] += Ridge;
            }
            if (!SolveLinear5(ata, atwt, out double[] sol))
            {
                break;
            }
            c = sol;
            for (int i = 0; i < n; i++)
            {
                double r = t[i];
                for (int p = 0; p < 5; p++)
                {
                    r -= a[i][p] * c[p];
                }
                wgt[i] = meas[i].Weight / (1.0 + (r / IrlsScaleDeg) * (r / IrlsScaleDeg));
            }
        }
        return c;
    }

    /// <summary>Gaussian elimination with partial pivoting, sized for the 5-term tilt polynomial.
    /// Kept separate from <see cref="LevenbergMarquardt"/>'s own solver: this is a single
    /// normal-equations solve, not an iterative damped one, and the two modules stay decoupled.</summary>
    private static bool SolveLinear5(double[,] a, double[] b, out double[] x)
    {
        const int n = 5;
        var m = (double[,])a.Clone();
        var v = (double[])b.Clone();
        x = new double[n];
        for (int col = 0; col < n; col++)
        {
            int pivotRow = col;
            double pivotValue = Math.Abs(m[col, col]);
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(m[row, col]) > pivotValue)
                {
                    pivotRow = row;
                    pivotValue = Math.Abs(m[row, col]);
                }
            }
            if (pivotValue < 1e-14)
            {
                return false;
            }
            if (pivotRow != col)
            {
                for (int k = 0; k < n; k++)
                {
                    (m[col, k], m[pivotRow, k]) = (m[pivotRow, k], m[col, k]);
                }
                (v[col], v[pivotRow]) = (v[pivotRow], v[col]);
            }
            for (int row = col + 1; row < n; row++)
            {
                double factor = m[row, col] / m[col, col];
                if (factor == 0.0)
                {
                    continue;
                }
                for (int k = col; k < n; k++)
                {
                    m[row, k] -= factor * m[col, k];
                }
                v[row] -= factor * v[col];
            }
        }
        for (int row = n - 1; row >= 0; row--)
        {
            double sum = v[row];
            for (int k = row + 1; k < n; k++)
            {
                sum -= m[row, k] * x[k];
            }
            if (Math.Abs(m[row, row]) < 1e-14)
            {
                return false;
            }
            x[row] = sum / m[row, row];
        }
        return true;
    }

    /// <summary>
    /// Vertical displacement map v(x, y) in pixels, shape (h, w): the x-integral of the tilt
    /// polynomial, zero on the centre column. Port of <c>displacement</c> (line_dewarp.py:166-176) —
    /// evaluated on a coarse grid (w/8 x h/8) and resized up, matching the reference's own
    /// "smooth cubic" shortcut.
    /// </summary>
    private static float[] Displacement(double[] c, int w, int h)
    {
        double bigW = w, bigH = h;
        int gw = Math.Max(2, w / 8), gh = Math.Max(2, h / 8);
        double c0 = c[0] * Math.PI / 180, c1 = c[1] * Math.PI / 180, c2 = c[2] * Math.PI / 180,
            c3 = c[3] * Math.PI / 180, c4 = c[4] * Math.PI / 180;
        using var grid = new Mat(gh, gw, MatType.CV_32FC1);
        for (int gy = 0; gy < gh; gy++)
        {
            double yRaw = gy * (bigH - 1) / (gh - 1);
            double yn = (yRaw - bigH / 2) / (bigW / 2);
            for (int gx = 0; gx < gw; gx++)
            {
                double xRaw = gx * (bigW - 1) / (gw - 1);
                double xn = (xRaw - bigW / 2) / (bigW / 2);
                double v = c0 * xn + c1 * xn * xn / 2 + c2 * xn * yn + c3 * xn * xn * xn / 3
                    + c4 * xn * xn * yn / 2;
                grid.Set(gy, gx, (float)(v * (bigW / 2)));
            }
        }
        using var full = new Mat();
        Cv2.Resize(grid, full, new Size(w, h), 0, 0, InterpolationFlags.Linear);
        var outArr = new float[h * w];
        var idx = full.GetGenericIndexer<float>();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                outArr[y * w + x] = idx[y, x];
            }
        }
        return outArr;
    }

    /// <summary>
    /// Displacement map (h, w) that unbends the page (<c>y_src = y + v</c>), or null with the reason.
    /// Port of <c>dewarp_by_lines</c> (line_dewarp.py:179-219).
    /// </summary>
    public static (float[]? V, LineRefine.StraightenInfo Info) DewarpByLines(Mat gray, int inset)
    {
        int h = gray.Rows, w = gray.Cols;
        CellRow[] meas = MeasureCells(gray, inset);
        CellRow[] cells = [.. meas.Where(m => m.Source == 1)];
        if (cells.Length < MinCells)
        {
            return (null, new LineRefine.StraightenInfo(false, "too few cells"));
        }
        double before = LineRefine.WMedian([.. cells.Select(c => Math.Abs(c.Tilt))],
            [.. cells.Select(c => c.Weight)]);

        double[] c = FitTilt(meas, w, h);
        float[] v = Displacement(c, w, h);
        double vmax = v.Length > 0 ? v.Max(x => Math.Abs(x)) : 0.0;
        if (vmax < MinDispPx)
        {
            return (null, new LineRefine.StraightenInfo(false, "flat enough"));
        }
        if (vmax > MaxDispFrac * h)
        {
            return (null, new LineRefine.StraightenInfo(false, "bend too large"));
        }

        // Residual judged on ALL the evidence, segments included: a map that satisfies the cells but
        // tilts the long segments is a wrong map (line_dewarp.py:204-207).
        var afterAll = new double[meas.Length];
        var beforeAll = new double[meas.Length];
        var weights = new double[meas.Length];
        var afterCellsVals = new List<double>();
        var afterCellsW = new List<double>();
        for (int i = 0; i < meas.Length; i++)
        {
            double xn = (meas[i].X - w / 2.0) / (w / 2.0);
            double yn = (meas[i].Y - h / 2.0) / (w / 2.0);
            double[] basis = Basis5(xn, yn);
            double fit = 0;
            for (int p = 0; p < 5; p++)
            {
                fit += basis[p] * c[p];
            }
            afterAll[i] = Math.Abs(meas[i].Tilt - fit);
            beforeAll[i] = Math.Abs(meas[i].Tilt);
            weights[i] = meas[i].Weight;
            if (meas[i].Source == 1)
            {
                afterCellsVals.Add(afterAll[i]);
                afterCellsW.Add(meas[i].Weight);
            }
        }
        double after = LineRefine.WMedian([.. afterCellsVals], [.. afterCellsW]);
        double beforeAllM = LineRefine.WMedian(beforeAll, weights);
        double afterAllM = LineRefine.WMedian(afterAll, weights);
        if (after > (1.0 - MinGain) * before || afterAllM > beforeAllM)
        {
            return (null, new LineRefine.StraightenInfo(false, "no gain"));
        }
        return (v, new LineRefine.StraightenInfo(true, null));
    }

    /// <summary>Remaps a page by the vertical displacement map. Port of <c>apply_dewarp</c>
    /// (line_dewarp.py:222-225).</summary>
    public static Image ApplyDewarp(Image page, float[] v)
    {
        int w = page.Width, h = page.Height;
        using var mapX = new Mat(h, w, MatType.CV_32FC1);
        using var mapY = new Mat(h, w, MatType.CV_32FC1);
        var mxIdx = mapX.GetGenericIndexer<float>();
        var myIdx = mapY.GetGenericIndexer<float>();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                mxIdx[y, x] = x;
                myIdx[y, x] = (float)(y + v[y * w + x]);
            }
        }
        var dst = new Mat();
        Cv2.Remap(page.Mat, dst, mapX, mapY, InterpolationFlags.Linear, BorderTypes.Replicate);
        return Image.Wrap(dst);
    }
}
