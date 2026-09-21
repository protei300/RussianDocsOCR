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
/// <b>Simplified relative to the reference, deliberately — matching the Go port's own choice:</b>
/// Python's <c>_cell_tilts</c> rotates the WHOLE region once per candidate angle and slices every
/// cell's row-sum out of a shared cumulative sum (a performance optimisation: 25 cells x 22 angles
/// would otherwise be 550 small warps). This port instead calls <see cref="LineRefine.ProfileTilt"/>
/// PER CELL directly — simpler, reuses already-verified code, and is if anything MORE faithful to
/// the per-cell blob-height threshold than Python's own approximation (which reuses the FIRST cell's
/// height, <c>cells[0][3]</c>, for every cell's blob filter — line_dewarp.py:84). Slower in the
/// abstract, but page counts here are always 1-2 per document, so the cost is not observable.
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
        Cv2.Resize(gray, half, new Size(Math.Max(1, gray.Cols / 2), Math.Max(1, gray.Rows / 2)),
            0, 0, InterpolationFlags.Area);
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
        foreach (double yb in LineRefine.Linspace(y0, y1b - ch, Grid))
        {
            foreach (double xb in LineRefine.Linspace(x0, x1b - cw, Grid))
            {
                int left = (int)xb, top = (int)yb, right = (int)(xb + cw), bottom = (int)(yb + ch);
                if (right <= left || bottom <= top)
                {
                    continue;
                }
                using Mat cell = new(gray, new Rect(left, top, right - left, bottom - top));
                (double tilt, double ratio) = LineRefine.ProfileTilt(cell);
                if (ratio >= ProfileMinPeak)
                {
                    double weight = ProfileWeight * Math.Min(1.0, ratio - 1.0);
                    rows.Add(new CellRow(xb + 0.5 * cw, yb + 0.5 * ch, tilt, weight, 1));
                }
            }
        }
        return [.. rows];
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
