using RussianDocs.DocumentProcessing.Imaging;

namespace RussianDocs.DocumentProcessing.Maps;

/// <summary>
/// Where a point of a pipeline canvas lies on the input image. Port of <c>document_processing/geometry.py</c>
/// (PR #19, issue #18).
///
/// <para>
/// A caller that shows a document to a person needs the field on the PHOTO, not on the canvas the pipeline
/// built: the canvas is resized, turned upright, warped page by page, stitched and deskewed. Every stage
/// that changes the geometry of the image it passes on describes that change as a map from its OUTPUT back
/// to its INPUT, and the pipeline chains the maps of the stages that actually ran, in the order they ran.
/// </para>
///
/// <para>
/// <b>Coordinates are continuous pixels.</b> An image spans [0, width] x [0, height], and the pixel with index
/// (i, j) is the unit square centred at (i + 0.5, j + 0.5); box edges from the detectors are read the same
/// way. OpenCV's warps put pixel centres on integer coordinates instead, so <see cref="HomographyMap"/>
/// shifts by half a pixel on the way in and out and the matrices OpenCV computed are used unchanged.
/// <c>cv2.resize</c> and <c>cv2.rotate</c> need no shift in these coordinates.
/// </para>
///
/// <para>
/// <b>A map receives the points of ONE shape.</b> A stitched canvas picks the piece a shape lies on by the
/// shape's centroid, so the corners of one box are never sent to different pages.
/// </para>
///
/// <para>
/// Named <c>...Map</c> here because <c>Geometry</c> and <c>Homography</c> are already taken in this library
/// (the imaging helpers and the registration matrices); the reference's names are in each summary.
/// </para>
/// </summary>
public interface IMap
{
    /// <summary>
    /// Points of the output, on the input; null - the way back is not known.
    ///
    /// <para>
    /// NOT KNOWN IS NOT THE SAME AS UNCHANGED: a stage that passes its image on untouched contributes no map
    /// at all (<see cref="ChainMap.Then"/> with null), while a stage whose effect no point map can express
    /// contributes <see cref="UnknownMap"/> and makes every answer downstream null. A box that quietly lands
    /// somewhere else is worse than no box: the caller draws it over a photo and trusts it.
    /// </para>
    /// </summary>
    Point[]? ToInput(IReadOnlyList<Point> points);
}

/// <summary>Corners of an axis-aligned box, clockwise from the top-left (<c>geometry.corners</c>).</summary>
public static class Corners
{
    public static Point[] Of(double x0, double y0, double x1, double y1) =>
        [new(x0, y0), new(x1, y0), new(x1, y1), new(x0, y1)];
}

/// <summary>The input resized: output = input * (sx, sy). <c>Scale</c> in the reference.</summary>
public sealed record ScaleMap(double Sx, double Sy) : IMap
{
    public Point[]? ToInput(IReadOnlyList<Point> points) =>
        [.. points.Select(p => new Point(p.X / Sx, p.Y / Sy))];
}

/// <summary>The input shifted: output = input + (dx, dy); a crop is a negative shift. <c>Offset</c> in the reference.</summary>
public sealed record OffsetMap(double Dx, double Dy) : IMap
{
    public Point[]? ToInput(IReadOnlyList<Point> points) =>
        [.. points.Select(p => new Point(p.X - Dx, p.Y - Dy))];
}

/// <summary><c>cv2.ROTATE_90_COUNTERCLOCKWISE</c> applied <paramref name="Turns"/> times to a width x height input. <c>QuarterTurns</c>.</summary>
public sealed record QuarterTurnsMap(int Width, int Height, int Turns) : IMap
{
    public Point[]? ToInput(IReadOnlyList<Point> points)
    {
        Point[] current = [.. points];
        var widths = new List<int>();
        int w = Width, h = Height;
        int turns = ((Turns % 4) + 4) % 4;   // Python's % is never negative
        for (int i = 0; i < turns; i++)
        {
            widths.Add(w);
            (w, h) = (h, w);
        }
        // One turn sends (x, y) of a W-wide image to (y, W - x); the last turn is undone first.
        for (int i = widths.Count - 1; i >= 0; i--)
        {
            int width = widths[i];
            current = [.. current.Select(p => new Point(width - p.Y, p.X))];
        }
        return current;
    }
}

/// <summary>
/// The output of <c>cv2.warpPerspective</c> or <c>cv2.warpAffine</c> with <c>matrix</c> (input to output).
/// <c>Homography</c> in the reference; a 2x3 affine matrix is completed to 3x3.
/// </summary>
public sealed class HomographyMap : IMap
{
    private readonly double[,] _matrix;

    public HomographyMap(double[,] matrix)
    {
        if (matrix.GetLength(1) != 3 || matrix.GetLength(0) is not (2 or 3))
        {
            throw new ArgumentException("maps: a homography is 3x3, or 2x3 for an affine map");
        }
        _matrix = new double[3, 3];
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                _matrix[i, j] = matrix[i, j];
            }
        }
        if (matrix.GetLength(0) == 2)
        {
            _matrix[2, 2] = 1.0;
        }
    }

    /// <summary>The matrix, 3x3 input to output.</summary>
    public double[,] Matrix => (double[,])_matrix.Clone();

    public Point[]? ToInput(IReadOnlyList<Point> points)
    {
        double[,]? inverse = Invert(_matrix);
        if (inverse is null)
        {
            return null;
        }
        var result = new Point[points.Count];
        for (int i = 0; i < result.Length; i++)
        {
            double x = points[i].X - 0.5, y = points[i].Y - 0.5;
            double mx = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2];
            double my = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2];
            double mw = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2];
            result[i] = new Point(mx / mw + 0.5, my / mw + 0.5);
        }
        return result;
    }

    /// <summary>3x3 inverse by the adjugate; null for a singular matrix (<c>np.linalg.inv</c> raises there).</summary>
    private static double[,]? Invert(double[,] m)
    {
        double c00 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1];
        double c01 = m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2];
        double c02 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0];
        double det = m[0, 0] * c00 + m[0, 1] * c01 + m[0, 2] * c02;
        if (det == 0.0 || double.IsNaN(det))
        {
            return null;
        }
        double inv = 1.0 / det;
        return new[,]
        {
            { c00 * inv, (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * inv, (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * inv },
            { c01 * inv, (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * inv, (m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]) * inv },
            { c02 * inv, (m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]) * inv, (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) * inv },
        };
    }
}

/// <summary>
/// The output of <c>cv2.remap(img, xs, ys + v)</c>: a page unbent by a vertical displacement map.
/// <c>VerticalRemap</c> in the reference.
///
/// <para>
/// The bend map of <c>page_registration.line_dewarp</c> is exactly this call, so the output pixel (i, j)
/// SAMPLED the input at (i, j + v[j, i]) - the map is already the way back, no matrix and no inversion
/// needed. <c>v</c> is read bilinearly between pixel centres and held at the edge outside the map, the way
/// the remap replicated its border. A point of the output moves only along y.
/// </para>
/// </summary>
public sealed class BendMap(float[] v, int width, int height) : IMap
{
    public Point[]? ToInput(IReadOnlyList<Point> points)
    {
        var result = new Point[points.Count];
        for (int i = 0; i < result.Length; i++)
        {
            // sample v at the output point in OpenCV's pixel-centre coordinates
            double x = Math.Clamp(points[i].X - 0.5, 0.0, width - 1.0);
            double y = Math.Clamp(points[i].Y - 0.5, 0.0, height - 1.0);
            int x0 = (int)Math.Floor(x), y0 = (int)Math.Floor(y);
            int x1 = Math.Min(x0 + 1, width - 1), y1 = Math.Min(y0 + 1, height - 1);
            double fx = x - x0, fy = y - y0;
            double shift = v[y0 * width + x0] * (1 - fx) * (1 - fy) + v[y0 * width + x1] * fx * (1 - fy)
                + v[y1 * width + x0] * (1 - fx) * fy + v[y1 * width + x1] * fx * fy;
            result[i] = new Point(points[i].X, points[i].Y + shift);
        }
        return result;
    }
}

/// <summary>
/// A stage that changed the image in a way no point map expresses. <c>Unknown</c> in the reference: the canvas
/// is still correct and recognition is unaffected - only the way back is gone, and it stays gone for every
/// stage after this one.
/// </summary>
public sealed record UnknownMap : IMap
{
    public Point[]? ToInput(IReadOnlyList<Point> points) => null;
}

/// <summary>Maps of stages in the order the stages ran: the first one reads the input. <c>Chain</c> in the reference.</summary>
public sealed class ChainMap(IReadOnlyList<IMap> maps) : IMap
{
    public ChainMap() : this([])
    {
    }

    public IReadOnlyList<IMap> Maps { get; } = maps;

    /// <summary>This chain followed by one more stage; null - the stage passed its input on unchanged.</summary>
    public ChainMap Then(IMap? later) => later is null ? this : new ChainMap([.. Maps, later]);

    /// <summary>The chain is READ output to input: the last stage's map answers first.</summary>
    public Point[]? ToInput(IReadOnlyList<Point> points)
    {
        Point[]? current = [.. points];
        for (int i = Maps.Count - 1; i >= 0; i--)
        {
            current = Maps[i].ToInput(current);
            // One stage that cannot answer ends the walk: the stages before it are fine, but their input
            // is no longer known.
            if (current is null)
            {
                return null;
            }
        }
        return current;
    }
}

/// <summary>
/// A canvas made of pieces: a rectangle (x0, y0, x1, y1) of the output and the map of what fills it.
/// <c>Pieces</c> in the reference.
/// </summary>
public sealed class PiecesMap(IReadOnlyList<((double X0, double Y0, double X1, double Y1) Rect, IMap Map)> pieces) : IMap
{
    public IReadOnlyList<((double X0, double Y0, double X1, double Y1) Rect, IMap Map)> Pieces { get; } = pieces;

    public Point[]? ToInput(IReadOnlyList<Point> points)
    {
        double cx = points.Average(p => p.X), cy = points.Average(p => p.Y);

        double Distance((double X0, double Y0, double X1, double Y1) r)
        {
            double dx = Math.Max(Math.Max(r.X0 - cx, 0.0), cx - r.X1);
            double dy = Math.Max(Math.Max(r.Y0 - cy, 0.0), cy - r.Y1);
            return dx * dx + dy * dy;
        }

        // min() keeps the FIRST of equal distances
        int best = 0;
        double bestDistance = Distance(Pieces[0].Rect);
        for (int i = 1; i < Pieces.Count; i++)
        {
            double d = Distance(Pieces[i].Rect);
            if (d < bestDistance)
            {
                best = i;
                bestDistance = d;
            }
        }
        return Pieces[best].Map.ToInput(points);   // null if that piece cannot answer
    }
}
