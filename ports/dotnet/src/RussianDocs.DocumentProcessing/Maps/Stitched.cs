namespace RussianDocs.DocumentProcessing.Maps;

/// <summary>What <c>stitch_pages</c> did to one page: the resize <paramref name="Scale"/> and where the page starts on the canvas.</summary>
public readonly record struct Placement(double Scale, double Dx, double Dy);

/// <summary>
/// The maps of a canvas made by <c>stitch_pages</c> (<c>image_transformation.stitched_geometry</c>,
/// <c>placement_rect</c>, <c>_placed</c>).
/// </summary>
public static class Stitched
{
    /// <summary>
    /// Where the stitch put a page on the canvas: (dx, dy, width, height). The width and height are the page's
    /// size after the resize to the common side, with the rounding the stitch applies - so a map built from
    /// them puts a point where the page's pixels went, not where <c>scale</c> alone would put it.
    /// </summary>
    public static (double Dx, double Dy, int W, int H) Rect(int width, int height, Placement placement)
    {
        int newW = placement.Scale != 1.0 ? Math.Max(1, (int)Math.Round(width * placement.Scale, MidpointRounding.ToEven)) : width;
        int newH = placement.Scale != 1.0 ? Math.Max(1, (int)Math.Round(height * placement.Scale, MidpointRounding.ToEven)) : height;
        return (placement.Dx, placement.Dy, newW, newH);
    }

    /// <summary>
    /// Map from a canvas built by the stitch back to the image the pages came from: one rectangle of the
    /// canvas per page, with the page's own resize and offset composed after the page's own map.
    /// </summary>
    /// <param name="pages">Sizes of the pages handed to the stitch, in order.</param>
    /// <param name="placements">What the stitch returned for them, same order.</param>
    /// <param name="pageMaps">One map per page, back to the image the pages were cut from; null for a page
    /// that was handed on unchanged.</param>
    public static PiecesMap Geometry(IReadOnlyList<(int W, int H)> pages, IReadOnlyList<Placement> placements,
        IReadOnlyList<IMap?> pageMaps)
    {
        var pieces = new List<((double, double, double, double), IMap)>(pages.Count);
        for (int i = 0; i < pages.Count; i++)
        {
            (double dx, double dy, int newW, int newH) = Rect(pages[i].W, pages[i].H, placements[i]);
            var placed = new ChainMap([new ScaleMap((double)newW / pages[i].W, (double)newH / pages[i].H),
                new OffsetMap(dx, dy)]);
            ChainMap chain = pageMaps[i] is null ? new ChainMap([placed]) : new ChainMap([pageMaps[i]!, placed]);
            pieces.Add(((dx, dy, dx + newW, dy + newH), chain));
        }
        return new PiecesMap(pieces);
    }

    /// <summary>
    /// Placements for pages stacked in <paramref name="order"/>: each is resized to the common side
    /// (<c>scale = common / its own side</c>, the other side rounded half to even) and put after the
    /// previous ones. One entry per INPUT page, in input order (<c>stitch_pages</c>' <c>placements</c>).
    /// </summary>
    public static (Placement[] Placements, int[] NewWidth, int[] NewHeight) Place(
        IReadOnlyList<(int W, int H)> pages, IReadOnlyList<int> order, bool horizontal)
    {
        var placements = new Placement[pages.Count];
        var newW = new int[pages.Count];
        var newH = new int[pages.Count];
        int common = horizontal ? order.Min(i => pages[i].H) : order.Min(i => pages[i].W);
        double offset = 0;
        foreach (int i in order)
        {
            (int w, int h) = pages[i];
            if (horizontal)
            {
                double scale = (double)common / h;
                newW[i] = Math.Max(1, (int)Math.Round(w * scale, MidpointRounding.ToEven));
                newH[i] = common;
                placements[i] = new Placement(scale, offset, 0.0);
                offset += newW[i];
            }
            else
            {
                double scale = (double)common / w;
                newH[i] = Math.Max(1, (int)Math.Round(h * scale, MidpointRounding.ToEven));
                newW[i] = common;
                placements[i] = new Placement(scale, 0.0, offset);
                offset += newH[i];
            }
        }
        return (placements, newW, newH);
    }
}
