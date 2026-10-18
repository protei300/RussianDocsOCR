using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Maps;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// One registration stamp of the registration page (<c>results.registration_stamps</c> in the reference, an
/// entry of <c>meta_results['RegistrationStamps']</c>).
/// </summary>
public sealed class RegistrationStamp
{
    /// <summary><c>registered</c>, <c>deregistered</c> or <c>unclear</c>.</summary>
    public string Kind { get; init; } = "unclear";

    /// <summary>The detector's score, a float32 widened to double.</summary>
    public double Confidence { get; init; }

    /// <summary>Corners on the image passed to <c>Run</c>, clockwise from the top-left of the upright stamp; null - the way back is not known.</summary>
    public Point[]? Quad { get; init; }

    /// <summary>Indices into the address lines of the lines that lie inside this stamp.</summary>
    public List<int> Lines { get; } = [];

    /// <summary>The stamp's own lines joined by a newline (a handwritten line reads as the placeholder).</summary>
    public string Address { get; set; } = "";

    /// <summary>For a «снят» stamp, the index (in the same list) of the registration it cancels; null otherwise.</summary>
    public int? Cancels { get; set; }

    /// <summary>True on the registration in force: the freshest one nothing cancels.</summary>
    public bool Current { get; set; }
}

/// <summary>
/// Registration stamps on the registration page (INTPASSPORTADDR, issue #27): which stamps there are, which
/// address lines belong to which stamp, and the order of the stamps. Port of <c>pipeline/stamps.py</c>
/// (<c>StampsStage</c>).
/// </summary>
public static class Stamps
{
    /// <summary>
    /// A canvas wider than this many heights is a spread of two pages, split down the middle (the
    /// page-registration canvas is either one page or a spread).
    /// </summary>
    public const double SpreadAspect = 1.2;

    /// <summary>
    /// Finds the stamps on the address canvas and ties the address lines to them. Runs after the lines are read,
    /// on the same canvas.
    ///
    /// <para>
    /// A line belongs to the stamp its CENTRE lies in (the reference measured it on the 19 held-out canvases of
    /// the issue's author: all 113 lines fell inside a stamp); a line inside no stamp keeps no stamp. A «снят»
    /// stamp holds no address - it is tied to the registration it cancels, and the registration nothing cancels
    /// is the current one. The stamps come freshest first (<see cref="Order"/>). The per-stamp address is the
    /// one that keeps a registration's lines together: the page-wide <c>Address</c> follows the reading order,
    /// so the lines of two stamps side by side alternate in it.
    /// </para>
    /// </summary>
    /// <param name="lines">The address lines as read; each gets its <c>Stamp</c> set.</param>
    /// <param name="lineBoxes">The oriented box of each line, aligned with <paramref name="lines"/>.</param>
    /// <param name="toInput">Canvas points to the input image, null when the way back is not known.</param>
    public static List<RegistrationStamp> Find(Image canvas, RegistrationStamps detector,
        IReadOnlyList<AddressLineMeta> lines, IReadOnlyList<ObbBox> lineBoxes,
        Func<IReadOnlyList<Point>, Point[]?> toInput)
    {
        List<ObbBox> boxes = detector.Predict(canvas);
        List<Point[]> quads = [.. boxes.Select(ObbQuad)];
        List<int> order = Order(quads, canvas.Width, canvas.Height);
        var position = new Dictionary<int, int>();
        for (int k = 0; k < order.Count; k++)
        {
            position[order[k]] = k;
        }

        var stamps = new List<RegistrationStamp>(order.Count);
        foreach (int det in order)
        {
            stamps.Add(new RegistrationStamp
            {
                Kind = RegistrationStamps.Kinds.GetValueOrDefault(boxes[det].Label, "unclear"),
                Confidence = boxes[det].Conf,
                Quad = toInput(quads[det]),
            });
        }

        for (int li = 0; li < Math.Min(lines.Count, lineBoxes.Count); li++)
        {
            int? det = ContainingStamp(new Point(lineBoxes[li].Cx, lineBoxes[li].Cy), quads);
            lines[li].Stamp = det is int d ? position[d] : null;
            if (det is int found)
            {
                stamps[position[found]].Lines.Add(li);
            }
        }
        foreach (RegistrationStamp stamp in stamps)
        {
            stamp.Address = string.Join("\n", stamp.Lines.Select(li =>
                lines[li].Kind == "handwritten" ? AddressReader.HandwrittenPlaceholder : lines[li].Text ?? ""));
        }
        TieCancellations(stamps);
        return stamps;
    }

    /// <summary>
    /// Detection indices, freshest stamp first (<c>_order_stamps</c>).
    ///
    /// <para>
    /// The page is filled in order: on a spread the left page before the right one, on a page top to bottom; the
    /// freshest stamp is the last one filled. Checked by the reference on 15 labelled spreads with at least
    /// three stamps and a «снят» each, dates read by eye: the chronology matched this order on all 15. Two
    /// columns of stamps on one page did not occur there and are not covered (x only breaks ties).
    /// </para>
    /// </summary>
    public static List<int> Order(IReadOnlyList<Point[]> quads, int width, int height)
    {
        bool spread = width > SpreadAspect * height;

        (int Page, double Cy, double Cx) Filled(int k)
        {
            (double cx, double cy) = Centre(quads[k]);
            return (spread && cx >= width / 2.0 ? 1 : 0, cy, cx);
        }

        // sorted(range(n), key=filled)[::-1]: a stable ascending sort, then the whole list turned round
        List<int> ascending = [.. Enumerable.Range(0, quads.Count)
            .OrderBy(k => Filled(k).Page).ThenBy(k => Filled(k).Cy).ThenBy(k => Filled(k).Cx)];
        ascending.Reverse();
        return ascending;
    }

    /// <summary>
    /// Sets <c>Cancels</c> on every «снят» stamp and <c>Current</c> on the registration in force
    /// (<c>_tie_cancellations</c>). <paramref name="stamps"/> is freshest first.
    ///
    /// <para>
    /// A «снят» is put in the next free place right after the registration it cancels, so it cancels the nearest
    /// registration filled before it - the next <c>registered</c> in the list; one that opens the canvas cancels a
    /// registration on a page not in the photo, and keeps null. The current registration is the freshest one
    /// nothing cancels.
    /// </para>
    /// </summary>
    public static void TieCancellations(List<RegistrationStamp> stamps)
    {
        var cancelled = new HashSet<int>();
        for (int k = 0; k < stamps.Count; k++)
        {
            if (stamps[k].Kind != "deregistered")
            {
                continue;
            }
            int? j = null;
            for (int next = k + 1; next < stamps.Count; next++)
            {
                if (stamps[next].Kind == "registered")
                {
                    j = next;
                    break;
                }
            }
            stamps[k].Cancels = j;
            if (j is int hit)
            {
                cancelled.Add(hit);
            }
        }
        for (int k = 0; k < stamps.Count; k++)
        {
            if (stamps[k].Kind == "registered" && !cancelled.Contains(k))
            {
                stamps[k].Current = true;
                break;
            }
        }
    }

    /// <summary>
    /// The stamp's corners on the canvas, clockwise from the top-left corner of its upright patch - the same
    /// convention as the address-line quads (<c>_obbox_quad</c>).
    /// </summary>
    public static Point[] ObbQuad(ObbBox box)
    {
        (int w, int h) = AddressLinesDetector.PatchSize(box);
        return AddressLinesDetector.CropGeometry(box).ToInput(Corners.Of(0, 0, w, h))
            ?? throw new InvalidDataException("pipeline: a stamp's crop geometry has no inverse");
    }

    /// <summary>
    /// Index of the stamp whose quad holds the point; when quads overlap, the one whose centre is nearest. Null
    /// when no quad holds it (<c>_containing_stamp</c>). The point is inside when the cross products of every edge
    /// with the vector to the point share a sign.
    /// </summary>
    public static int? ContainingStamp(Point point, IReadOnlyList<Point[]> quads)
    {
        int? best = null;
        double bestDistance = 0;
        for (int k = 0; k < quads.Count; k++)
        {
            Point[] quad = quads[k];
            bool anyNegative = false, anyPositive = false;
            for (int e = 0; e < 4; e++)
            {
                Point from = quad[e], to = quad[(e + 1) % 4];
                double cross = (to.X - from.X) * (point.Y - from.Y) - (to.Y - from.Y) * (point.X - from.X);
                anyNegative |= cross < 0;
                anyPositive |= cross > 0;
            }
            // `(cross >= 0).all() or (cross <= 0).all()`
            if (anyNegative && anyPositive)
            {
                continue;
            }
            (double cx, double cy) = Centre(quad);
            double distance = double.Hypot(point.X - cx, point.Y - cy);
            if (best is null || distance < bestDistance)
            {
                best = k;
                bestDistance = distance;
            }
        }
        return best;
    }

    /// <summary><c>quad.mean(axis=0)</c>: four values added in order, then divided.</summary>
    private static (double X, double Y) Centre(Point[] quad) =>
        (((quad[0].X + quad[1].X) + quad[2].X + quad[3].X) / 4.0,
         ((quad[0].Y + quad[1].Y) + quad[2].Y + quad[3].Y) / 4.0);
}
