using RussianDocs.DocumentProcessing.Maps;
using RussianDocs.DocumentProcessing.Modules;
using RussianDocs.DocumentProcessing.Postprocess;
using Point = RussianDocs.DocumentProcessing.Imaging.Point;

namespace RussianDocs.DocumentProcessing.Pipeline;

/// <summary>
/// Field and word quadrilaterals on the input image: the content of the conformance stage <c>quads</c> and of
/// <c>results.field_quads</c> / <c>word_quads</c>.
/// </summary>
/// <param name="Fields">label to one quadrilateral per detection, in the order the fields were read; null when
/// the way back is not known for this run.</param>
/// <param name="Words">label to the quadrilateral of each word patch of that field; null on the same terms.</param>
public sealed record FieldQuadSet(Dictionary<string, List<Point[]>>? Fields, Dictionary<string, List<Point[]>>? Words)
{
    /// <summary>
    /// The stage payload: <c>{"fields": {label: [quad, ...]}, "words": {...}}</c>, quad = four <c>[x, y]</c>
    /// from the top-left corner of the canvas box, numbers NOT rounded; both keys null when unknown.
    /// (<c>"address_lines"</c> exists only on the address path, which this port does not read.)
    /// </summary>
    public Dictionary<string, object?> Payload() => new()
    {
        ["fields"] = Wire(Fields),
        ["words"] = Wire(Words),
    };

    private static Dictionary<string, double[][][]>? Wire(Dictionary<string, List<Point[]>>? quads) =>
        quads?.ToDictionary(kv => kv.Key,
            kv => kv.Value.Select(q => q.Select(p => new[] { p.X, p.Y }).ToArray()).ToArray(),
            StringComparer.Ordinal);
}

public static class FieldQuads
{
    /// <summary>
    /// Where the read fields and their word patches lie on the input image. Port of <c>Pipeline._field_quads</c>.
    ///
    /// <para>
    /// A field patch is the canvas cut at its box - its frame (<see cref="Field.Origin"/>, size as cut) says
    /// where - and the series/number patch is also turned; a word patch is its box cut from the field patch the
    /// way the words detector cuts it. **The chain is read output to input**: the first port of PR #19 built a
    /// map "forward" and sent every box off the photo. **Not known is said once**, for the whole run, not per
    /// box: both results are then null.
    /// </para>
    /// </summary>
    /// <param name="canvasMap">Canvas the fields were read on, back to the input image (<c>results.geometry</c>).</param>
    public static FieldQuadSet Build(IMap? canvasMap, IReadOnlyList<Field> fields, SplitOutcome split,
        OcrOptions options)
    {
        var fieldQuads = new Dictionary<string, List<Point[]>>(StringComparer.Ordinal);
        var wordQuads = new Dictionary<string, List<Point[]>>(StringComparer.Ordinal);

        Point[]? ToInput(IReadOnlyList<Point> points) => canvasMap is null ? [.. points] : canvasMap.ToInput(points);

        foreach (SplitDetection detection in split.Detections)
        {
            Field field = fields[detection.FieldIndex];
            string label = detection.Label;
            var toCanvas = new OffsetMap(-field.Origin.X, -field.Origin.Y);

            Point[]? onInput = ToInput(toCanvas.ToInput(Corners.Of(0.0, 0.0, field.CutWidth, field.CutHeight))!);
            if (onInput is null)
            {
                return new FieldQuadSet(null, null);
            }
            Add(fieldQuads, label, onInput);

            if (detection.WordBoxes is null)
            {
                Add(wordQuads, label, onInput);
                continue;
            }
            int h = field.Patch.Height, w = field.Patch.Width;
            var patch = new ChainMap([toCanvas]);
            if (options.NeedsLicenceRotation && label == "Licence_number")
            {
                // the patch the words were found on is the field turned once: w x h of h x w
                patch = patch.Then(new QuarterTurnsMap(h, w, 1));
            }
            foreach (Box box in detection.WordBoxes)
            {
                double wx0 = Math.Max(0, (int)box.X1), wy0 = Math.Max(0, (int)box.Y1);
                double wx1 = Math.Min(w, (int)box.X2), wy1 = Math.Min(h, (int)box.Y2);
                Point[] quad = patch.ToInput(Corners.Of(wx0, wy0, wx1, wy1))!;
                Add(wordQuads, label, ToInput(quad)!);
            }
        }
        return new FieldQuadSet(fieldQuads, wordQuads);
    }

    private static void Add(Dictionary<string, List<Point[]>> into, string label, Point[] quad)
    {
        if (!into.TryGetValue(label, out List<Point[]>? list))
        {
            into[label] = list = [];
        }
        list.Add(quad);
    }
}
