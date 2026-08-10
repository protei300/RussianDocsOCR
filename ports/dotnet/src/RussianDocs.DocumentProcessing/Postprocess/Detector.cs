using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Postprocess;

/// <summary>One detection. Mutable, because the coordinates are rewritten in place when mapped back.</summary>
public sealed class Box
{
    public double X1, Y1, X2, Y2;
    public double Conf;
    public int Cls;
    public string Label = "";

    /// <summary>Mask coefficients, for a segmentation head. Empty for a plain detector.</summary>
    public double[] Seg = [];

    public Box Clone() => new()
    {
        X1 = X1, Y1 = Y1, X2 = X2, Y2 = Y2, Conf = Conf, Cls = Cls, Label = Label,
        Seg = (double[])Seg.Clone(),
    };
}

public sealed record DetectResult(List<Box> Boxes) : IResult;

/// <summary>Which suppression the model's output type asks for.</summary>
public enum NmsMode
{
    ClassAgnostic,
    PerClass,
}

/// <summary>
/// Decodes an anchors-first YOLO output and suppresses overlaps.
///
/// <para>
/// One type with an <see cref="NmsMode"/> field rather than a base class and a subclass overriding
/// the suppression — CONVENTIONS §5. The reference has <c>PerClassYOLODetectorPostprocessing</c>
/// inheriting from the plain one; flattening it removes a place where Go's non-virtual embedding
/// would silently call the wrong method, and reads the same in all four languages.
/// </para>
/// </summary>
public sealed class YoloDetector(string[] labels, double iou, double cls, NmsMode mode)
    : IPostprocessor
{
    /// <summary>
    /// When set, the decode stops before truncation and labelling.
    ///
    /// <para>
    /// The segmentation path needs the RAW float coordinates and the mask coefficients — truncating
    /// first loses sub-pixel information the mask crop depends on.
    /// </para>
    /// </summary>
    public bool NumpyOnly { get; private init; }

    /// <summary>
    /// A copy with raw-coordinate mode on.
    ///
    /// <para>
    /// Exists so the loader switch stays the ONE place a detector is constructed: the segmentation
    /// model needs the same detector with one flag flipped, and building a second one there would
    /// duplicate the argument list that the switch already owns.
    /// </para>
    /// </summary>
    public YoloDetector WithNumpyOnly() =>
        new(labels, iou, cls, mode) { NumpyOnly = true };

    public IResult Apply(NdArray output, Context context) =>
        new DetectResult(Decode(output, context));

    public List<Box> Decode(NdArray output, Context context)
    {
        ReadOnlySpan<float> data = output.AsFloat32();
        int[] shape = output.Shape;
        if (shape.Length == 3 && shape[0] == 1)
        {
            shape = shape[1..];
        }
        if (shape.Length != 2)
        {
            throw new InvalidDataException(
                $"postprocess: detector expects [anchors, 4+nc(+seg)], got " +
                $"{NdArray.Describe(output.Shape)}");
        }

        int anchors = shape[0], stride = shape[1];
        int nc = labels.Length;
        if (stride < 4 + nc)
        {
            throw new InvalidDataException(
                $"postprocess: row width {stride} is too small for {nc} classes");
        }
        int segLen = stride - 4 - nc;

        var boxes = new List<Box>(64);
        for (int a = 0; a < anchors; a++)
        {
            ReadOnlySpan<float> row = data.Slice(a * stride, stride);

            // Strict `>` for the best class, matching np.argmax's first-maximum rule.
            int best = 0;
            double bestScore = double.NegativeInfinity;
            for (int c = 0; c < nc; c++)
            {
                if (row[4 + c] > bestScore)
                {
                    best = c;
                    bestScore = row[4 + c];
                }
            }
            // `!(score > cls)` rather than `score <= cls`: identical for real numbers, and it keeps
            // the reference's own spelling, which also handles NaN the same way.
            if (!(bestScore > cls))
            {
                continue;
            }

            double cx = row[0], cy = row[1], w = row[2], h = row[3];
            var box = new Box
            {
                X1 = cx - w / 2, Y1 = cy - h / 2,
                X2 = cx + w / 2, Y2 = cy + h / 2,
                Conf = bestScore, Cls = best,
            };
            if (segLen > 0)
            {
                box.Seg = new double[segLen];
                for (int i = 0; i < segLen; i++)
                {
                    box.Seg[i] = row[4 + nc + i];
                }
            }
            boxes.Add(box);
        }

        if (boxes.Count == 0)
        {
            return [];
        }

        List<int> keep = mode == NmsMode.PerClass
            ? NmsPerClass(boxes, iou)
            : Nms([.. Enumerable.Range(0, boxes.Count)], boxes, iou);

        var kept = keep.Select(i => boxes[i]).ToList();

        // **Stable sort, reading order.** LINQ's OrderBy is stable; List.Sort is NOT. Two boxes with
        // the same y1 must keep the order suppression left them in, because that order decides which
        // word comes first in a joined field string.
        kept = [.. kept.OrderBy(b => b.Y1).ThenBy(b => b.X1)];

        if (context.Resize)
        {
            MapBack(kept, context);
        }

        if (!NumpyOnly)
        {
            foreach (Box b in kept)
            {
                // TRUNCATION, not rounding — the reference casts to int here, after having rounded
                // half-to-even in MapBack. Two different operations, in that order.
                b.X1 = Math.Truncate(b.X1);
                b.Y1 = Math.Truncate(b.Y1);
                b.X2 = Math.Truncate(b.X2);
                b.Y2 = Math.Truncate(b.Y2);
                b.Label = b.Cls >= 0 && b.Cls < labels.Length
                    ? labels[b.Cls]
                    : "Unsupported class";
            }
        }
        return kept;
    }

    /// <summary>
    /// Undoes the letterbox and the extra padding, then clamps and rounds.
    ///
    /// <para>
    /// The order matters and is the reference's: unpad, unscale, un-extra-pad, clamp negatives to
    /// zero, THEN round half-to-even. Rounding before clamping would turn -0.4 into -0 and then 0,
    /// which happens to agree; rounding before unscaling would not.
    /// </para>
    /// </summary>
    private static void MapBack(List<Box> boxes, Context ctx)
    {
        foreach (Box b in boxes)
        {
            b.X1 = (b.X1 - ctx.PadLetter[0]) / ctx.Ratio - ctx.PadExtra[0];
            b.X2 = (b.X2 - ctx.PadLetter[0]) / ctx.Ratio - ctx.PadExtra[0];
            b.Y1 = (b.Y1 - ctx.PadLetter[1]) / ctx.Ratio - ctx.PadExtra[1];
            b.Y2 = (b.Y2 - ctx.PadLetter[1]) / ctx.Ratio - ctx.PadExtra[1];

            b.X1 = Math.Max(0, b.X1);
            b.Y1 = Math.Max(0, b.Y1);
            b.X2 = Math.Max(0, b.X2);
            b.Y2 = Math.Max(0, b.Y2);
            b.Conf = Math.Max(0, b.Conf);
            for (int j = 0; j < b.Seg.Length; j++)
            {
                // **Clamps the WHOLE row, mask coefficients included.** It reads like an oversight —
                // the intent was surely to clamp coordinates — but the reference does
                // `detect_res[detect_res < 0] = 0` across every column, and those coefficients go
                // straight to the segmentor. A port that clamps only the coordinates gets different
                // masks.
                b.Seg[j] = Math.Max(0, b.Seg[j]);
            }

            b.X1 = PyNum.RoundHalfEven(b.X1);
            b.Y1 = PyNum.RoundHalfEven(b.Y1);
            b.X2 = PyNum.RoundHalfEven(b.X2);
            b.Y2 = PyNum.RoundHalfEven(b.Y2);
            b.Conf = Ops.RoundHalfEven(b.Conf, 3);
        }
    }

    /// <summary>
    /// Non-maximum suppression over a subset of indices.
    ///
    /// <para>
    /// **The tie rule is load-bearing.** The candidates are sorted ASCENDING by score with a stable
    /// sort, and the highest — the LAST element — is picked each round. On a score tie that keeps the
    /// one with the greater original index, which is what <c>np.argsort</c> plus <c>order[-1]</c>
    /// does. Picking the first instead changes which of two equally-confident boxes survives.
    /// </para>
    ///
    /// <para><c>ratio &lt; threshold</c> survives, matching <c>np.where(ratio &lt; threshold)</c>.</para>
    /// </summary>
    private static List<int> Nms(List<int> indices, List<Box> boxes, double threshold)
    {
        var areas = indices.ToDictionary(i => i,
            i => (boxes[i].X2 - boxes[i].X1) * (boxes[i].Y2 - boxes[i].Y1));

        // Stable ascending by score — np.argsort.
        var order = indices.OrderBy(i => boxes[i].Conf).ToList();

        var picked = new List<int>();
        while (order.Count > 0)
        {
            int index = order[^1];
            picked.Add(index);
            order.RemoveAt(order.Count - 1);

            var survivors = new List<int>(order.Count);
            foreach (int j in order)
            {
                double x1 = Math.Max(boxes[index].X1, boxes[j].X1);
                double y1 = Math.Max(boxes[index].Y1, boxes[j].Y1);
                double x2 = Math.Min(boxes[index].X2, boxes[j].X2);
                double y2 = Math.Min(boxes[index].Y2, boxes[j].Y2);
                double inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
                double ratio = inter / (areas[index] + areas[j] - inter);
                if (ratio < threshold)
                {
                    survivors.Add(j);
                }
            }
            order = survivors;
        }
        return picked;
    }

    /// <summary>
    /// Suppression within each class independently.
    ///
    /// <para>
    /// Classes are visited in ASCENDING order, matching <c>np.unique</c>. The resulting index order
    /// feeds the reading-order sort afterwards, so a different visit order can break ties
    /// differently — which is why this sorts the class list rather than iterating a set.
    /// </para>
    /// </summary>
    private static List<int> NmsPerClass(List<Box> boxes, double threshold)
    {
        var classes = boxes.Select(b => b.Cls).Distinct().OrderBy(c => c).ToList();
        var keep = new List<int>();
        foreach (int c in classes)
        {
            var indices = Enumerable.Range(0, boxes.Count).Where(i => boxes[i].Cls == c).ToList();
            keep.AddRange(Nms(indices, boxes, threshold));
        }
        return keep;
    }
}
