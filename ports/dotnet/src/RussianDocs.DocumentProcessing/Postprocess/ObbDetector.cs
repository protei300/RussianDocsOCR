using OpenCvSharp;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Postprocess;

/// <summary>
/// One oriented detection, on the image the detector read. <c>[cx, cy, w, h, angle, conf, cls, label]</c> in the
/// reference. The angle is in radians (ultralytics convention, clockwise in image space).
/// </summary>
public sealed class ObbBox
{
    public double Cx, Cy, W, H, Angle, Conf;
    public int Cls;
    public string Label = "";
}

public sealed record ObbResult(List<ObbBox> Boxes) : IResult;

/// <summary>
/// Decodes the raw ultralytics OBB head and suppresses overlaps. Port of <c>YOLOOBBDetectorPostprocessing</c>.
///
/// <para>
/// **Float32 where the reference is float32.** The head's output is float32 and numpy keeps a float32 scalar
/// float32 through arithmetic with a Python float (NEP 50), so the confidence threshold, the letterbox undo and
/// the rotated IoU are all computed in float32 here. A double would differ in the seventh digit, which would
/// decide a suppression only on a pair that sits on the threshold, but the cost of copying the reference is nil.
/// </para>
///
/// <para>
/// The NMS is greedy and rotated, WITHOUT regard to class (two lines of different classes suppress each other),
/// and the survivors are put in reading order: rows by <c>round(cy / 10)</c>, then by <c>cx</c> - a stable sort,
/// as Python's is.
/// </para>
/// </summary>
public sealed class YoloObbDetector(string[] labels, double iou, double cls) : IPostprocessor
{
    /// <summary><c>np.degrees</c> on a float32: the factor is a float32 constant.</summary>
    private static readonly float DegreesPerRadian = 180f / 3.14159265358979323846f;

    public IResult Apply(NdArray output, Context context) => new ObbResult(Decode(output, context));

    public List<ObbBox> Decode(NdArray output, Context context)
    {
        if (output.Length == 0)
        {
            return [];
        }
        // np.squeeze: every dimension of size 1 goes.
        int[] shape = [.. output.Shape.Where(d => d != 1)];
        if (shape.Length != 2)
        {
            throw new InvalidDataException(
                $"postprocess: OBB detector expects [features, anchors] or its transpose, got " +
                $"{NdArray.Describe(output.Shape)}");
        }
        float[] data = output.AsFloat32().ToArray();

        // `if vector.shape[0] < vector.shape[1]: vector = vector.T` - the anchors come last in the export.
        bool anchorsLast = shape[0] < shape[1];
        int anchors = anchorsLast ? shape[1] : shape[0];
        int features = anchorsLast ? shape[0] : shape[1];
        int nc = labels.Length;
        if (features < 4 + nc + 1)
        {
            throw new InvalidDataException(
                $"postprocess: OBB row of {features} values is too short for {nc} classes");
        }
        float At(int anchor, int feature) =>
            anchorsLast ? data[feature * anchors + anchor] : data[anchor * features + feature];

        float threshold = (float)cls;
        var cx = new List<float>();
        var cy = new List<float>();
        var w = new List<float>();
        var h = new List<float>();
        var angle = new List<float>();
        var conf = new List<float>();
        var cid = new List<int>();
        for (int a = 0; a < anchors; a++)
        {
            // det.max(axis=1) and det.argmax(axis=1): the first maximum wins
            int best = 0;
            float bestScore = At(a, 4);
            for (int c = 1; c < nc; c++)
            {
                float score = At(a, 4 + c);
                if (score > bestScore)
                {
                    best = c;
                    bestScore = score;
                }
            }
            if (!(bestScore > threshold))
            {
                continue;
            }
            cx.Add(At(a, 0));
            cy.Add(At(a, 1));
            w.Add(At(a, 2));
            h.Add(At(a, 3));
            angle.Add(At(a, 4 + nc));
            conf.Add(bestScore);
            cid.Add(best);
        }
        if (conf.Count == 0)
        {
            return [];
        }

        // argsort(-conf): highest first; ties keep the lower index (a stable sort - numpy's quicksort leaves
        // ties unspecified, and equal float32 scores among survivors are not expected)
        List<int> order = [.. Enumerable.Range(0, conf.Count).OrderByDescending(i => conf[i])];
        var kept = new List<int>();
        float iouThreshold = (float)iou;
        while (order.Count > 0)
        {
            int i = order[0];
            order.RemoveAt(0);
            kept.Add(i);
            var survivors = new List<int>(order.Count);
            foreach (int j in order)
            {
                if (RotatedIou(cx[i], cy[i], w[i], h[i], angle[i], cx[j], cy[j], w[j], h[j], angle[j])
                    < iouThreshold)
                {
                    survivors.Add(j);
                }
            }
            order = survivors;
        }

        var results = new List<ObbBox>(kept.Count);
        foreach (int i in kept)
        {
            // Back from the letterboxed space to the image: unpad, unscale, un-extra-pad - in float32, each
            // step as numpy does it (`(cx - pad) / ratio - extra`).
            float ratio = (float)context.Ratio;
            float x = (cx[i] - (float)context.PadLetter[0]) / ratio - context.PadExtra[0];
            float y = (cy[i] - (float)context.PadLetter[1]) / ratio - context.PadExtra[1];
            float bw = w[i] / ratio;
            float bh = h[i] / ratio;
            results.Add(new ObbBox
            {
                Cx = x, Cy = y, W = bw, H = bh,
                Angle = angle[i], Conf = conf[i], Cls = cid[i],
                Label = labels.Length > 0 ? labels[cid[i]] : cid[i].ToString(),
            });
        }

        // reading order: top to bottom, then left to right. OrderBy is stable, as list.sort is.
        return [.. results.OrderBy(r => PyNum.RoundHalfEven(r.Cy / 10)).ThenBy(r => r.Cx)];
    }

    /// <summary>
    /// Rotated-rectangle IoU through OpenCV's intersection (<c>rotated_iou</c>). The areas and the division are
    /// float32, the intersection area OpenCV's double.
    /// </summary>
    public static float RotatedIou(float ax, float ay, float aw, float ah, float aAngle,
        float bx, float by, float bw, float bh, float bAngle)
    {
        var ra = new RotatedRect(new Point2f(ax, ay), new Size2f(aw, ah), aAngle * DegreesPerRadian);
        var rb = new RotatedRect(new Point2f(bx, by), new Size2f(bw, bh), bAngle * DegreesPerRadian);
        Cv2.RotatedRectangleIntersection(ra, rb, out Point2f[] region);
        if (region is null || region.Length == 0)
        {
            return 0f;
        }
        float inter = (float)Cv2.ContourArea(region);
        float areaA = aw * ah;
        float areaB = bw * bh;
        float union = areaA + areaB;
        union -= inter;
        return union > 0f ? inter / union : 0f;
    }
}
