using OpenCvSharp;
using RussianDocs.DocumentProcessing.Config;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Inference;
using RussianDocs.DocumentProcessing.Models;
using RussianDocs.DocumentProcessing.Postprocess;

namespace RussianDocs.DocumentProcessing.Modules;

/// <summary>
/// One detected field: its box and the cropped patch. **The patch is owned by the holder.**
/// </summary>
public sealed class Field(Box box, Image patch) : IDisposable
{
    public Box Box { get; } = box;
    public Image Patch { get; } = patch;

    public void Dispose() => Patch.Dispose();
}

public static class Fields
{
    /// <summary>Disposes every field's patch. Safe on a partially built list.</summary>
    public static void CloseAll(IEnumerable<Field>? fields)
    {
        if (fields is null)
        {
            return;
        }
        foreach (Field field in fields)
        {
            field.Dispose();
        }
    }
}

/// <summary>Locates the text fields on a corrected canvas and crops each one.</summary>
public sealed class TextFieldsDetector : IDisposable
{
    private readonly DetectionModel _model;

    public TextFieldsDetector(string root, IReadOnlyDictionary<string, string> paths, Device device,
        int threads)
        => _model = new DetectionModel(
            Path.Combine(ModelPaths.Resolve(root, paths, "TextFieldsDetector"), "ONNX"),
            device, threads);

    /// <summary>
    /// Detects and crops.
    ///
    /// <para>
    /// The crop goes through <see cref="Crop.ClampedCrop"/>, which is not optional: this detector
    /// routinely returns boxes a pixel or two outside the canvas, and a literal translation of the
    /// reference's slice would throw on them.
    /// </para>
    ///
    /// <para>
    /// <paramref name="rotateLicence"/> rotates the <c>Licence_number</c> patch a quarter turn. The
    /// internal passport prints its series and number sideways, so without this the OCR reads a
    /// vertical strip. Only that one field, and only for that document type.
    /// </para>
    ///
    /// <para>
    /// On any failure every patch cropped so far is disposed before the exception leaves — a partial
    /// list of owned Mats that nobody holds is how a leak starts.
    /// </para>
    /// </summary>
    public List<Field> PredictTransform(Image canvas, bool rotateLicence)
    {
        List<Box> boxes = _model.Predict(canvas);
        var fields = new List<Field>(boxes.Count);
        try
        {
            foreach (Box box in boxes)
            {
                Image patch = Crop.ClampedCrop(canvas, (int)box.X1, (int)box.Y1,
                    (int)box.X2, (int)box.Y2);
                if (rotateLicence && box.Label == "Licence_number")
                {
                    Image rotated = Rotate90Ccw(patch);
                    patch.Dispose();
                    patch = rotated;
                }
                fields.Add(new Field(box, patch));
            }
            return fields;
        }
        catch
        {
            Fields.CloseAll(fields);
            throw;
        }
    }

    private static Image Rotate90Ccw(Image src)
    {
        var dst = new Mat();
        Cv2.Rotate(src.Mat, dst, RotateFlags.Rotate90Counterclockwise);
        return Image.Wrap(dst);
    }

    public void Dispose() => _model.Dispose();
}

/// <summary>Splits a field patch into word crops, in reading order.</summary>
public sealed class WordsDetector : IDisposable
{
    private readonly DetectionModel _model;

    public WordsDetector(string root, IReadOnlyDictionary<string, string> paths, Device device,
        int threads)
        => _model = new DetectionModel(
            Path.Combine(ModelPaths.Resolve(root, paths, "WordsDetector"), "ONNX"), device, threads);

    /// <summary>
    /// Sorts word boxes into reading order: cluster into lines by vertical centre proximity (within
    /// half a word height), lines top-to-bottom, words left-to-right inside a line. Port of
    /// <c>WordsDetector._reading_order</c>.
    ///
    /// <para>
    /// A plain x-sort interleaves the lines of a multi-line field — measured on the birth
    /// certificates' Birth_place/ZAGS fields as word salad — so this is a correctness rule, not a
    /// tidiness one. On a single-line field it reproduces the old x-sorted order exactly.
    /// </para>
    ///
    /// <para>
    /// Two things are load-bearing. Every sort is STABLE (<c>OrderBy</c> is, <c>List.Sort</c> is
    /// not), or two words sharing a centre or an x1 would swap. And the running means are updated
    /// per box, in the reference's order: a box joins the FIRST line it fits, and the line's centre
    /// and height are the means over the boxes admitted so far — comparing against the first box
    /// instead would cluster differently on a field whose line drifts.
    /// </para>
    /// </summary>
    public static List<Box> ReadingOrder(List<Box> boxes)
    {
        var lines = new List<(double Cy, double H, List<Box> Boxes)>();

        foreach (Box box in boxes.OrderBy(b => (b.Y1 + b.Y2) / 2))
        {
            double cy = (box.Y1 + box.Y2) / 2, h = box.Y2 - box.Y1;
            bool placed = false;
            for (int i = 0; i < lines.Count; i++)
            {
                if (Math.Abs(cy - lines[i].Cy) < 0.5 * Math.Max(h, lines[i].H))
                {
                    double n = lines[i].Boxes.Count;
                    lines[i].Boxes.Add(box);
                    lines[i] = ((lines[i].Cy * n + cy) / (n + 1), (lines[i].H * n + h) / (n + 1),
                        lines[i].Boxes);
                    placed = true;
                    break;
                }
            }
            if (!placed)
            {
                lines.Add((cy, h, [box]));
            }
        }

        var ordered = new List<Box>(boxes.Count);
        foreach ((_, _, List<Box> lineBoxes) in lines)   // already top-to-bottom
        {
            ordered.AddRange(lineBoxes.OrderBy(b => b.X1));
        }
        return ordered;
    }

    /// <summary>
    /// Word boxes and their crops, in reading order.
    ///
    /// <para>
    /// The boxes are returned REORDERED, not just the crops: that order is what the conformance
    /// <c>words.&lt;field&gt;.bbox</c> stage records and what the OCR loop walks, so the two must agree.
    /// </para>
    ///
    /// <para>
    /// An empty result is normal — the caller falls back to the whole patch, as the reference does.
    /// </para>
    /// </summary>
    public (List<Box> Boxes, List<Image> Words) PredictTransform(Image patch)
    {
        List<Box> boxes = ReadingOrder(_model.Predict(patch));

        var words = new List<Image>(boxes.Count);
        try
        {
            foreach (Box box in boxes)
            {
                // Cut ON the box. Python pads small word boxes by 2 px since 1cc8468, and the ports
                // deliberately do NOT follow yet: the words detector is being retrained with the
                // margin inside the labelled box, which may remove the compensation altogether. The
                // ports are synced to the FINAL Python behaviour in one pass before the goldens are
                // regenerated.
                words.Add(Crop.ClampedCrop(patch, (int)box.X1, (int)box.Y1, (int)box.X2, (int)box.Y2));
            }
            return (boxes, words);
        }
        catch
        {
            // Release what was already cropped: a sibling failing must not leave these orphaned.
            foreach (Image word in words)
            {
                word.Dispose();
            }
            throw;
        }
    }

    public void Dispose() => _model.Dispose();
}
