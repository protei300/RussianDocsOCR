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

/// <summary>Splits a field patch into word crops, left to right.</summary>
public sealed class WordsDetector : IDisposable
{
    private readonly DetectionModel _model;

    public WordsDetector(string root, IReadOnlyDictionary<string, string> paths, Device device,
        int threads)
        => _model = new DetectionModel(
            Path.Combine(ModelPaths.Resolve(root, paths, "WordsDetector"), "ONNX"), device, threads);

    /// <summary>
    /// Word boxes and their crops, left to right.
    ///
    /// <para>
    /// **The ordering is the one trap here.** The reference sorts with
    /// <c>bbox.sort(key=lambda x: x[0])</c>, and Python's sort is STABLE — so words keep the
    /// reading-order sort the detector already applied whenever their x1 ties. LINQ's
    /// <c>OrderBy</c> is stable and <c>List.Sort</c> is not; two words sharing an x1 would otherwise
    /// swap and reorder two tokens of the joined field string.
    /// </para>
    ///
    /// <para>
    /// An empty result is normal — the caller falls back to the whole patch, as the reference does.
    /// </para>
    /// </summary>
    public (List<Box> Boxes, List<Image> Words) PredictTransform(Image patch)
    {
        List<Box> boxes = _model.Predict(patch);
        boxes = [.. boxes.OrderBy(b => b.X1)];

        var words = new List<Image>(boxes.Count);
        try
        {
            foreach (Box box in boxes)
            {
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
