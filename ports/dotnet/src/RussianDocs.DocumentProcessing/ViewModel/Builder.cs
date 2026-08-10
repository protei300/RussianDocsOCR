using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.DocumentProcessing.Postprocess;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.ViewModel;

/// <summary>
/// Everything the builder needs, as plain data.
///
/// <para>
/// Takes the canvas WIDTH and HEIGHT rather than the image itself. That is deliberate: it keeps the
/// one type whose whole purpose is to be pure and testable from a literal free of any ownership
/// question — the property the reference's own docstring calls intentional.
/// </para>
/// </summary>
public sealed record Input
{
    public string DocType { get; init; } = "NONE";
    public string Device { get; init; } = "cpu";
    public int CanvasW { get; init; }
    public int CanvasH { get; init; }
    public bool CanvasMissing { get; init; }
    public List<Box2>? Boxes { get; init; }
    public Dictionary<string, string>? Ocr { get; init; }
    public Dictionary<string, object>? Quality { get; init; }
    public Dictionary<string, double>? Timings { get; init; }
    public List<Point[]>? Segments { get; init; }
}

/// <summary>Alias so this file does not shadow the view model's own <c>Box</c>.</summary>
public sealed record Box2(double X1, double Y1, double X2, double Y2, double Conf, int Cls,
    string Label);

public static class Builder
{
    public const int FloatPrecision = 4;

    private const string CoordSpaceNote =
        "Box coordinates are in canvas pixel space and match the canvas image exactly. " +
        "They cannot be mapped onto the original upload: the library does not retain the deskew angle.";

    public static Payload Build(Input input, bool includeDebug)
    {
        Dictionary<string, string> ocr = input.Ocr ?? [];
        List<Box> boxes = BuildBoxes(input.Boxes ?? [], ocr);

        var canvas = new Canvas();
        if (input.CanvasMissing)
        {
            canvas.IsFallback = true;
        }
        else
        {
            canvas.Width = input.CanvasW;
            canvas.Height = input.CanvasH;
        }

        string b = Labels.BaseDocType(input.DocType);
        var payload = new Payload
        {
            DocType = input.DocType,
            DocTypeBase = string.IsNullOrEmpty(b) ? null : b,
            DocTypeEra = Labels.DocTypeEra(input.DocType),
            // An unrecognised document is not an error — the SPA renders it as a legitimate state —
            // so `recognised` is a flag rather than an exception.
            Recognised = input.DocType.Length > 0 && input.DocType != "NONE",
            Device = input.Device,
            Canvas = canvas,
            CoordSpace = "canvas",
            CoordSpaceNote = CoordSpaceNote,
            Boxes = boxes,
            Fields = BuildFields(input.DocType, ocr, boxes),
            Ocr = ocr,
            Quality = input.Quality ?? [],
            Timings = input.Timings ?? [],
            Address = null,
        };

        if (includeDebug)
        {
            payload.Debug = new Debug
            {
                DocOutline = new DocOutline
                {
                    CoordSpace = "prewarp",
                    Polygon = PolygonOf(input.Segments),
                },
            };
        }
        return payload;
    }

    /// <summary>
    /// Turns detector boxes into view-model boxes, deciding which one owns each field's text.
    ///
    /// <para>
    /// The owner is the HIGHEST-CONFIDENCE detection of a label, chosen with strict <c>&gt;</c> so a
    /// tie keeps the earliest. Every other detection of an OCR'd label is marked
    /// <c>ambiguous</c> — the per-detection text genuinely cannot be recovered from the library's
    /// output, so saying so is more honest than attaching the same string to both.
    /// </para>
    /// </summary>
    private static List<Box> BuildBoxes(List<Box2> raw, Dictionary<string, string> ocr)
    {
        if (raw.Count == 0)
        {
            return [];
        }

        var bestByLabel = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < raw.Count; i++)
        {
            if (!bestByLabel.TryGetValue(raw[i].Label, out int previous)
                || raw[i].Conf > raw[previous].Conf)
            {
                bestByLabel[raw[i].Label] = i;
            }
        }

        var boxes = new List<Box>(raw.Count);
        for (int i = 0; i < raw.Count; i++)
        {
            Box2 b = raw[i];
            bool ownsText = bestByLabel[b.Label] == i;
            bool inOcr = ocr.ContainsKey(b.Label);

            boxes.Add(new Box
            {
                // Positional ids, so the field-to-box links are stable within one response.
                Id = $"b{i}",
                Label = b.Label,
                Display = Labels.FieldDisplay(b.Label),
                Kind = Labels.IsNonText(b.Label) ? "visual" : "text",
                X1 = (int)b.X1,
                Y1 = (int)b.Y1,
                X2 = (int)b.X2,
                Y2 = (int)b.Y2,
                Conf = Round(b.Conf),
                Cls = b.Cls,
                Text = ownsText && ocr.TryGetValue(b.Label, out string? text) ? text : null,
                Ambiguous = inOcr && !ownsText,
            });
        }
        return boxes;
    }

    /// <summary>
    /// Builds the ordered field list, linking each to its boxes.
    ///
    /// <para>
    /// An ARRAY rather than a dictionary, which removes three problems at once: the link to boxes, the
    /// reading order (a dictionary has none, and insertion order is not reading order), and the
    /// font choice, which travels as <c>script</c>.
    /// </para>
    /// </summary>
    private static List<Field> BuildFields(string docType, Dictionary<string, string> ocr,
        List<Box> boxes)
    {
        var byLabel = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var confByLabel = new Dictionary<string, double?>(StringComparer.Ordinal);
        foreach (Box box in boxes)
        {
            if (!byLabel.TryGetValue(box.Label, out List<string>? ids))
            {
                ids = [];
                byLabel[box.Label] = ids;
            }
            ids.Add(box.Id);

            // The confidence reported for a field is the OWNING box's, not the maximum or the mean:
            // it is the confidence of the detection whose text is being shown.
            if (box.Text is not null)
            {
                confByLabel[box.Label] = box.Conf;
            }
        }

        // Sorted before ordering, so the unknown tail is deterministic across languages — see
        // Labels.OrderFields.
        List<string> ordered = Labels.OrderFields(docType,
            ocr.Keys.OrderBy(k => k, StringComparer.Ordinal));

        var fields = new List<Field>(ordered.Count);
        foreach (string name in ordered)
        {
            fields.Add(new Field
            {
                Name = name,
                Display = Labels.FieldDisplay(name),
                Value = ocr.TryGetValue(name, out string? value) ? value : null,
                Script = Labels.FieldScript(name),
                Conf = confByLabel.TryGetValue(name, out double? conf) ? conf : null,
                BoxIds = byLabel.TryGetValue(name, out List<string>? ids) ? ids : [],
            });
        }
        return fields;
    }

    private static List<List<int[]>>? PolygonOf(List<Point[]>? segments) =>
        segments?.Select(contour => contour.Select(p => new[] { (int)p.X, (int)p.Y }).ToList())
                .ToList();

    /// <summary>
    /// Rounds a wire float to four places, half to even.
    ///
    /// <para>
    /// **Rounding on the server is what makes the goldens comparable at all.** Left unrounded, the
    /// text of a float differs between languages in the seventeenth digit and every port's JSON
    /// diverges from the reference's for no semantic reason. NaN and infinity become null rather than
    /// unparseable JSON tokens.
    /// </para>
    /// </summary>
    public static double? Round(double value) =>
        double.IsNaN(value) || double.IsInfinity(value)
            ? null
            : Ops.RoundHalfEven(value, FloatPrecision);
}
