using System.Text.Json.Serialization;

namespace RussianDocs.DocumentProcessing.ViewModel;

/// <summary>
/// The view model — the detail response the SPA reads.
///
/// <para>
/// **Every JSON name is written by hand and nothing is omitted.** No naming policy, and
/// <c>JsonIgnoreCondition.Never</c> on the serialiser: the SPA reads about sixty named fields, and a
/// missing key is a real defect rather than a tolerance question — it makes a page render blank. A
/// policy that gets fifty-nine names right is worse than none, because the one it misses looks like a
/// typo somewhere else entirely.
/// </para>
///
/// <para>
/// Nullable properties are nullable ON PURPOSE. "Absent" must serialise as <c>null</c>, not as 0 or
/// "" — the SPA distinguishes a field that was not read from one that read as empty.
/// </para>
///
/// <para>
/// Exactly FOURTEEN top-level keys when debug is off. The count is asserted by the harness, because
/// an extra key is as much a contract break as a missing one.
/// </para>
/// </summary>
public sealed class Payload
{
    [JsonPropertyName("doc_type")] public string? DocType { get; set; }
    [JsonPropertyName("doc_type_base")] public string? DocTypeBase { get; set; }
    [JsonPropertyName("doc_type_era")] public string? DocTypeEra { get; set; }
    [JsonPropertyName("recognised")] public bool Recognised { get; set; }
    [JsonPropertyName("device")] public string? Device { get; set; }
    [JsonPropertyName("canvas")] public Canvas Canvas { get; set; } = new();
    [JsonPropertyName("coord_space")] public string CoordSpace { get; set; } = "canvas";
    [JsonPropertyName("coord_space_note")] public string CoordSpaceNote { get; set; } = "";
    [JsonPropertyName("boxes")] public List<Box> Boxes { get; set; } = [];
    [JsonPropertyName("fields")] public List<Field> Fields { get; set; } = [];
    [JsonPropertyName("ocr")] public Dictionary<string, string> Ocr { get; set; } = [];
    [JsonPropertyName("quality")] public Dictionary<string, object> Quality { get; set; } = [];
    [JsonPropertyName("timings")] public Dictionary<string, double> Timings { get; set; } = [];

    /// <summary>Null for every type except INTPASSPORTADDR, which this port does not implement.</summary>
    [JsonPropertyName("address")] public Address? Address { get; set; }

    /// <summary>
    /// Only present under <c>?include=debug</c> — the one key that IS omitted when absent, because
    /// the reference omits it rather than sending null.
    /// </summary>
    [JsonPropertyName("debug")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Debug? Debug { get; set; }
}

public sealed class Canvas
{
    [JsonPropertyName("width")] public int? Width { get; set; }
    [JsonPropertyName("height")] public int? Height { get; set; }

    /// <summary>
    /// True when the pipeline short-circuited and there is no corrected canvas.
    ///
    /// <para>
    /// A separate flag rather than inferring it from null dimensions, because the UI must be able to
    /// say "this is the original upload" rather than silently drawing boxes on the wrong image.
    /// </para>
    /// </summary>
    [JsonPropertyName("is_fallback")] public bool IsFallback { get; set; }
}

public sealed class Box
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("label")] public string Label { get; set; } = "";
    [JsonPropertyName("display")] public string Display { get; set; } = "";

    /// <summary><c>"text"</c> or <c>"visual"</c> — Face and Signature are the visual ones.</summary>
    [JsonPropertyName("kind")] public string Kind { get; set; } = "text";

    [JsonPropertyName("x1")] public int? X1 { get; set; }
    [JsonPropertyName("y1")] public int? Y1 { get; set; }
    [JsonPropertyName("x2")] public int? X2 { get; set; }
    [JsonPropertyName("y2")] public int? Y2 { get; set; }
    [JsonPropertyName("conf")] public double? Conf { get; set; }
    [JsonPropertyName("cls")] public int? Cls { get; set; }

    /// <summary>
    /// The field's text, on the box that OWNS it.
    ///
    /// <para>
    /// A field can be detected more than once — the internal passport prints its number twice — and
    /// the library discards the per-detection text, so only the highest-confidence box carries the
    /// value. The others get <c>ambiguous: true</c> instead.
    /// </para>
    /// </summary>
    [JsonPropertyName("text")] public string? Text { get; set; }

    /// <summary>True on a duplicate detection of an OCR'd field that does NOT own the text.</summary>
    [JsonPropertyName("ambiguous")] public bool Ambiguous { get; set; }
}

public sealed class Field
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("display")] public string Display { get; set; } = "";
    [JsonPropertyName("value")] public string? Value { get; set; }
    [JsonPropertyName("script")] public string Script { get; set; } = "ru";
    [JsonPropertyName("conf")] public double? Conf { get; set; }

    /// <summary>
    /// Every box belonging to this field.
    ///
    /// <para>
    /// An ARRAY, not one id: split fields produce several boxes and duplicated fields produce more.
    /// This is what lets the UI highlight all of a field's boxes at once — linking by label string
    /// would be ambiguous exactly where it matters.
    /// </para>
    /// </summary>
    [JsonPropertyName("box_ids")] public List<string> BoxIds { get; set; } = [];
}

public sealed class Debug
{
    [JsonPropertyName("doc_outline")] public DocOutline DocOutline { get; set; } = new();
}

public sealed class DocOutline
{
    /// <summary>
    /// Always <c>"prewarp"</c>.
    ///
    /// <para>
    /// The document contour is in the space BEFORE the perspective correction, so it must not be
    /// drawn on the canvas. Tagging it explicitly is the only thing stopping a UI from doing so.
    /// </para>
    /// </summary>
    [JsonPropertyName("coord_space")] public string CoordSpace { get; set; } = "prewarp";

    [JsonPropertyName("polygon")] public List<List<int[]>>? Polygon { get; set; }
}

public sealed class Address
{
    [JsonPropertyName("aligned")] public bool Aligned { get; set; }
    [JsonPropertyName("lines")] public List<AddressLine> Lines { get; set; } = [];
}

/// <summary>
/// One address line. Declared but never produced by this port.
///
/// <para>
/// The type exists because the contract includes it and an integrator needs the shape; INTPASSPORTADDR
/// is blocked on an anonymised sample, not on code. An omitted type reads as an oversight that the
/// next port would invent differently.
/// </para>
/// </summary>
public sealed class AddressLine
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("kind")] public string? Kind { get; set; }
    [JsonPropertyName("text")] public string? Text { get; set; }
    [JsonPropertyName("p_handwritten")] public double? PHandwritten { get; set; }
    [JsonPropertyName("obbox")] public Obbox? Obbox { get; set; }
}

public sealed class Obbox
{
    [JsonPropertyName("cx")] public double? Cx { get; set; }
    [JsonPropertyName("cy")] public double? Cy { get; set; }
    [JsonPropertyName("w")] public double? W { get; set; }
    [JsonPropertyName("h")] public double? H { get; set; }
    [JsonPropertyName("angle_rad")] public double? AngleRad { get; set; }
    [JsonPropertyName("conf")] public double? Conf { get; set; }
    [JsonPropertyName("label")] public string? Label { get; set; }
}
