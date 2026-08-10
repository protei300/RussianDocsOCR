using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace RussianDocs.Service.Model;

public static class DocumentStatus
{
    public const string Queued = "queued";
    public const string Processing = "processing";
    public const string Done = "done";
    public const string Failed = "failed";

    public static readonly HashSet<string> Valid = new(StringComparer.Ordinal)
        { Queued, Processing, Done, Failed };
}

/// <summary>
/// Serialises a <see cref="DateTime"/> the way the record format requires, or <c>null</c>.
///
/// <para>
/// The format is fixed and shared: UTC, up to nine fractional digits, trailing <c>Z</c>. **The two
/// implementations share a data directory** — the Go and Python services can be pointed at the same
/// one, and the seed corpus is written by Python — so a record written by either must be readable by
/// the other. That is also why parsing is lenient about the exact number of fractional digits while
/// writing is not.
/// </para>
/// </summary>
public sealed class NullableUtcConverter : JsonConverter<DateTime?>
{
    /// <summary>
    /// UTC, up to seven fractional digits, trailing Z.
    ///
    /// <para>
    /// **Two traps in one nine-character format string, and both throw rather than misformat.**
    /// First, <c>T</c> and <c>Z</c> must be QUOTED: in a .NET custom date format an unquoted letter
    /// is read as a specifier, and neither is one. Second, <b>SEVEN <c>F</c>s is the maximum</b> —
    /// a <see cref="DateTime"/> tick is 100 ns, so the nine digits the record format permits cannot
    /// all be expressed and asking for them is a <see cref="FormatException"/>.
    /// </para>
    ///
    /// <para>
    /// Both were found the same way and cost the same thing: every record failed to persist with
    /// "Input string was not in a correct format" while the in-memory index looked perfectly fine —
    /// a service that appeared to work right up until it restarted. Go's
    /// <c>2006-01-02T15:04:05.999999999Z</c> reference layout has neither trap, which is exactly
    /// why this line needs the comment. Seven digits is a SUBSET of what the format allows, so
    /// anything written here still parses on the Python and Go sides.
    /// </para>
    /// </summary>
    private const string Pattern = "yyyy-MM-dd'T'HH:mm:ss.FFFFFFF'Z'";

    /// <summary>
    /// The one place that spells a timestamp, so every projection agrees.
    ///
    /// <para>
    /// Public because a <c>[JsonConverter]</c> attribute reaches properties only — a timestamp placed
    /// inside a dictionary of object needs this explicitly.
    /// </para>
    /// </summary>
    public static string? Format(DateTime? value) => value?.ToUniversalTime()
        .ToString(Pattern, CultureInfo.InvariantCulture);

    public override DateTime? Read(ref Utf8JsonReader reader, Type type,
        JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Null)
        {
            return null;
        }
        string? text = reader.GetString();
        return string.IsNullOrEmpty(text)
            ? null
            : DateTime.Parse(text, CultureInfo.InvariantCulture,
                DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal);
    }

    public override void Write(Utf8JsonWriter writer, DateTime? value,
        JsonSerializerOptions options)
    {
        if (Format(value) is not { } text)
        {
            writer.WriteNullValue();
            return;
        }
        writer.WriteStringValue(text);
    }
}

/// <summary>
/// One uploaded document and everything known about it.
///
/// <para>
/// **The JSON names are the on-disk record format AND the future SQL column names**, which is why
/// every one is written by hand: three languages have three default naming policies, and a record
/// written by one implementation has to be readable by the others.
/// </para>
/// </summary>
public sealed class Document
{
    [JsonPropertyName("id")] public int Id { get; set; }

    /// <summary>
    /// Sanitised, and for DISPLAY ONLY — never used as a path.
    ///
    /// <para>
    /// On disk the file is always <c>original.&lt;ext&gt;</c>, which is what makes a hostile filename
    /// harmless rather than a directory-traversal vector.
    /// </para>
    /// </summary>
    [JsonPropertyName("filename")] public string Filename { get; set; } = "";

    [JsonPropertyName("content_type")] public string ContentType { get; set; } = "";
    [JsonPropertyName("size_bytes")] public long SizeBytes { get; set; }
    [JsonPropertyName("status")] public string Status { get; set; } = DocumentStatus.Queued;

    [JsonPropertyName("doc_type")] public string? DocType { get; set; }
    [JsonPropertyName("doc_conf")] public double? DocConf { get; set; }
    [JsonPropertyName("recognised")] public bool Recognised { get; set; }
    [JsonPropertyName("field_count")] public int FieldCount { get; set; }

    /// <summary>
    /// Denormalised quality verdicts, so the list page can show them without loading each result blob.
    ///
    /// <para>
    /// Values are whatever the library reports — currently <c>good</c>/<c>bad</c> for glare and blur
    /// but <c>REAL</c>/<c>FAKE</c> for the spoofing checks. Clients must NOT assume one vocabulary;
    /// the inconsistency is in the library and the wire carries it.
    /// </para>
    /// </summary>
    [JsonPropertyName("quality")] public Dictionary<string, object> Quality { get; set; } = [];

    [JsonPropertyName("device")] public string? Device { get; set; }
    [JsonPropertyName("processing_ms")] public int? ProcessingMs { get; set; }

    /// <summary>Human-readable failure text. May be in Russian even though the UI is English.</summary>
    [JsonPropertyName("error")] public string? Error { get; set; }

    /// <summary>
    /// A machine-readable failure code beside the message.
    ///
    /// <para>
    /// Present precisely because the message may arrive in Russian from the library while the UI is
    /// English — a client that needs to branch on the failure cannot parse prose.
    /// </para>
    /// </summary>
    [JsonPropertyName("error_code")] public string? ErrorCode { get; set; }

    [JsonPropertyName("retry_count")] public int RetryCount { get; set; }

    [JsonPropertyName("original_ext")] public string OriginalExt { get; set; } = "";
    [JsonPropertyName("original_w")] public int? OriginalW { get; set; }
    [JsonPropertyName("original_h")] public int? OriginalH { get; set; }
    [JsonPropertyName("canvas_w")] public int? CanvasW { get; set; }
    [JsonPropertyName("canvas_h")] public int? CanvasH { get; set; }
    [JsonPropertyName("has_canvas")] public bool HasCanvas { get; set; }

    /// <summary>
    /// Pre-computed lowercase haystack: filename, document type and every OCR value.
    ///
    /// <para>
    /// Denormalised so the list filter never parses a result blob. In SQL this becomes an indexable
    /// column, which is the whole point of computing it at write time.
    /// </para>
    /// </summary>
    [JsonPropertyName("search_text")] public string SearchText { get; set; } = "";

    [JsonPropertyName("created_at")]
    [JsonConverter(typeof(NullableUtcConverter))]
    public DateTime? CreatedAt { get; set; }

    [JsonPropertyName("started_at")]
    [JsonConverter(typeof(NullableUtcConverter))]
    public DateTime? StartedAt { get; set; }

    [JsonPropertyName("finished_at")]
    [JsonConverter(typeof(NullableUtcConverter))]
    public DateTime? FinishedAt { get; set; }

    [JsonPropertyName("updated_at")]
    [JsonConverter(typeof(NullableUtcConverter))]
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// The full recognition view model.
    ///
    /// <para>
    /// Kept OUT of the in-memory index — it can be 100 KB of boxes per document — and loaded lazily
    /// by the repository's get-by-id. <see cref="JsonIgnoreAttribute"/> because it lives in its own
    /// file, not in the record.
    /// </para>
    /// </summary>
    [JsonIgnore] public JsonNode? Result { get; set; }

    public static DateTime UtcNow() => DateTime.UtcNow;

    public static Document New(int id, string filename, string contentType, long sizeBytes,
        string ext)
    {
        DateTime now = UtcNow();
        return new Document
        {
            Id = id,
            Filename = filename,
            ContentType = contentType,
            SizeBytes = sizeBytes,
            OriginalExt = ext,
            Status = DocumentStatus.Queued,
            CreatedAt = now,
            UpdatedAt = now,
        };
    }

    /// <summary>
    /// An independent copy.
    ///
    /// <para>
    /// The store hands out copies rather than references, so a caller mutating what it read cannot
    /// corrupt the index. That is also how the reference's repositories behave — <c>update()</c>
    /// returns a NEW record and the caller rebinds — and the pattern is worth keeping because it is
    /// what a SQL-backed implementation would do naturally.
    /// </para>
    /// </summary>
    public Document Clone() => new()
    {
        Id = Id, Filename = Filename, ContentType = ContentType, SizeBytes = SizeBytes,
        Status = Status, DocType = DocType, DocConf = DocConf, Recognised = Recognised,
        FieldCount = FieldCount,
        Quality = new Dictionary<string, object>(Quality, StringComparer.Ordinal),
        Device = Device, ProcessingMs = ProcessingMs, Error = Error, ErrorCode = ErrorCode,
        RetryCount = RetryCount, OriginalExt = OriginalExt, OriginalW = OriginalW,
        OriginalH = OriginalH, CanvasW = CanvasW, CanvasH = CanvasH, HasCanvas = HasCanvas,
        SearchText = SearchText, CreatedAt = CreatedAt, StartedAt = StartedAt,
        FinishedAt = FinishedAt, UpdatedAt = UpdatedAt,
        // Result is intentionally NOT deep-copied: it is loaded lazily, treated as immutable once
        // read, and copying 100 KB of boxes onto every list row would be pure waste.
        Result = Result,
    };
}

/// <summary>An API key. Only the HASH is stored; the plaintext is shown once, at creation.</summary>
public sealed class ApiKey
{
    [JsonPropertyName("id")] public int Id { get; set; }
    [JsonPropertyName("label")] public string Label { get; set; } = "";

    /// <summary>sha256 of the key. **Never the key itself.**</summary>
    [JsonPropertyName("key_hash")] public string KeyHash { get; set; } = "";

    /// <summary>A short display prefix, so a key can be recognised without being revealed.</summary>
    [JsonPropertyName("prefix")] public string Prefix { get; set; } = "";

    /// <summary>
    /// The key from the environment, which cannot be deleted.
    ///
    /// <para>
    /// DELETE on it answers 409. Without that rule the service could be left with no way in at all.
    /// </para>
    /// </summary>
    [JsonPropertyName("is_default")] public bool IsDefault { get; set; }

    [JsonPropertyName("created_at")]
    [JsonConverter(typeof(NullableUtcConverter))]
    public DateTime? CreatedAt { get; set; }

    [JsonPropertyName("last_used_at")]
    [JsonConverter(typeof(NullableUtcConverter))]
    public DateTime? LastUsedAt { get; set; }

    public ApiKey Clone() => new()
    {
        Id = Id, Label = Label, KeyHash = KeyHash, Prefix = Prefix, IsDefault = IsDefault,
        CreatedAt = CreatedAt, LastUsedAt = LastUsedAt,
    };

    /// <summary>
    /// What the UI may see. **Never the hash.**
    ///
    /// <para>
    /// A separate shape rather than <see cref="JsonIgnoreAttribute"/> on the hash, because the same
    /// type is persisted WITH the hash — one type with two audiences needs two explicit projections,
    /// not one attribute that has to be right in both directions.
    /// </para>
    /// </summary>
    public Dictionary<string, object?> Public() => new(StringComparer.Ordinal)
    {
        ["id"] = Id,
        ["label"] = Label,
        ["prefix"] = Prefix,
        ["masked"] = Prefix + "••••••••",
        ["is_default"] = IsDefault,
        // Formatted here rather than left as DateTime?, because a [JsonConverter] attribute on a
        // property does NOT apply to the same value inside a dictionary of object — the timestamps
        // would silently take the framework's default spelling and stop matching the record format.
        ["created_at"] = NullableUtcConverter.Format(CreatedAt),
        ["last_used_at"] = NullableUtcConverter.Format(LastUsedAt),
    };
}
