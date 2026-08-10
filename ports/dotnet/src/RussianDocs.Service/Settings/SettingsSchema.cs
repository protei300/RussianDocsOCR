using System.Globalization;
using System.Text.Json.Serialization;

namespace RussianDocs.Service.Settings;

/// <summary>
/// The types a setting can have.
///
/// <para>
/// String constants, not an enum, because they go to the UI which switches on them to pick a widget.
/// </para>
/// </summary>
public static class SettingType
{
    public const string Bool = "bool";
    public const string Int = "int";
    public const string Float = "float";
    public const string Choice = "choice";
    public const string Str = "str";
}

/// <summary>
/// One setting's description.
///
/// <para>
/// Nullable numeric bounds, so "no minimum" is distinguishable from "minimum zero" — a distinction
/// that matters for <c>docconf</c>, whose valid range genuinely starts at 0.
/// </para>
/// </summary>
public sealed record SettingDef
{
    [JsonPropertyName("key")] public required string Key { get; init; }
    [JsonPropertyName("type")] public required string Type { get; init; }
    [JsonPropertyName("default")] public required string Default { get; init; }
    [JsonPropertyName("label")] public required string Label { get; init; }
    [JsonPropertyName("description")] public string Description { get; init; } = "";
    [JsonPropertyName("group")] public required string Group { get; init; }
    [JsonPropertyName("min_value")] public double? MinValue { get; init; }
    [JsonPropertyName("max_value")] public double? MaxValue { get; init; }
    [JsonPropertyName("choices")] public string[]? Choices { get; init; }

    /// <summary>
    /// Marks a setting baked into the pipeline's construction.
    ///
    /// <para>
    /// Changing <c>ocr_mode</c> in the UI cannot affect a pipeline that is already built, and
    /// silently pretending otherwise is worse than saying so.
    /// </para>
    /// </summary>
    [JsonPropertyName("restart_required")] public bool RestartRequired { get; init; }
}

/// <summary>Thrown for a rejected value. The message reaches the UI, so it names the bound.</summary>
public sealed class SettingValidationException(string message) : Exception(message);

/// <summary>
/// The server-owned schema for runtime-tunable settings.
///
/// <para>
/// The server describes its own knobs — type, bounds, choices, help text, group — and the UI renders
/// itself from that. The alternative, a hand-written form, means every new pipeline knob is a
/// frontend change and the defaults end up duplicated on both sides, where they drift.
/// </para>
///
/// <para>
/// Values are STORED AS STRINGS (the store is JSON; SQL would use a key/value table). Coercion and
/// validation happen here, in one place, on the way IN.
/// </para>
///
/// <para>Port of <c>service/core/settings_schema.py</c>.</para>
/// </summary>
public static class SettingsSchema
{
    /// <summary>
    /// The ordered list. **ORDER IS THE UI ORDER** — the settings page renders it as given, so this
    /// is grouping and sequencing, not just a registry.
    /// </summary>
    public static readonly SettingDef[] All =
    [
        new()
        {
            Key = "compute_device", Type = SettingType.Choice, Default = "auto",
            Label = "Compute device",
            Description = "GPU is used only when onnxruntime reports a CUDA provider AND the " +
                          "pipeline actually builds on it. Applied at startup.",
            Group = "Recognition", Choices = ["auto", "cpu", "gpu"], RestartRequired = true,
        },
        new()
        {
            Key = "ocr_mode", Type = SettingType.Choice, Default = "accurate",
            Label = "OCR engine",
            Description = "'accurate' is MobileNetV4 (best quality); 'fast' is EdgeNext. " +
                          "Baked into the pipeline at construction.",
            Group = "Recognition", Choices = ["accurate", "fast"], RestartRequired = true,
        },
        new()
        {
            Key = "docconf", Type = SettingType.Float, Default = "0.5",
            Label = "Document confidence threshold",
            Description = "Minimum confidence for accepting a detected document type.",
            Group = "Recognition", MinValue = 0.0, MaxValue = 1.0,
        },
        new()
        {
            Key = "img_size", Type = SettingType.Int, Default = "1500",
            Label = "Processing image size",
            Description = "Longest side the image is scaled to before inference. Only ever " +
                          "downscales — a smaller upload is not enlarged.",
            Group = "Recognition", MinValue = 640, MaxValue = 2560,
        },
        new()
        {
            Key = "job_timeout_sec", Type = SettingType.Int, Default = "120",
            Label = "Job timeout (seconds)",
            Description = "Typical processing is well under one second; this is a wedge " +
                          "detector, not a performance limit.",
            Group = "Queue", MinValue = 10, MaxValue = 600,
        },
        new()
        {
            Key = "max_retries", Type = SettingType.Int, Default = "2", Label = "Max retries",
            Description = "Applies to transient failures only. A corrupt image fails " +
                          "immediately and is never retried.",
            Group = "Queue", MinValue = 0, MaxValue = 5,
        },
        new()
        {
            Key = "log_level", Type = SettingType.Choice, Default = "INFO", Label = "Log level",
            Group = "Service", Choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
        },
    ];

    /// <summary>
    /// Indexes the schema. The write whitelist is DERIVED from it rather than duplicated, so a new
    /// setting cannot be readable but not writable.
    /// </summary>
    public static readonly Dictionary<string, SettingDef> ByKey =
        All.ToDictionary(d => d.Key, StringComparer.Ordinal);

    /// <summary>Whether a key may be written through the settings endpoint.</summary>
    public static bool IsUiKey(string key) => ByKey.ContainsKey(key);

    /// <summary>
    /// Validates against the schema and normalises to the stored string form.
    ///
    /// <para>
    /// **A KNOWN key with a bad value is an ERROR, not a silent drop**: a UI that reports "saved"
    /// while discarding the value is worse than one that shows a message.
    /// </para>
    /// </summary>
    public static string Coerce(string key, object? value)
    {
        if (!ByKey.TryGetValue(key, out SettingDef? def))
        {
            throw new SettingValidationException($"unknown setting \"{key}\"");
        }
        string raw = (Convert.ToString(value, CultureInfo.InvariantCulture) ?? "").Trim();

        switch (def.Type)
        {
            case SettingType.Bool:
                return raw.ToLowerInvariant() switch
                {
                    "1" or "true" or "yes" or "on" => "1",
                    _ => "0",
                };

            case SettingType.Int:
            case SettingType.Float:
                if (!double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture,
                        out double number))
                {
                    throw new SettingValidationException(
                        $"{key} must be a number, got \"{raw}\"");
                }
                if (def.MinValue is { } min && number < min)
                {
                    throw new SettingValidationException(
                        $"{key} must be >= {min.ToString(CultureInfo.InvariantCulture)}");
                }
                if (def.MaxValue is { } max && number > max)
                {
                    throw new SettingValidationException(
                        $"{key} must be <= {max.ToString(CultureInfo.InvariantCulture)}");
                }
                // The shortest round-trippable form, so 0.5 stores as "0.5" and not "0.500000" —
                // the stored form is compared against the previous value to decide whether a
                // restart is required, and a formatting change would look like an edit.
                return def.Type == SettingType.Int
                    ? ((int)number).ToString(CultureInfo.InvariantCulture)
                    : number.ToString(CultureInfo.InvariantCulture);

            case SettingType.Choice:
                if (def.Choices is { Length: > 0 } choices)
                {
                    if (Array.IndexOf(choices, raw) >= 0)
                    {
                        return raw;
                    }
                    throw new SettingValidationException(
                        $"{key} must be one of {string.Join(", ", choices)}");
                }
                return raw;

            default:
                return raw;
        }
    }

    /// <summary>
    /// Converts a stored string to the value the worker wants.
    ///
    /// <para>
    /// **A malformed STORED value must not take the worker down** — it falls back to the schema
    /// default. That is the opposite policy from <see cref="Coerce"/>, deliberately: bad input is
    /// rejected at the boundary, but a store that somehow holds a bad value must still yield a
    /// running service.
    /// </para>
    /// </summary>
    public static int TypedInt(string key, string stored, int fallback) =>
        TypedNumber(key, stored) is { } value ? (int)value : fallback;

    public static double TypedFloat(string key, string stored, double fallback) =>
        TypedNumber(key, stored) ?? fallback;

    public static string TypedString(string key, string stored, string fallback)
    {
        if (!ByKey.TryGetValue(key, out SettingDef? def))
        {
            return stored.Length > 0 ? stored : fallback;
        }
        string raw = stored.Length > 0 ? stored : def.Default;
        return raw.Length > 0 ? raw : fallback;
    }

    public static bool TypedBool(string key, string stored)
    {
        string raw = stored;
        if (raw.Length == 0 && ByKey.TryGetValue(key, out SettingDef? def))
        {
            raw = def.Default;
        }
        return raw.ToLowerInvariant() is "1" or "true" or "yes" or "on";
    }

    private static double? TypedNumber(string key, string stored)
    {
        if (!ByKey.TryGetValue(key, out SettingDef? def))
        {
            return null;
        }
        string raw = stored.Length > 0 ? stored : def.Default;
        if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture,
                out double value))
        {
            return value;
        }
        return double.TryParse(def.Default, NumberStyles.Float, CultureInfo.InvariantCulture,
            out double fallbackValue)
            ? fallbackValue
            : null;
    }
}
