using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using RussianDocs.DocumentProcessing.Config;

namespace RussianDocs.DocumentProcessing.Models;

/// <summary>
/// The <c>model.json</c> that sits beside every artifact.
///
/// <para>
/// **Every optional numeric field is nullable**, and that is load-bearing rather than pedantic:
/// <c>BlankIndex</c> is legitimately 0 and <c>Threshold</c> legitimately defaults to 0.5, so a
/// non-nullable <c>double</c> cannot tell "absent" from "zero" — and the two mean different things.
/// </para>
/// </summary>
public sealed class ModelConfig
{
    [JsonPropertyName("Name")] public string Name { get; set; } = "";
    [JsonPropertyName("File")] public string File { get; set; } = "";
    [JsonPropertyName("ModelType")] public string ModelType { get; set; } = "";
    [JsonPropertyName("Runtime")] public string? Runtime { get; set; }
    [JsonPropertyName("Inputs")] public List<ModelInput> Inputs { get; set; } = [];
    [JsonPropertyName("Outputs")] public List<ModelOutput> Outputs { get; set; } = [];

    /// <summary>Directory the config was read from. Not part of the JSON.</summary>
    [JsonIgnore] public string Dir { get; set; } = "";

    /// <summary>Absolute path to the weights, with separators normalised for this platform.</summary>
    [JsonIgnore] public string ModelPath =>
        Path.Combine(Dir, ModelPaths.NormaliseSeparators(File));

    /// <summary>
    /// Reads a <c>model.json</c>.
    ///
    /// <para>
    /// **The BOM must be stripped before parsing** (D-10). PowerShell's
    /// <c>Set-Content -Encoding utf8</c> writes one, and <c>JsonSerializer</c> then fails on the very
    /// first character with a message about invalid JSON rather than about encoding — which sends the
    /// reader looking for a syntax error that is not there.
    /// </para>
    /// </summary>
    public static ModelConfig Load(string dir)
    {
        string path = Path.Combine(dir, "model.json");
        string text = System.IO.File.ReadAllText(path, Encoding.UTF8).TrimStart('﻿');

        ModelConfig config = JsonSerializer.Deserialize<ModelConfig>(text)
            ?? throw new InvalidDataException($"models: {path} deserialised to null");
        config.Dir = dir;

        if (config.Inputs.Count == 0 || config.Outputs.Count == 0)
        {
            throw new InvalidDataException($"models: {path} declares no inputs or no outputs");
        }
        return config;
    }
}

public sealed class ModelInput
{
    [JsonPropertyName("Type")] public string Type { get; set; } = "";
    [JsonPropertyName("Name")] public string? Name { get; set; }
    [JsonPropertyName("Shape")] public List<int>? Shape { get; set; }
    [JsonPropertyName("Normalization")] public List<double>? Normalization { get; set; }
    [JsonPropertyName("PaddingSize")] public List<int>? PaddingSize { get; set; }
    [JsonPropertyName("PaddingColor")] public List<int>? PaddingColor { get; set; }
    [JsonPropertyName("Height")] public int? Height { get; set; }
    [JsonPropertyName("ColorOrder")] public string? ColorOrder { get; set; }
    [JsonPropertyName("Dtype")] public string? Dtype { get; set; }
}

public sealed class ModelOutput
{
    [JsonPropertyName("Type")] public string Type { get; set; } = "";
    [JsonPropertyName("Name")] public string? Name { get; set; }
    [JsonPropertyName("Shape")] public List<int>? Shape { get; set; }

    /// <summary>
    /// Raw, because it is sometimes a list of strings and sometimes a list of numbers — the angle
    /// head declares <c>[0, 90, 180, 270]</c> while the detectors declare names.
    /// </summary>
    [JsonPropertyName("Labels")] public JsonElement Labels { get; set; }

    [JsonPropertyName("Threshold")] public double? Threshold { get; set; }
    [JsonPropertyName("IOU")] public double? Iou { get; set; }
    [JsonPropertyName("CLS")] public double? Cls { get; set; }
    [JsonPropertyName("MaskFilter")] public double? MaskFilter { get; set; }
    [JsonPropertyName("Metric")] public string? Metric { get; set; }
    [JsonPropertyName("Centers")] public string? Centers { get; set; }
    [JsonPropertyName("Alphabet")] public string? Alphabet { get; set; }
    [JsonPropertyName("Script")] public string? Script { get; set; }
    [JsonPropertyName("Country")] public string? Country { get; set; }
    [JsonPropertyName("BlankIndex")] public int? BlankIndex { get; set; }

    /// <summary>
    /// Labels as strings, whichever way they were written.
    ///
    /// <para>
    /// Numbers are formatted the way Python's <c>str()</c> would — <c>90</c>, not <c>90.0</c> —
    /// because the angle lookup compares label strings, and a formatting difference there turns a
    /// valid angle into "not one of the declared labels".
    /// </para>
    /// </summary>
    public string[] LabelsAsStrings()
    {
        if (Labels.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var result = new List<string>();
        foreach (JsonElement item in Labels.EnumerateArray())
        {
            result.Add(item.ValueKind switch
            {
                JsonValueKind.String => item.GetString() ?? "",
                JsonValueKind.Number => FormatNumber(item.GetDouble()),
                _ => throw new InvalidDataException(
                    $"models: Labels contains a {item.ValueKind}, expected string or number"),
            });
        }
        return [.. result];
    }

    public int[] LabelsAsInts()
    {
        var result = new List<int>();
        foreach (string label in LabelsAsStrings())
        {
            if (!int.TryParse(label, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v))
            {
                throw new InvalidDataException($"models: label \"{label}\" is not an integer");
            }
            result.Add(v);
        }
        return [.. result];
    }

    /// <summary>Matches Go's <c>%g</c> and Python's <c>str()</c> for the integral values used here.</summary>
    private static string FormatNumber(double value) =>
        value == Math.Floor(value) && Math.Abs(value) < 1e15
            ? ((long)value).ToString(CultureInfo.InvariantCulture)
            : value.ToString("G", CultureInfo.InvariantCulture);
}
