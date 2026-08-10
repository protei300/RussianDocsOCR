using System.Text.Json;
using System.Text.Json.Serialization;

namespace RussianDocs.DocumentProcessing.Config;

/// <summary>
/// The per-document charsets from <c>config/ocr_alphabets.json</c>.
///
/// <para>
/// This is NOT the model's alphabet. The model declares its full alphabet in <c>model.json</c>; this
/// table says which of those characters a given script and country is allowed to produce, and the
/// decoder substitutes anything else. Conflating the two silently disables masking.
/// </para>
/// </summary>
public static class Alphabets
{
    private sealed class Table
    {
        [JsonPropertyName("description")] public string? Description { get; set; }
        [JsonPropertyName("specials")] public string Specials { get; set; } = "";
        [JsonPropertyName("default_country")]
        public Dictionary<string, string> DefaultCountry { get; set; } = [];
        [JsonPropertyName("letters_per_country")]
        public Dictionary<string, Dictionary<string, string>> LettersPerCountry { get; set; } = [];
    }

    // Loaded once. The file is small, but it is read per OCR engine and there are four of them.
    // A plain object: System.Threading.Lock is .NET 9+, and this port targets net8.0.
    private static readonly object Gate = new();
    private static Table? _cached;
    private static string? _cachedRoot;

    private static Table Load(string root)
    {
        lock (Gate)
        {
            if (_cached is not null && _cachedRoot == root)
            {
                return _cached;
            }

            string path = Path.Combine(root, "document_processing", "config", "ocr_alphabets.json");
            string text = File.ReadAllText(path).TrimStart('﻿');
            Table table = JsonSerializer.Deserialize<Table>(text)
                ?? throw new InvalidDataException($"config: {path} deserialised to null");

            if (string.IsNullOrEmpty(table.Specials) || table.LettersPerCountry.Count == 0)
            {
                throw new InvalidDataException(
                    $"config: {Path.GetFileName(path)} is missing specials or letters_per_country");
            }

            _cached = table;
            _cachedRoot = root;
            return table;
        }
    }

    public static string DefaultCountry(string root, string script) =>
        Load(root).DefaultCountry.TryGetValue(script, out string? country)
            ? country
            : throw new InvalidDataException($"config: no default country for script \"{script}\"");

    /// <summary>
    /// The characters a script and country may produce, INCLUDING the shared specials.
    ///
    /// <para>
    /// Returned as a set of strings rather than chars so the decoder can compare text elements: the
    /// shipped alphabets are all BMP, but a set of chars would break silently on the first that is
    /// not.
    /// </para>
    /// </summary>
    public static IReadOnlySet<string> AllowedCharset(string root, string script, string? country)
    {
        Table table = Load(root);
        country = string.IsNullOrEmpty(country) ? DefaultCountry(root, script) : country;

        if (!table.LettersPerCountry.TryGetValue(script, out Dictionary<string, string>? byCountry))
        {
            throw new InvalidDataException($"config: unknown script \"{script}\"");
        }
        if (!byCountry.TryGetValue(country, out string? letters))
        {
            throw new InvalidDataException(
                $"config: script \"{script}\" has no country \"{country}\"");
        }

        var set = new HashSet<string>(StringComparer.Ordinal);
        foreach (string source in new[] { letters, table.Specials })
        {
            var enumerator = System.Globalization.StringInfo.GetTextElementEnumerator(source);
            while (enumerator.MoveNext())
            {
                set.Add((string)enumerator.Current);
            }
        }
        return set;
    }
}
