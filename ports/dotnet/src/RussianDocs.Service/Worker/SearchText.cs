using System.Text.Json;
using System.Text.Json.Nodes;
using RussianDocs.DocumentProcessing.ViewModel;

namespace RussianDocs.Service.Worker;

public static class SearchText
{
    /// <summary>
    /// The lowercased haystack for the list page's free-text search.
    ///
    /// <para>
    /// Precomputed at write time so filtering never has to parse the stored result blob. In a SQL
    /// backend this becomes an indexed computed column.
    /// </para>
    ///
    /// <para>
    /// **The OCR values are appended in SORTED KEY ORDER.** Although the haystack is only ever
    /// substring-matched — so order cannot change a search RESULT — an order that depends on
    /// dictionary internals would differ between two runs over the same document, which makes the
    /// stored records non-reproducible and any diff of them noise.
    /// </para>
    /// </summary>
    public static string Build(string filename, Payload payload)
    {
        var parts = new List<string> { filename };
        if (payload.DocType is { } docType)
        {
            parts.Add(docType);
        }

        foreach (string key in payload.Ocr.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            parts.Add(payload.Ocr[key]);
        }

        if (payload.Address is { } address)
        {
            foreach (AddressLine line in address.Lines)
            {
                if (line.Text is { } text)
                {
                    parts.Add(text);
                }
            }
        }
        return string.Join(" ", parts).ToLowerInvariant();
    }

    /// <summary>
    /// Converts the view model into the generic node the store persists.
    ///
    /// <para>
    /// Via JSON rather than by hand, deliberately: the stored blob must be EXACTLY what the API
    /// serves, and a hand-written projection would be a second definition of the wire format, free to
    /// drift from the attributes.
    /// </para>
    /// </summary>
    public static JsonNode ToNode(Payload payload) =>
        JsonNode.Parse(JsonSerializer.SerializeToUtf8Bytes(payload))
        ?? throw new JsonException("worker: view model serialised to nothing");
}
