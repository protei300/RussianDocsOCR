using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using RussianDocs.Service.Model;

namespace RussianDocs.Service.Store;

/// <summary>
/// The whitelist of sortable columns, shared by every backend so they cannot drift apart.
///
/// <para>
/// A whitelist rather than dynamic member lookup: in a SQL backend that difference is an injection
/// vector, and here it is what stops a typo in a query string from silently sorting by nothing.
/// </para>
/// </summary>
public static class SortColumns
{
    public static readonly HashSet<string> All = new(StringComparer.Ordinal)
    {
        "created_at", "filename", "status", "doc_type", "doc_conf", "processing_ms", "size_bytes",
    };
}

/// <summary>
/// The filter/sort/page request for a document listing.
///
/// <para>
/// A type rather than a long parameter list because it crosses the store boundary and grows: a
/// positional call is where "swap date_from and date_to" hides.
/// </para>
/// </summary>
public sealed record DocumentQuery
{
    public string Status { get; init; } = "";
    public string DocType { get; init; } = "";
    public string Search { get; init; } = "";
    public string DateFrom { get; init; } = "";
    public string DateTo { get; init; } = "";
    public int Page { get; init; } = 1;
    public int PageSize { get; init; } = 20;
    public string SortBy { get; init; } = "created_at";
    public string SortDir { get; init; } = "desc";
}

/// <summary>The aggregate summary the status page shows.</summary>
public sealed record StoreStats
{
    [JsonPropertyName("queued")] public int Queued { get; init; }
    [JsonPropertyName("processing")] public int Processing { get; init; }
    [JsonPropertyName("done")] public int Done { get; init; }
    [JsonPropertyName("failed")] public int Failed { get; init; }
    [JsonPropertyName("total")] public int Total { get; init; }
    [JsonPropertyName("recognised")] public int Recognised { get; init; }
    [JsonPropertyName("avg_processing_ms")] public int? AvgProcessingMs { get; init; }
}

/// <summary>
/// Everything the service needs from a storage backend.
///
/// <para>
/// **SQL SWAP POINT.** Implementing this interface over a real database, and constructing that
/// instead of <see cref="FileStore"/>, is the whole migration as far as callers are concerned.
/// Controller and worker code does not change.
/// </para>
///
/// <para>
/// **Query methods live HERE rather than in the repositories**, deliberately: filtering a list in
/// memory is correct for a few hundred JSON files and wrong for a table, so each backend has to
/// express "the newest twenty matching rows" in its own terms. Putting the queries behind the
/// interface is what lets it.
/// </para>
///
/// <para>Port of <c>service/core/store.py</c> and <c>service/core/database.py</c>.</para>
/// </summary>
public interface IDocumentStore
{
    /// <summary>
    /// "files" or "sql" — surfaced on the status page, because "why did my data vanish" is answered
    /// by this one word.
    /// </summary>
    string Backend { get; }

    /// <summary>Whether the contents survive a restart.</summary>
    bool IsEphemeral { get; }

    int NextDocumentId();
    Document? GetRecord(int id);
    Document PutRecord(Document record);
    void DropRecord(int id);
    (IReadOnlyList<Document> Rows, int Total) QueryDocuments(DocumentQuery query);
    IReadOnlyList<Document> AllRecords();
    int? NextQueuedId();
    int? QueuePosition(int id);
    Dictionary<string, int> CountByStatus();
    StoreStats AggregateStats();

    void SaveResultPayload(int id, JsonNode payload);
    JsonNode? LoadResultPayload(int id);

    IReadOnlyList<ApiKey> AllApiKeys();
    int NextApiKeyId();
    ApiKey PutApiKey(ApiKey key);
    bool DropApiKey(int id);

    Dictionary<string, string> AllSettings();
    Dictionary<string, string> SetSettings(IReadOnlyDictionary<string, string> values);

    /// <summary>
    /// A plain directory in every backend: binary artifacts stay on the filesystem regardless of
    /// where the metadata lives.
    /// </summary>
    string DocDir(int id);

    long DiskUsageBytes();
}
