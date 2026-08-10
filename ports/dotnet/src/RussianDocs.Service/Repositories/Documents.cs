using System.Text.Json.Nodes;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Model;
using RussianDocs.Service.Store;

namespace RussianDocs.Service.Repositories;

/// <summary>
/// The repository functions for documents: query, create, mutate.
///
/// <para>
/// **These signatures ARE the migration contract.** They are copied from
/// <c>service/repositories/*</c>, deliberately, and the whole point of the layer is that swapping the
/// store implementation underneath changes nothing above it.
/// </para>
///
/// <para>
/// Thin by design. Every function takes the store first and delegates the actual query to it, because
/// the backends must express the same question differently — in-memory filtering over JSON files
/// versus real SQL. What lives here is the genuinely shared part: validation, timestamp rules, and
/// the denormalisation performed when a result is saved.
/// </para>
///
/// <para>
/// Mutating functions return a NEW record; callers rebind (<c>record = Documents.Update(db, record,
/// …)</c>). The store hands out copies, so mutating what you got back never touches storage on its
/// own — a property to rely on, not a limitation to work around.
/// </para>
/// </summary>
public static class Documents
{
    /// <summary>The statuses that mean "the list page should keep polling".</summary>
    public static readonly HashSet<string> ActiveStatuses = new(StringComparer.Ordinal)
        { DocumentStatus.Queued, DocumentStatus.Processing };

    /// <summary>One page of matching records plus the unpaged total.</summary>
    public static (IReadOnlyList<Document> Rows, int Total) GetAll(IDocumentStore db,
        DocumentQuery query) => db.QueryDocuments(query);

    /// <summary>The full record, including the recognition result.</summary>
    public static Document? GetById(IDocumentStore db, int id) => db.GetRecord(id);

    /// <summary>
    /// Claims an id WITHOUT inserting a row yet.
    ///
    /// <para>
    /// This exists so a caller can write the upload's bytes BEFORE the document becomes visible to
    /// the worker. Inserting first looks harmless and is a real race: the row lands in
    /// <c>queued</c>, the drain loop runs on its own schedule, and if it claims the document in the
    /// window before the file is written the job fails with "has no stored original" — a good upload
    /// reported as a failed document.
    /// </para>
    /// </summary>
    public static int ReserveId(IDocumentStore db) => db.NextDocumentId();

    /// <summary>Inserts a record. Pass an id from <see cref="ReserveId"/> when artifacts came first.</summary>
    public static Document Create(IDocumentStore db, Document record) => db.PutRecord(record);

    /// <summary>
    /// Applies mutations to a COPY and persists it.
    ///
    /// <para>
    /// <c>UpdatedAt</c> is stamped here, once, so no caller can forget it. <c>Result</c> is carried
    /// across because it is stored separately: a plain field update must not look like a request to
    /// clear it.
    /// </para>
    /// </summary>
    public static Document Update(IDocumentStore db, Document record,
        params Action<Document>[] mutations)
    {
        Document next = record.Clone();
        foreach (Action<Document> mutation in mutations)
        {
            mutation(next);
        }
        next.UpdatedAt = Document.UtcNow();
        next.Result = record.Result;
        return db.PutRecord(next);
    }

    /// <summary>
    /// Moves a document between statuses and stamps the matching timestamp.
    ///
    /// <para>
    /// The status is VALIDATED rather than trusted: it reaches the store, the wire and the SPA's
    /// badge classes, and an invented value would render as an unstyled row somebody then reports as
    /// a UI bug.
    /// </para>
    /// </summary>
    public static Document UpdateStatus(IDocumentStore db, Document record, string status,
        string? errorText, string? errorCode)
    {
        if (!DocumentStatus.Valid.Contains(status))
        {
            throw new ArgumentException($"repo: invalid status \"{status}\"", nameof(status));
        }
        var mutations = new List<Action<Document>>
        {
            d =>
            {
                d.Status = status;
                d.Error = errorText;
                d.ErrorCode = errorCode;
            },
        };
        switch (status)
        {
            case DocumentStatus.Processing:
                mutations.Add(d => d.StartedAt = Document.UtcNow());
                break;
            case DocumentStatus.Done:
            case DocumentStatus.Failed:
                mutations.Add(d => d.FinishedAt = Document.UtcNow());
                break;
        }
        return Update(db, record, mutations.ToArray());
    }

    /// <summary>
    /// Stores the view model and denormalises the columns the list page needs.
    ///
    /// <para>
    /// **The denormalisation IS the point**: without it, filtering or sorting the log means opening
    /// every result blob on every keystroke.
    /// </para>
    /// </summary>
    public static Document SaveResult(IDocumentStore db, Document record, JsonNode payload,
        string searchText, int processingMs)
    {
        db.SaveResultPayload(record.Id, payload);

        JsonObject? quality = payload["quality"] as JsonObject;
        JsonObject? canvas = payload["canvas"] as JsonObject;

        // DocConf is lifted OUT of the quality map into its own column, because the list page sorts
        // by it. The remaining keys stay together: they are verdict strings with no single vocabulary
        // ('good'/'bad' and 'REAL'/'FAKE'), so a column each would invite a client to assume
        // otherwise.
        double? docConf = null;
        var trimmedQuality = new Dictionary<string, object>(StringComparer.Ordinal);
        if (quality is not null)
        {
            foreach ((string key, JsonNode? value) in quality)
            {
                if (key == "DocConf")
                {
                    docConf = AsDouble(value);
                    continue;
                }
                trimmedQuality[key] = value?.DeepClone() ?? (object)"";
            }
        }

        int fieldCount = (payload["fields"] as JsonArray)?.Count ?? 0;
        bool recognised = payload["recognised"]?.GetValue<bool>() ?? false;
        string? docType = AsStringOrNull(payload["doc_type"]);
        string? device = AsStringOrNull(payload["device"]);
        int? canvasW = canvas is null ? null : AsInt(canvas["width"]);
        int? canvasH = canvas is null ? null : AsInt(canvas["height"]);

        return Update(db, record, d =>
        {
            d.Status = DocumentStatus.Done;
            d.Error = null;
            d.ErrorCode = null;
            d.DocType = docType;
            d.DocConf = docConf;
            d.Quality = trimmedQuality;
            d.Recognised = recognised;
            d.FieldCount = fieldCount;
            d.Device = device;
            d.ProcessingMs = processingMs;
            d.CanvasW = canvasW;
            d.CanvasH = canvasH;
            d.HasCanvas = canvasW is not null;
            d.SearchText = searchText;
            d.FinishedAt = Document.UtcNow();
        });
    }

    /// <summary>
    /// Resets a document for another attempt, clearing the previous outcome.
    ///
    /// <para>
    /// <c>RetryCount</c> goes back to zero because this is an OPERATOR action, not an automatic
    /// retry: a human asking for a reprocess should get the full retry budget, not whatever was left.
    /// </para>
    /// </summary>
    public static Document Requeue(IDocumentStore db, Document record) =>
        Update(db, record, d =>
        {
            d.Status = DocumentStatus.Queued;
            d.RetryCount = 0;
            d.Error = null;
            d.ErrorCode = null;
            d.StartedAt = null;
            d.FinishedAt = null;
        });

    public static void Delete(IDocumentStore db, Document record) => db.DropRecord(record.Id);

    public static int? NextQueued(IDocumentStore db) => db.NextQueuedId();

    public static int? QueuePosition(IDocumentStore db, int id) => db.QueuePosition(id);

    /// <summary>
    /// Recovers jobs interrupted mid-flight by a restart.
    ///
    /// <para>
    /// Without it a document caught in <c>processing</c> when the process died sits there forever:
    /// the drain loop only ever claims <c>queued</c> rows. Called once at startup.
    /// </para>
    /// </summary>
    public static int ResetStaleProcessing(IDocumentStore db)
    {
        int count = 0;
        foreach (Document record in db.AllRecords())
        {
            if (record.Status != DocumentStatus.Processing)
            {
                continue;
            }
            Update(db, record, d =>
            {
                d.Status = DocumentStatus.Queued;
                d.StartedAt = null;
            });
            count++;
        }
        return count;
    }

    public static Dictionary<string, int> CountByStatus(IDocumentStore db) => db.CountByStatus();

    public static StoreStats Stats(IDocumentStore db) => db.AggregateStats();

    // -- JSON coercion ------------------------------------------------------
    // The view model arrives as a JsonNode because it round-trips through JSON, where a number has
    // no declared type. These helpers are the one place that knows it, so no caller has to guess
    // whether a field is int or double.

    private static double? AsDouble(JsonNode? node)
    {
        try
        {
            return node?.GetValue<double>();
        }
        catch (Exception ex) when (ex is InvalidOperationException or FormatException)
        {
            return null;
        }
    }

    private static int? AsInt(JsonNode? node) => AsDouble(node) is { } value ? (int)value : null;

    /// <summary>An empty string reads as absent, matching the reference.</summary>
    private static string? AsStringOrNull(JsonNode? node)
    {
        try
        {
            string? value = node?.GetValue<string>();
            return string.IsNullOrEmpty(value) ? null : value;
        }
        catch (Exception ex) when (ex is InvalidOperationException or FormatException)
        {
            return null;
        }
    }
}
