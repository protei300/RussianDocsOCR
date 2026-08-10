using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Logging;
using RussianDocs.Service.Model;

namespace RussianDocs.Service.Store;

/// <summary>
/// The filesystem backend.
///
/// <para>
/// On-disk layout:
/// <code>
/// $DATA_DIR/
///   documents/42/
///     record.json     the "row"
///     original.jpg    exactly the bytes uploaded
///     canvas.png      the deskewed/rectified canvas
///     result.json     the full recognition view model
///   api_keys.json
///   settings.json
/// </code>
/// </para>
///
/// <para>
/// Four design notes worth reading before changing anything here:
/// </para>
/// <list type="bullet">
/// <item><b>The index lives in memory; disk is scanned once at startup.</b> The service is pinned to
/// ONE process — the pipeline singleton and this index both are — so a shared in-memory index is
/// legitimate rather than a shortcut.</item>
/// <item><b>Writes are atomic</b> (temp file plus rename, atomic on NTFS and ext4). A half-written
/// record.json would survive a crash and poison the next boot.</item>
/// <item><b>Reads return COPIES.</b> A live shared instance would let one request's edit leak into
/// another's view, so update returns a NEW record and callers must rebind.</item>
/// <item><b>result is not held in the index</b> — it can be 100 KB of boxes per document. Get-by-id
/// loads it lazily; list queries never touch it.</item>
/// </list>
///
/// <para>
/// **Concurrency: ONE lock guards the index and all mutations**, because both the worker and the
/// request handlers write here. Long I/O — writing a 2 MB PNG — happens OUTSIDE the lock; only the
/// rename and the index update are inside it. The lock is NOT reentrant (C# <c>lock</c> on a monitor
/// technically is, but the Go port cannot be, and both are held to the same rule so the two read
/// alike): every public method takes it at most once and calls only <c>…Locked</c> helpers beneath
/// it. That is a real constraint on edits to this file.
/// </para>
/// </summary>
public sealed class FileStore : IDocumentStore
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    private readonly string _root;
    private readonly string _docsDir;
    private readonly ILogger _log;

    private readonly object _gate = new();
    private readonly Dictionary<int, Document> _records = [];
    private readonly Dictionary<int, ApiKey> _apiKeys = [];
    private Dictionary<string, string> _settings = new(StringComparer.Ordinal);
    private int _nextDocId = 1;
    private int _nextKeyId = 1;

    public FileStore(string root, ILogger log)
    {
        _root = Path.GetFullPath(root);
        _docsDir = Path.Combine(_root, "documents");
        _log = log;
        Directory.CreateDirectory(_docsDir);
        Scan();
    }

    public string Backend => "files";
    public bool IsEphemeral => true;

    /// <summary>
    /// Writes JSON so a crash can never leave a partial file behind.
    ///
    /// <para>
    /// Temp file plus rename, which is atomic on NTFS and ext4. Not a nicety: a truncated
    /// record.json survives the crash and poisons the next boot, and the failure then looks like
    /// data corruption rather than an interrupted write.
    /// </para>
    /// </summary>
    public static void AtomicWriteJson(string path, object payload) =>
        AtomicWriteBytes(path, JsonSerializer.SerializeToUtf8Bytes(payload, Json));

    public static void AtomicWriteBytes(string path, byte[] data)
    {
        string tmp = path + ".tmp";
        File.WriteAllBytes(tmp, data);
        try
        {
            // Overwrite: File.Move with overwrite is the atomic replace on both platforms.
            File.Move(tmp, path, overwrite: true);
        }
        catch
        {
            try { File.Delete(tmp); } catch { /* the original write error is what matters */ }
            throw;
        }
    }

    /// <summary>
    /// Empties the data directory. Called before construction when configured.
    ///
    /// <para>
    /// Deliberate, and the reason the "ephemeral" promise is true: `docker restart` KEEPS the
    /// writable layer, so the absence of a volume is not enough on its own.
    /// </para>
    ///
    /// <para>
    /// **The CONTENTS go, the directory stays** — and that is not a detail. Removing the directory
    /// itself needs write permission on its PARENT, which a non-root container does not have for
    /// <c>/app</c>, and a directory that is a MOUNT POINT can never be unlinked at all, by anyone.
    /// So deleting the directory fails in both of the configurations this service is actually
    /// deployed in; the Go port found it the first time the image ran, as
    /// <c>unlinkat /app/data: permission denied</c>, with the store then unusable. Emptying achieves
    /// the same observable result — nothing survives a restart — and works as non-root, and works
    /// under <c>-v</c>.
    /// </para>
    /// </summary>
    public static long Wipe(string root)
    {
        string abs = Path.GetFullPath(root);
        if (!Directory.Exists(abs))
        {
            return 0; // nothing to wipe is success, not an error
        }
        long size = DirSize(abs);
        foreach (string dir in Directory.GetDirectories(abs))
        {
            Directory.Delete(dir, recursive: true);
        }
        foreach (string file in Directory.GetFiles(abs))
        {
            File.Delete(file);
        }
        return size;
    }

    /// <summary>
    /// Rebuilds the in-memory index from disk. Cheap: N small JSON reads.
    ///
    /// <para>
    /// A corrupt record is SKIPPED with a log line rather than failing the scan. The rest of the
    /// scratch data is still perfectly usable, and a service that refuses to start because one of two
    /// hundred files is truncated is worse than one that starts with 199.
    /// </para>
    /// </summary>
    private void Scan()
    {
        string[] dirs;
        try
        {
            dirs = Directory.GetDirectories(_docsDir);
        }
        catch (Exception ex)
        {
            _log.LogWarning("[STORE] cannot list documents in {Dir}: {Error}", _docsDir, ex.Message);
            return;
        }
        Array.Sort(dirs, StringComparer.Ordinal);

        int loaded = 0;
        foreach (string dir in dirs)
        {
            string file = Path.Combine(dir, "record.json");
            if (!File.Exists(file))
            {
                continue;
            }
            Document? record;
            try
            {
                record = JsonSerializer.Deserialize<Document>(File.ReadAllBytes(file));
            }
            catch (Exception ex)
            {
                _log.LogWarning("[STORE] skipping unreadable record {File}: {Error}", file,
                    ex.Message);
                continue;
            }
            if (record is null)
            {
                continue;
            }
            _records[record.Id] = record;
            _nextDocId = Math.Max(_nextDocId, record.Id + 1);
            loaded++;
        }

        if (File.Exists(ApiKeysPath))
        {
            try
            {
                foreach (ApiKey key in
                         JsonSerializer.Deserialize<List<ApiKey>>(File.ReadAllBytes(ApiKeysPath))
                         ?? [])
                {
                    _apiKeys[key.Id] = key;
                    _nextKeyId = Math.Max(_nextKeyId, key.Id + 1);
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning("[STORE] api_keys.json unreadable — starting with none: {Error}",
                    ex.Message);
            }
        }

        if (File.Exists(SettingsPath))
        {
            try
            {
                _settings =
                    JsonSerializer.Deserialize<Dictionary<string, string>>(
                        File.ReadAllBytes(SettingsPath)) ?? _settings;
            }
            catch (Exception ex)
            {
                _log.LogWarning("[STORE] settings.json unreadable — using defaults: {Error}",
                    ex.Message);
            }
        }

        if (loaded > 0)
        {
            _log.LogInformation("[STORE] recovered {Count} documents from {Dir}", loaded, _docsDir);
        }
    }

    private string ApiKeysPath => Path.Combine(_root, "api_keys.json");
    private string SettingsPath => Path.Combine(_root, "settings.json");

    public string DocDir(int id) =>
        Path.Combine(_docsDir, id.ToString(CultureInfo.InvariantCulture));

    // -- documents ----------------------------------------------------------

    public int NextDocumentId()
    {
        lock (_gate)
        {
            return _nextDocId++;
        }
    }

    public IReadOnlyList<Document> AllRecords()
    {
        lock (_gate)
        {
            // Deterministic order regardless of dictionary enumeration. Callers sort by their own
            // key afterwards, but an unstable base order makes equal keys shuffle between requests
            // — visible in the UI as rows jumping.
            return _records.Values.Select(r => r.Clone()).OrderBy(r => r.Id).ToList();
        }
    }

    /// <summary>
    /// Returns a COPY with the lazily-stored result attached.
    ///
    /// <para>
    /// A copy rather than the indexed instance: callers mutate what they get back, and sharing would
    /// let one request's edit leak into another's view.
    /// </para>
    /// </summary>
    public Document? GetRecord(int id)
    {
        Document? record;
        lock (_gate)
        {
            _records.TryGetValue(id, out record);
        }
        if (record is null)
        {
            return null;
        }
        Document copy = record.Clone();
        // Loaded OUTSIDE the lock: it is a file read of up to 100 KB, and holding the index lock
        // across it would serialise every other reader for no benefit.
        copy.Result = LoadResultPayload(id);
        return copy;
    }

    /// <summary>
    /// Persists a record and indexes it.
    ///
    /// <para>
    /// **The FILE IS WRITTEN BEFORE THE INDEX ENTRY**, and the order matters: a record is what makes
    /// a document visible to the worker, so indexing it before the bytes exist lets the drain loop
    /// claim a document whose file is not written yet.
    /// </para>
    /// </summary>
    public Document PutRecord(Document record)
    {
        string dir = DocDir(record.Id);
        try
        {
            Directory.CreateDirectory(dir);
            AtomicWriteJson(Path.Combine(dir, "record.json"), record);
        }
        catch (Exception ex)
        {
            _log.LogError("[STORE] cannot write record {Id}: {Error}", record.Id, ex.Message);
            return record;
        }
        Document stored = record.Clone();
        lock (_gate)
        {
            _records[record.Id] = stored;
            _nextDocId = Math.Max(_nextDocId, record.Id + 1);
        }
        return record;
    }

    public void DropRecord(int id)
    {
        lock (_gate)
        {
            _records.Remove(id);
        }
        // Outside the lock: removing a directory of multi-megabyte artifacts is slow, and nothing
        // else can reach the record now that it is out of the index.
        try
        {
            if (Directory.Exists(DocDir(id)))
            {
                Directory.Delete(DocDir(id), recursive: true);
            }
        }
        catch (Exception ex)
        {
            _log.LogWarning("[STORE] could not remove artifacts for {Id}: {Error}", id, ex.Message);
        }
    }

    // -- api keys -----------------------------------------------------------

    public IReadOnlyList<ApiKey> AllApiKeys()
    {
        lock (_gate)
        {
            return _apiKeys.Values.Select(k => k.Clone()).OrderBy(k => k.Id).ToList();
        }
    }

    public int NextApiKeyId()
    {
        lock (_gate)
        {
            return _nextKeyId++;
        }
    }

    public ApiKey PutApiKey(ApiKey key)
    {
        lock (_gate)
        {
            _apiKeys[key.Id] = key.Clone();
            _nextKeyId = Math.Max(_nextKeyId, key.Id + 1);
            FlushApiKeysLocked();
        }
        return key;
    }

    public bool DropApiKey(int id)
    {
        lock (_gate)
        {
            if (!_apiKeys.Remove(id))
            {
                return false;
            }
            FlushApiKeysLocked();
            return true;
        }
    }

    /// <summary>
    /// Assumes the lock is held. Named so the assumption is visible at the call site — see the note
    /// on <see cref="FileStore"/> about non-reentrancy.
    /// </summary>
    private void FlushApiKeysLocked() =>
        AtomicWriteJson(ApiKeysPath, _apiKeys.Values.OrderBy(k => k.Id).ToList());

    // -- settings -----------------------------------------------------------

    public Dictionary<string, string> AllSettings()
    {
        lock (_gate)
        {
            return new Dictionary<string, string>(_settings, StringComparer.Ordinal);
        }
    }

    public Dictionary<string, string> SetSettings(IReadOnlyDictionary<string, string> values)
    {
        lock (_gate)
        {
            foreach ((string key, string value) in values)
            {
                _settings[key] = value;
            }
            AtomicWriteJson(SettingsPath, _settings);
            return new Dictionary<string, string>(_settings, StringComparer.Ordinal);
        }
    }

    // -- results ------------------------------------------------------------

    public void SaveResultPayload(int id, JsonNode payload)
    {
        string dir = DocDir(id);
        Directory.CreateDirectory(dir);
        AtomicWriteBytes(Path.Combine(dir, "result.json"),
            JsonSerializer.SerializeToUtf8Bytes(payload, Json));
    }

    public JsonNode? LoadResultPayload(int id)
    {
        string file = Path.Combine(DocDir(id), "result.json");
        if (!File.Exists(file))
        {
            return null;
        }
        try
        {
            return JsonNode.Parse(File.ReadAllBytes(file));
        }
        catch (Exception ex)
        {
            _log.LogWarning("[STORE] unreadable result.json for {Id}: {Error}", id, ex.Message);
            return null;
        }
    }

    // -- queries ------------------------------------------------------------
    // Implemented over the in-memory index. Correct at this scale (a few hundred records) and honest
    // about it: a SQL backend answers the same questions with real queries.

    public (IReadOnlyList<Document> Rows, int Total) QueryDocuments(DocumentQuery query)
    {
        IEnumerable<Document> rows = AllRecords();

        if (query.Status.Length > 0)
        {
            rows = rows.Where(r => r.Status == query.Status);
        }
        // '__none__' means "unrecognised", which is not the same as "no doc_type": a failed document
        // has neither, and the UI offers one filter for both.
        if (query.DocType == "__none__")
        {
            rows = rows.Where(r => !r.Recognised);
        }
        else if (query.DocType.Length > 0)
        {
            rows = rows.Where(r => r.DocType is not null &&
                                   r.DocType.StartsWith(query.DocType, StringComparison.Ordinal));
        }
        if (ParseDay(query.DateFrom) is { } start)
        {
            rows = rows.Where(r => r.CreatedAt is not null && r.CreatedAt >= start);
        }
        if (ParseDay(query.DateTo) is { } end)
        {
            // Inclusive of the whole named day, which is what a date picker means by "to".
            DateTime limit = end.AddDays(1);
            rows = rows.Where(r => r.CreatedAt is not null && r.CreatedAt < limit);
        }
        string needle = query.Search.Trim().ToLowerInvariant();
        if (needle.Length > 0)
        {
            rows = rows.Where(r => r.SearchText.Contains(needle, StringComparison.Ordinal));
        }

        List<Document> matched = rows.ToList();
        int total = matched.Count;

        string column = SortColumns.All.Contains(query.SortBy) ? query.SortBy : "created_at";
        bool desc = query.SortDir != "asc";
        List<Document> sorted = SortRows(matched, column, desc);

        int pageSize = query.PageSize > 0 ? query.PageSize : 20;
        int page = query.Page < 1 ? 1 : query.Page;
        int offset = Math.Min((page - 1) * pageSize, sorted.Count);
        return (sorted.GetRange(offset, Math.Min(pageSize, sorted.Count - offset)), total);
    }

    /// <summary>
    /// Orders by one whitelisted column.
    ///
    /// <para>
    /// **NULLS LAST IN BOTH DIRECTIONS**, matching what a SQL backend must do: a queued document has
    /// no doc_conf and must not lead an ascending sort. That is why the ordering is on an
    /// (isNull, value) pair rather than on the value alone, and why the null test is never reversed.
    /// </para>
    ///
    /// <para>
    /// **LINQ <c>OrderBy</c>, not <c>List.Sort</c>**: equal keys must keep their previous relative
    /// order, or rows jump between refreshes of the list page for no visible reason.
    /// <c>List.Sort</c> is an unstable introsort; <c>OrderBy</c> is documented stable.
    /// </para>
    /// </summary>
    private static List<Document> SortRows(List<Document> rows, string column, bool desc)
    {
        IOrderedEnumerable<Document> ordered = rows.OrderBy(r => SortKey(r, column).IsNull);
        ordered = desc
            ? ordered.ThenByDescending(r => SortKey(r, column).Key, StringComparer.Ordinal)
            : ordered.ThenBy(r => SortKey(r, column).Key, StringComparer.Ordinal);
        return ordered.ToList();
    }

    /// <summary>
    /// Returns (isNull, comparable) for a column.
    ///
    /// <para>
    /// Every column reduces to a STRING key so that one comparator covers dates, numbers and text
    /// alike, instead of a type switch inside the comparator. Numbers go through
    /// <see cref="NumKey"/>, which renders them in a lexicographically ordered fixed width;
    /// timestamps use a round-trip UTC format, which sorts correctly as text by construction.
    /// </para>
    /// </summary>
    private static (bool IsNull, string Key) SortKey(Document r, string column) => column switch
    {
        "filename" => (false, r.Filename.ToLowerInvariant()),
        "status" => (false, r.Status),
        "doc_type" => r.DocType is null ? (true, "") : (false, r.DocType),
        "doc_conf" => r.DocConf is null ? (true, "") : (false, NumKey(r.DocConf.Value)),
        "processing_ms" => r.ProcessingMs is null
            ? (true, "")
            : (false, NumKey(r.ProcessingMs.Value)),
        "size_bytes" => (false, NumKey(r.SizeBytes)),
        _ => r.CreatedAt is null
            ? (true, "")
            : (false, r.CreatedAt.Value.ToUniversalTime()
                // Quoted T and Z: see the note on NullableUtcConverter.Pattern — unquoted, .NET
                // reads them as format specifiers and throws.
                .ToString("yyyy-MM-dd'T'HH:mm:ss.fffffff'Z'", CultureInfo.InvariantCulture)),
    };

    /// <summary>
    /// Renders a number as a fixed-width, lexicographically ordered string.
    ///
    /// <para>
    /// This exists so ONE string comparator can order every column, numeric and textual alike,
    /// instead of a type switch in the comparator. The offset keeps negatives ordered correctly; the
    /// width covers every value these columns can hold (a byte count, a millisecond count, a 0..1
    /// confidence).
    /// </para>
    /// </summary>
    private static string NumKey(double v) =>
        (v + 1e9).ToString("F6", CultureInfo.InvariantCulture).PadLeft(20, '0');

    public int? NextQueuedId()
    {
        Document? best = null;
        foreach (Document r in AllRecords())
        {
            if (r.Status != DocumentStatus.Queued)
            {
                continue;
            }
            if (best is null || Earlier(r, best))
            {
                best = r;
            }
        }
        return best?.Id;
    }

    public int? QueuePosition(int id)
    {
        List<Document> queued = AllRecords()
            .Where(r => r.Status == DocumentStatus.Queued)
            .OrderBy(r => r.CreatedAt ?? DateTime.MaxValue)
            .ThenBy(r => r.Id)
            .ToList();
        int index = queued.FindIndex(r => r.Id == id);
        return index < 0 ? null : index;
    }

    /// <summary>
    /// FIFO by creation, with the id as the tie-breaker.
    ///
    /// <para>
    /// The tie-break is not decoration: two uploads inside the same clock tick would otherwise have
    /// an unspecified order, and the queue would not be FIFO in exactly the case where somebody is
    /// testing it by uploading twice quickly.
    /// </para>
    /// </summary>
    private static bool Earlier(Document a, Document b)
    {
        if (a.CreatedAt is { } at && b.CreatedAt is { } bt && at != bt)
        {
            return at < bt;
        }
        return a.Id < b.Id;
    }

    public Dictionary<string, int> CountByStatus()
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            [DocumentStatus.Queued] = 0,
            [DocumentStatus.Processing] = 0,
            [DocumentStatus.Done] = 0,
            [DocumentStatus.Failed] = 0,
        };
        foreach (Document r in AllRecords())
        {
            counts[r.Status] = counts.GetValueOrDefault(r.Status) + 1;
        }
        return counts;
    }

    public StoreStats AggregateStats()
    {
        IReadOnlyList<Document> rows = AllRecords();
        Dictionary<string, int> counts = CountByStatus();

        int sum = 0, n = 0, recognised = 0;
        foreach (Document r in rows)
        {
            if (r.Recognised)
            {
                recognised++;
            }
            if (r.Status == DocumentStatus.Done && r.ProcessingMs is > 0)
            {
                sum += r.ProcessingMs.Value;
                n++;
            }
        }

        return new StoreStats
        {
            Queued = counts[DocumentStatus.Queued],
            Processing = counts[DocumentStatus.Processing],
            Done = counts[DocumentStatus.Done],
            Failed = counts[DocumentStatus.Failed],
            Total = rows.Count,
            Recognised = recognised,
            AvgProcessingMs = n > 0 ? (int)((double)sum / n + 0.5) : null,
        };
    }

    public long DiskUsageBytes() => DirSize(_docsDir);

    private static long DirSize(string root)
    {
        if (!Directory.Exists(root))
        {
            return 0;
        }
        long total = 0;
        // A vanished file mid-walk is normal here (the worker writes while the status page reads),
        // so an error skips the entry rather than the walk.
        foreach (string file in Directory.EnumerateFiles(root, "*",
                     new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true }))
        {
            try
            {
                total += new FileInfo(file).Length;
            }
            catch
            {
                // skipped on purpose, see above
            }
        }
        return total;
    }

    /// <summary>
    /// Accepts YYYY-MM-DD.
    ///
    /// <para>
    /// **A HALF-TYPED DATE DISABLES THE FILTER** rather than erroring: the list page sends the field
    /// on every keystroke, and rejecting "2026-0" would make the page flash an error while somebody
    /// is still typing.
    /// </para>
    /// </summary>
    private static DateTime? ParseDay(string value)
    {
        if (value.Length == 0)
        {
            return null;
        }
        return DateTime.TryParseExact(value, "yyyy-MM-dd", CultureInfo.InvariantCulture,
            DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out DateTime day)
            ? day
            : null;
    }
}
