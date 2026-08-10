using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using RussianDocs.Service.Model;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Store;

namespace RussianDocs.Service.Seed;

/// <summary>One manifest row. Names match the committed <c>manifest.json</c> exactly.</summary>
public sealed record SeedEntry
{
    [JsonPropertyName("slug")] public string Slug { get; init; } = "";
    [JsonPropertyName("sample")] public string Sample { get; init; } = "";
    [JsonPropertyName("filename")] public string Filename { get; init; } = "";
    [JsonPropertyName("original_ext")] public string OriginalExt { get; init; } = "";
    [JsonPropertyName("content_type")] public string ContentType { get; init; } = "";
    [JsonPropertyName("size_bytes")] public long SizeBytes { get; init; }
    [JsonPropertyName("search_text")] public string SearchText { get; init; } = "";
}

/// <summary>
/// Populates an empty store with pre-computed sample documents.
///
/// <para>
/// A blank log is a bad first impression and an unhelpful one: there is nothing to click, so nothing
/// demonstrates what the service does. Seeding means the box overlay, the field table and the timings
/// are visible the moment the page loads, across every supported document type.
/// </para>
///
/// <para>
/// **The results are pre-computed, not re-derived.** <c>service/seed_data/</c> holds one finished
/// recognition per document type — the view model, the rendered canvas and a thumbnail — generated once
/// by <c>service/tools/build_seed_data.py</c> and committed. Seeding is therefore a FILE COPY: no GPU,
/// no model load, no minute of startup latency, and the same rows every time regardless of the host's
/// hardware.
/// </para>
///
/// <para>
/// **THIS PORT READS THE SAME DIRECTORY AS THE PYTHON SERVICE.** That is the point: the seeded corpus
/// is ONE artifact with one generator, consumed by every port. A second copy under <c>ports/dotnet/</c>
/// would drift from the first the moment recognition changed, and then two services would disagree
/// about what the reference behaviour is while both looked internally consistent.
/// </para>
///
/// <para>
/// Three rules keep this from becoming a nuisance, all carried over from the reference:
/// </para>
/// <list type="bullet">
/// <item>Only into an EMPTY store, so nothing piles up and a deleted document stays deleted.</item>
/// <item>Only ANONYMISED repository samples. Never a user upload, never a local personal file —
/// everything seeded here is visible to anyone who can reach the UI.</item>
/// <item>ONE PER DOCUMENT TYPE, in the manifest's order, so the log shows the breadth of what the
/// library handles rather than nineteen driving licences.</item>
/// </list>
///
/// <para>
/// Re-run the builder after any change to recognition, or the seeded rows quietly describe an older
/// version's behaviour. Port of <c>service/core/seed.py</c>.
/// </para>
/// </summary>
public static class SeedData
{
    public static string Dir(string repoRoot) =>
        Path.Combine(repoRoot, "service", "seed_data");

    /// <summary>
    /// Inserts the pre-computed samples when the store holds nothing.
    ///
    /// <para>
    /// <paramref name="limit"/> caps how many are inserted; 0 means all available. Returns how many were
    /// added.
    /// </para>
    ///
    /// <para>
    /// **NEVER THROWS**: a service that cannot seed its demo data must still start and accept real
    /// uploads. Every failure path logs and continues, and one bad fixture does not stop the others.
    /// </para>
    /// </summary>
    public static int IfEmpty(IDocumentStore db, string? repoRoot, int limit, ILogger log)
    {
        if (repoRoot is null)
        {
            return 0;
        }
        if (db.CountByStatus().Values.Sum() > 0)
        {
            return 0;
        }

        string dir = Dir(repoRoot);
        List<SeedEntry> entries;
        try
        {
            entries = JsonSerializer.Deserialize<List<SeedEntry>>(
                File.ReadAllBytes(Path.Combine(dir, "manifest.json"))) ?? [];
        }
        catch (Exception ex)
        {
            entries = [];
            log.LogWarning(
                "[SEED] no pre-computed data in {Dir} — the log starts empty; run " +
                "`python service/tools/build_seed_data.py` ({Error})", dir, ex.Message);
        }
        if (entries.Count == 0)
        {
            return 0;
        }
        if (limit > 0 && limit < entries.Count)
        {
            entries = entries.GetRange(0, limit);
        }

        int added = 0;
        foreach (SeedEntry entry in entries)
        {
            try
            {
                SeedOne(db, repoRoot, dir, entry, log);
                added++;
            }
            catch (Exception ex)
            {
                log.LogWarning("[SEED] skipping fixture {Slug}: {Error}", entry.Slug, ex.Message);
            }
        }
        log.LogInformation("[SEED] inserted {Count} pre-computed sample document(s)", added);
        return added;
    }

    private static void SeedOne(IDocumentStore db, string repoRoot, string seedDir,
        SeedEntry entry, ILogger log)
    {
        string entryDir = Path.Combine(seedDir, entry.Slug);

        JsonNode payload = JsonNode.Parse(File.ReadAllBytes(Path.Combine(entryDir, "result.json")))
                           ?? throw new JsonException("result.json is empty");

        // The original is NOT duplicated into the fixture set — it is the repository sample the result
        // was computed from, which is also what keeps the seed data committable.
        byte[] data = File.ReadAllBytes(
            Path.Combine(repoRoot, entry.Sample.Replace('/', Path.DirectorySeparatorChar)));

        // Same BYTES-BEFORE-ROW ordering as an upload. Safe either way here, because seeding finishes
        // before the worker starts — but two orderings for one invariant is how the unsafe one survives
        // a refactor.
        int id = Documents.ReserveId(db);
        Artifacts.SaveOriginal(db, id, data, entry.OriginalExt);

        Document record = Document.New(id, entry.Filename, entry.ContentType, entry.SizeBytes,
            entry.OriginalExt);
        if (Artifacts.DecodeDimensions(data) is { } size)
        {
            record.OriginalW = size.Width;
            record.OriginalH = size.Height;
        }
        record.SearchText = entry.SearchText;
        // Timestamps are NOW rather than the build time, so the log's relative dates ("2 minutes ago")
        // stay sane however old the committed fixtures are.
        DateTime now = Document.UtcNow();
        record.CreatedAt = now;
        record.StartedAt = now;
        record = Documents.Create(db, record);

        string destination = Artifacts.DocDir(db, record.Id);
        foreach (string name in new[] { "canvas.jpg", "thumb.jpg" })
        {
            // A missing preview is not fatal: the fields are the product and the picture is a
            // convenience, exactly as in the worker's own canvas-write path.
            string source = Path.Combine(entryDir, name);
            if (!File.Exists(source))
            {
                continue;
            }
            try
            {
                File.Copy(source, Path.Combine(destination, name), overwrite: true);
            }
            catch (Exception ex)
            {
                log.LogWarning("[SEED] could not copy {File} for {Slug}: {Error}", name,
                    entry.Slug, ex.Message);
            }
        }

        // `timings.total` is the library's own value, in SECONDS (spec/viewmodel.md), while the record
        // stores milliseconds.
        int processingMs = 0;
        if (payload["timings"]?["total"] is JsonValue total &&
            total.TryGetValue(out double seconds))
        {
            processingMs = (int)(seconds * 1000 + 0.5);
        }
        Documents.SaveResult(db, record, payload, entry.SearchText, processingMs);
    }
}
