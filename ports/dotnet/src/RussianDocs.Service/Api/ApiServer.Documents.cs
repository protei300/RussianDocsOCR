using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Model;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Store;
using Results = Microsoft.AspNetCore.Http.Results;

namespace RussianDocs.Service.Api;

/// <summary>
/// The document resource: upload, browse, inspect, re-run, delete.
///
/// <para>
/// Serialisation is hand-written <c>Row</c>/<c>Detail</c> functions rather than derived from the record
/// type, which is the reference's convention and keeps the wire format visible in ONE place instead of
/// spread across attributes on a type that also has to satisfy the store.
/// </para>
/// </summary>
public sealed partial class ApiServer
{
    private const int MaxFilenameLen = 200;

    /// <summary>
    /// Keeps a DISPLAY NAME only — it never touches the filesystem.
    ///
    /// <para>
    /// Stored artifacts always use a fixed name, so even a hostile filename cannot escape the document
    /// directory. This is purely so the UI shows something sensible and bounded; it is NOT the
    /// path-traversal defence, and treating it as one would be a mistake, because the real defence is
    /// that the name is never used as a path at all.
    /// </para>
    /// </summary>
    internal static string SafeFilename(string raw)
    {
        string name = raw.Replace('\\', '/');
        int slash = name.LastIndexOf('/');
        if (slash >= 0)
        {
            name = name[(slash + 1)..];
        }
        name = name.Trim();

        var builder = new StringBuilder(name.Length);
        foreach (char c in name)
        {
            if (char.IsControl(c) || "<>:\"|?*".Contains(c, StringComparison.Ordinal))
            {
                continue;
            }
            builder.Append(c);
        }
        string output = builder.ToString();
        if (output.Length == 0)
        {
            output = "upload";
        }
        // Truncated by TEXT ELEMENTS, not UTF-16 units: a name cut mid-surrogate renders as a
        // replacement character, and these names are routinely Cyrillic here.
        var elements = new System.Globalization.StringInfo(output);
        return elements.LengthInTextElements > MaxFilenameLen
            ? elements.SubstringByTextElements(0, MaxFilenameLen)
            : output;
    }

    /// <summary>One line of the document log.</summary>
    private static Dictionary<string, object?> Row(Document record)
    {
        string? docBase = null, era = null;
        if (record.DocType is { } label)
        {
            (docBase, era) = SplitDocType(label);
        }
        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["id"] = record.Id,
            ["filename"] = record.Filename,
            ["size_bytes"] = record.SizeBytes,
            ["status"] = record.Status,
            ["doc_type"] = record.DocType,
            ["doc_type_base"] = docBase,
            ["doc_type_era"] = era,
            ["recognised"] = record.Recognised,
            ["doc_conf"] = record.DocConf,
            ["quality"] = record.Quality,
            ["field_count"] = record.FieldCount,
            ["device"] = record.Device,
            ["processing_ms"] = record.ProcessingMs,
            ["error"] = record.Error,
            ["error_code"] = record.ErrorCode,
            ["retry_count"] = record.RetryCount,
            ["has_canvas"] = record.HasCanvas,
            ["created_at"] = NullableUtcConverter.Format(record.CreatedAt),
            ["started_at"] = NullableUtcConverter.Format(record.StartedAt),
            ["finished_at"] = NullableUtcConverter.Format(record.FinishedAt),
        };
    }

    private static (string? Base, string? Era) SplitDocType(string label)
    {
        int i = label.LastIndexOf('_');
        if (i < 0)
        {
            return (label.Length > 0 ? label : null, null);
        }
        string docBase = label[..i], era = label[(i + 1)..];
        return (docBase.Length > 0 ? docBase : null, era.Length > 0 ? era : null);
    }

    /// <summary>
    /// The row plus the stored view model flattened into it.
    ///
    /// <para>
    /// The stored result already has the client-facing shape — boxes, fields, canvas dimensions,
    /// coordinate-space notes — so this adds URLs and the original's dimensions and otherwise passes it
    /// through. Re-deriving any of it here would create a second definition of the wire format.
    /// </para>
    /// </summary>
    private static Dictionary<string, object?> Detail(Document record)
    {
        Dictionary<string, object?> payload = Row(record);
        JsonObject result = record.Result as JsonObject ?? [];

        var canvas = new JsonObject();
        if (result["canvas"] is JsonObject stored)
        {
            foreach ((string key, JsonNode? value) in stored)
            {
                canvas[key] = value?.DeepClone();
            }
        }
        canvas["url"] = $"{Prefix}/documents/{record.Id}/image/canvas";

        payload["canvas"] = canvas;
        payload["original"] = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["url"] = $"{Prefix}/documents/{record.Id}/image/original",
            ["width"] = record.OriginalW,
            ["height"] = record.OriginalH,
            ["content_type"] = record.ContentType,
        };
        payload["coord_space"] = result["coord_space"]?.DeepClone();
        payload["coord_space_note"] = result["coord_space_note"]?.DeepClone();
        payload["boxes"] = OrEmptyArray(result["boxes"]);
        payload["fields"] = OrEmptyArray(result["fields"]);
        payload["ocr"] = OrEmptyObject(result["ocr"]);
        payload["quality"] = OrEmptyObject(result["quality"]);
        payload["timings"] = OrEmptyObject(result["timings"]);
        payload["address"] = result["address"]?.DeepClone();
        return payload;
    }

    /// <summary>
    /// Keeps a missing key from becoming a JSON null where the client expects a container.
    ///
    /// <para>
    /// The SPA iterates <c>boxes</c> and <c>fields</c> unconditionally, so a null there is a runtime
    /// error in the browser rather than an empty table.
    /// </para>
    /// </summary>
    private static JsonNode OrEmptyArray(JsonNode? node) =>
        node is JsonArray array ? array.DeepClone() : new JsonArray();

    private static JsonNode OrEmptyObject(JsonNode? node) =>
        node is JsonObject obj ? obj.DeepClone() : new JsonObject();

    /// <summary>
    /// Accepts one image and queues it. **202 with the FULL LIST ROW**, so the SPA can insert the row
    /// without a second request.
    ///
    /// <para>
    /// Everything cheap is checked HERE, so a bad upload fails immediately with an actionable message
    /// instead of becoming a mysterious failed job a minute later.
    /// </para>
    /// </summary>
    private IResult Upload(HttpRequest request)
    {
        long limit = cfg.MaxUploadBytes;

        if (!request.HasFormContentType)
        {
            throw ServiceException.BadRequest(
                "expected a multipart upload with a 'file' part");
        }

        IFormFile file;
        try
        {
            // The limit is enforced by the server's own body-size guard (configured in Program) as
            // well as by the check below, so an oversized upload cannot exhaust memory while being
            // measured.
            IFormCollection form = request.ReadFormAsync().GetAwaiter().GetResult();
            if (form.Files.GetFile("file") is not { } uploaded)
            {
                throw ServiceException.BadRequest("no 'file' part in the upload");
            }
            file = uploaded;
        }
        catch (BadHttpRequestException tooLarge)
            when (tooLarge.StatusCode == StatusCodes.Status413PayloadTooLarge)
        {
            // **An oversized upload fails HERE, not at the size check below.** The body-size guard
            // aborts the read as soon as the cap is passed, so form parsing is what reports it — and
            // reporting that as a malformed request would tell the user to fix their client when the
            // actual problem is a 40 MB file.
            return Results.Json(
                new ApiErrors.ErrorBody($"File exceeds the {cfg.MaxUploadMB} MB limit"),
                statusCode: StatusCodes.Status413PayloadTooLarge);
        }

        byte[] data;
        using (var buffer = new MemoryStream())
        {
            using Stream stream = file.OpenReadStream();
            // CopyToAsync, for the same reason ReadBody is async: Kestrel forbids synchronous reads.
            stream.CopyToAsync(buffer).GetAwaiter().GetResult();
            data = buffer.ToArray();
        }

        if (data.LongLength > limit)
        {
            return Results.Json(
                new ApiErrors.ErrorBody($"File exceeds the {cfg.MaxUploadMB} MB limit"),
                statusCode: StatusCodes.Status413PayloadTooLarge);
        }
        if (data.Length == 0)
        {
            throw ServiceException.BadRequest("Empty upload");
        }

        if (Artifacts.IsPdf(data))
        {
            // Called out separately because people WILL try it, and "unsupported image type" does not
            // tell them what to do about it.
            return Results.Json(
                new ApiErrors.ErrorBody(
                    "PDF is not supported — upload a JPEG, PNG, WEBP, BMP or TIFF image"),
                statusCode: StatusCodes.Status415UnsupportedMediaType);
        }
        if (Artifacts.SniffImage(data) is not { } sniffed)
        {
            // Sniffed from MAGIC BYTES, not the client's Content-Type, which is attacker-controlled
            // and wrong often enough to be useless.
            return Results.Json(
                new ApiErrors.ErrorBody("Unsupported file type — expected an image"),
                statusCode: StatusCodes.Status415UnsupportedMediaType);
        }
        if (Artifacts.DecodeDimensions(data) is not { } size)
        {
            throw ServiceException.Unreadable(
                "The image could not be decoded — it may be corrupt");
        }

        string filename = SafeFilename(file.FileName ?? "");

        // **BYTES FIRST, ROW SECOND.** The record is what makes the document visible to the worker, so
        // writing it before the file leaves a window in which the drain loop can claim a document
        // whose original does not exist yet — reporting a perfectly good upload as failed. See
        // Documents.ReserveId.
        int id = Documents.ReserveId(db);
        Artifacts.SaveOriginal(db, id, data, sniffed.Ext);

        Document record = Document.New(id, filename, sniffed.Media, data.LongLength, sniffed.Ext);
        record.OriginalW = size.Width;
        record.OriginalH = size.Height;
        record.SearchText = filename.ToLowerInvariant();
        record = Documents.Create(db, record);

        worker.NotifyNewWork();
        log.LogInformation("[API] queued document {Id} ({Filename}, {Bytes} bytes)", record.Id,
            filename, data.Length);

        Dictionary<string, object?> output = Row(record);
        output["queue_position"] = Documents.QueuePosition(db, record.Id);
        return Results.Json(output, statusCode: StatusCodes.Status202Accepted);
    }

    /// <summary>Serves one page of the document log.</summary>
    private IResult List(HttpRequest request)
    {
        IQueryCollection query = request.Query;

        // The filter parameter is named `status` on the wire. Keeping that name is a client dependency,
        // not a preference.
        string statusFilter = QueryParams.Str(query, "status");
        if (statusFilter.Length > 0 && !DocumentStatus.Valid.Contains(statusFilter))
        {
            throw ServiceException.BadRequest("Invalid status");
        }
        string sortDir = QueryParams.Str(query, "sort_dir");
        if (sortDir is not ("asc" or "desc"))
        {
            sortDir = "desc";
        }
        // Bounds copied from the reference's own declarations (service/api/documents.py:173-174): page
        // is ge=1 with NO upper bound, page_size is ge=1 le=100. Out of range is a 422, not a clamp —
        // see QueryParams.
        int page = QueryParams.Int(query, "page", 1, 1, 0);
        int pageSize = QueryParams.Int(query, "page_size", 20, 1, 100);

        (IReadOnlyList<Document> rows, int total) = Documents.GetAll(db, new DocumentQuery
        {
            Status = statusFilter,
            DocType = QueryParams.Str(query, "doc_type"),
            Search = QueryParams.Str(query, "search"),
            DateFrom = QueryParams.Str(query, "date_from"),
            DateTo = QueryParams.Str(query, "date_to"),
            Page = page,
            PageSize = pageSize,
            SortBy = QueryParams.Str(query, "sort_by"),
            SortDir = sortDir,
        });

        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["items"] = rows.Select(Row).ToList(),
            ["total"] = total,
            ["page"] = page,
            ["page_size"] = pageSize,
            ["stats"] = Documents.Stats(db),
        });
    }

    private IResult GetDocument(int id)
    {
        Document record = Documents.GetById(db, id)
                          ?? throw ServiceException.NotFound("Document not found");
        return Results.Json(Detail(record));
    }

    /// <summary>
    /// Live progress, a queue position, or a terminal state.
    ///
    /// <para>
    /// **200 with a JSON null when there is nothing to report — never 404.** The polling client would
    /// otherwise raise an error toast every two seconds for a document that finished perfectly well.
    /// </para>
    /// </summary>
    private IResult DocumentProgress(int id)
    {
        Document record = Documents.GetById(db, id)
                          ?? throw ServiceException.NotFound("Document not found");

        if (worker.DocumentProgress(id) is { } live)
        {
            return Results.Json(live);
        }

        switch (record.Status)
        {
            case DocumentStatus.Queued:
                int position = Documents.QueuePosition(db, id) ?? 0;
                return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["step"] = "queued",
                    ["label"] = $"Queued (#{position + 1})",
                    ["pct"] = 0,
                    // The estimate is "everything ahead of me at the current average", which is honest
                    // about being a guess and tracks reality because the average is an EMA of real
                    // completions.
                    ["eta_sec"] = Round1(position * worker.AverageDurationSec()),
                    ["queue_position"] = position,
                });

            case DocumentStatus.Done:
            case DocumentStatus.Failed:
                return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["step"] = record.Status,
                    ["label"] = char.ToUpperInvariant(record.Status[0]) + record.Status[1..],
                    ["pct"] = record.Status == DocumentStatus.Done ? 100 : 0,
                    ["eta_sec"] = null,
                    ["queue_position"] = null,
                });

            default:
                // A JSON null body, deliberately. See the method note.
                return Results.Json<object?>(null);
        }
    }

    /// <summary>
    /// Serves an artifact.
    ///
    /// <para>
    /// <c>no-cache</c> means REVALIDATE, not "do not store": the file result still sends ETag and
    /// Last-Modified, so a repeat request costs a 304 with no body. <c>max-age</c> would be wrong here —
    /// Reprocess overwrites canvas.png and thumb.jpg at the SAME URL, so the browser would keep showing
    /// the previous recognition's image while the field table beside it was already new.
    /// </para>
    /// </summary>
    private IResult ImageArtifact(int id, string kind)
    {
        if (kind is not ("original" or "canvas" or "thumb"))
        {
            throw ServiceException.NotFound("Unknown image kind");
        }
        if (Artifacts.OpenArtifact(db, id, kind) is not { } artifact)
        {
            throw ServiceException.NotFound("Image not available");
        }
        return new CachePrivateFileResult(artifact.Path, artifact.Media);
    }

    private IResult Reprocess(int id)
    {
        Document record = Documents.GetById(db, id)
                          ?? throw ServiceException.NotFound("Document not found");
        if (Documents.ActiveStatuses.Contains(record.Status))
        {
            throw ServiceException.Conflict($"Document is already {record.Status}");
        }
        record = Documents.Requeue(db, record);
        worker.NotifyNewWork();
        return Results.Json(Row(record));
    }

    /// <summary>Returns 204 and an EMPTY BODY.</summary>
    private IResult DeleteDocument(int id)
    {
        Document record = Documents.GetById(db, id)
                          ?? throw ServiceException.NotFound("Document not found");
        Documents.Delete(db, record);
        return Results.NoContent();
    }

    /// <summary>Clears the scratch store. SESSION ONLY — not something an integration does.</summary>
    private IResult Purge()
    {
        int removed = 0;
        foreach (Document record in db.AllRecords())
        {
            // An in-flight job is left alone: deleting its record while the worker holds it produces a
            // "document vanished" failure that looks like a bug.
            if (record.Status == DocumentStatus.Processing)
            {
                continue;
            }
            Documents.Delete(db, record);
            removed++;
        }
        log.LogInformation("[API] purged {Count} document(s)", removed);
        return Results.Json(new Dictionary<string, object?> { ["deleted"] = removed });
    }
}

/// <summary>
/// A file response that sets <c>Cache-Control: private, no-cache</c> before the body is written.
///
/// <para>
/// A small wrapper rather than setting the header at the call site: <c>Results.File</c> writes the
/// response as soon as it executes, so a header assigned afterwards never reaches the client. That
/// ordering bug is invisible locally and shows up as a stale canvas after a reprocess.
/// </para>
/// </summary>
internal sealed class CachePrivateFileResult(string path, string contentType) : IResult
{
    public Task ExecuteAsync(HttpContext context)
    {
        context.Response.Headers.CacheControl = "private, no-cache";
        return Results.File(path, contentType, enableRangeProcessing: true)
            .ExecuteAsync(context);
    }
}
