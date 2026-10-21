using System.Buffers;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.Logging;
using OpenCvSharp;
using RussianDocs.DocumentProcessing.ViewModel;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Ml;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Settings;
using Image = RussianDocs.DocumentProcessing.Imaging.Image;
using Results = Microsoft.AspNetCore.Http.Results;

namespace RussianDocs.Service.Api;

/// <summary>
/// Recognition in memory: one image in, one answer out, nothing kept. Port of
/// <c>service/api/recognize.py</c>; the contract is ports/MEMORY.md.
///
/// <para>
/// <c>POST /recognize</c> reads the image into a buffer it owns, decodes it, recognises it inside the
/// request, answers, and zeroes what it owns. No record, no file, no artifact, no queue; one log line and
/// one journal line with the outcome and the time — nothing about the document. Available in both
/// processing modes; in <c>PROCESSING_MODE=memory</c> it is the only way a document gets in.
/// </para>
///
/// <para>What "nothing kept" takes here, because each item was a real way to keep something:</para>
/// <list type="bullet">
/// <item><b>The body never goes through ASP.NET's form machinery.</b> <c>IFormFile</c>,
/// <c>Request.Form</c> and <c>ReadFormAsync</c> buffer a file section past 64 KB into
/// <c>ASPNETCORE_*.tmp</c> in the temp directory — a copy of a passport on disk before any code of ours
/// runs. This route reads <c>Request.Body</c> itself into one pooled array, with a hard cap.</item>
/// <item><b>The multipart body is split over that array, not with <c>MultipartReader</c>.</b> The
/// reader's buffered stream rents its own buffers from the shared <see cref="ArrayPool{T}"/> and returns
/// them uncleared, so pieces of the image would stay readable in a pool any later request reuses. A span
/// search over our own buffer copies nothing but the one part, and reads no header into a string — the
/// filename is never materialised at all.</item>
/// <item><b>Buffers we own are zeroed</b> (<see cref="CryptographicOperations.ZeroMemory"/>, then
/// returned with <c>clearArray: true</c>): the body, the image bytes, the decoded pixels, the canvas PNG.
/// The run's own images are zeroed inside the lease (<see cref="PipelineRuntime"/> rule 11).</item>
/// <item><b>Strings cannot be wiped.</b> The recognised text, the base64 of the PNG and the serialised JSON
/// are immutable .NET strings; they are released and the memory reused, not overwritten.</item>
/// <item><b>Nothing about the document reaches a log</b>: not the filename, the type, the size, a library
/// message or an exception's text — only its type name, and the one outcome line.</item>
/// <item><b>The answer is not cacheable</b>: <c>Cache-Control: no-store</c>.</item>
/// </list>
/// </summary>
public sealed partial class ApiServer
{
    /// <summary>Multipart framing on top of the image: a few hundred bytes in practice.</summary>
    private const int MultipartSlack = 64 * 1024;

    private const string DocumentRoutesOffDetail =
        "Document storage is disabled (PROCESSING_MODE=memory): documents are not stored and cannot be " +
        "listed or opened. Recognise with POST /api/v1/recognize.";

    /// <summary>The in-memory journal (§4): process memory only, lost at restart.</summary>
    internal RecognitionJournal Journal { get; } = new(cfg.RecognitionJournalMax);

    /// <summary>
    /// Recognises a decoded image. The runtime in production; tests put a fake here that keeps references
    /// to the buffers it was handed (ports/MEMORY.md §7.2).
    /// </summary>
    internal IImageRecogniser Recogniser { get; set; } = runtime;

    /// <summary>Where the body and image buffers are rented. Tests record what was rented and returned.</summary>
    internal ArrayPool<byte> BufferPool { get; set; } = ArrayPool<byte>.Shared;

    private bool StorageEnabled => Config.ProcessingMode.StorageEnabled(cfg);

    /// <summary>
    /// In <c>PROCESSING_MODE=memory</c> every <c>/documents</c> route answers 404 — **before the body is read
    /// and before authentication**.
    ///
    /// <para>
    /// A middleware, not a check in the handlers: the upload handler reads its body through
    /// <c>ReadFormAsync</c>, which spools a file section past 64 KB to a temporary file, and a check inside
    /// it would run after that. Answering here never touches <c>Request.Body</c>. The cost — an anonymous
    /// caller learns that storage is off — is not a secret worth a file on disk. Port of
    /// <c>_no_document_routes_in_memory_mode</c>.
    /// </para>
    /// </summary>
    internal async Task MemoryModeGate(HttpContext context, Func<Task> next)
    {
        string path = context.Request.Path.Value ?? "";
        if (path.StartsWith($"{Prefix}/documents", StringComparison.Ordinal) && !StorageEnabled)
        {
            context.Response.StatusCode = StatusCodes.Status404NotFound;
            await context.Response.WriteAsJsonAsync(new ApiErrors.ErrorBody(DocumentRoutesOffDetail));
            return;
        }
        await next();
    }

    /// <summary>A refusal with its status, detail and journal outcome.</summary>
    private sealed class Refused(int status, string detail, string outcome = "rejected", int? retryAfter = null)
        : Exception(detail)
    {
        public int Status { get; } = status;
        public string Detail { get; } = detail;
        public string Outcome { get; } = outcome;
        public int? RetryAfter { get; } = retryAfter;
    }

    /// <summary>Recognises one image in memory and answers. Nothing is stored.</summary>
    private async Task<IResult> Recognize(HttpRequest request)
    {
        long started = Stopwatch.GetTimestamp();
        long limit = cfg.MaxUploadBytes;
        bool withImages = QueryFlag(request.Query, "images");
        byte[]? body = null;
        int bodyLength = 0;
        byte[]? image = null;
        int imageLength = 0;
        byte[]? png = null;
        try
        {
            (body, bodyLength) = await ReadBodyCapped(request, limit + MultipartSlack);
            string contentType = request.ContentType ?? "";
            if (contentType.StartsWith("multipart/form-data", StringComparison.OrdinalIgnoreCase))
            {
                byte[] boundary = Boundary(contentType)
                                  ?? throw new Refused(StatusCodes.Status400BadRequest, "Malformed multipart body");
                (int start, int length) = ImagePart(body.AsSpan(0, bodyLength), boundary)
                    ?? throw new Refused(StatusCodes.Status400BadRequest,
                        "No image in the multipart body — send it as the 'file' part");
                image = BufferPool.Rent(length);
                imageLength = length;
                body.AsSpan(start, length).CopyTo(image);
                ReturnZeroed(ref body, bodyLength);
            }
            else
            {
                (image, imageLength, body, bodyLength) = (body, bodyLength, null, 0);
            }

            if (imageLength == 0)
            {
                throw new Refused(StatusCodes.Status400BadRequest, "Empty upload");
            }
            if (imageLength > limit)
            {
                throw new Refused(StatusCodes.Status413PayloadTooLarge,
                    $"File exceeds the {cfg.MaxUploadMB} MB limit");
            }
            RefuseNonImage(image, imageLength);

            var opts = new PipelineRuntime.RecogniseOptions
            {
                Docconf = SettingsSchema.TypedFloat("docconf", settings.SettingValue(db, "docconf"), cfg.Docconf),
                ImgSize = SettingsSchema.TypedInt("img_size", settings.SettingValue(db, "img_size"), cfg.ImgSize),
                LeaseTimeout = TimeSpan.FromSeconds(Math.Max(1, cfg.RecognizeWaitSec)),
            };
            TimeSpan budget = TimeSpan.FromSeconds(SettingsSchema.TypedInt("job_timeout_sec",
                settings.SettingValue(db, "job_timeout_sec"), cfg.JobTimeoutSec) + cfg.RecognizeWaitSec);

            // The image buffer is HANDED OVER: from here the recognition task zeroes and returns it, in its
            // own finally, whether it finishes in time or not — zeroing it here while an abandoned decode
            // still reads it would be a race.
            byte[] handed = image;
            int handedLength = imageLength;
            (image, imageLength) = (null, 0);

            // Its own thread, not the background worker's: a request must not queue behind a stored
            // document. The pipeline pool is still shared, and the lease is what serialises access to it.
            Task<MemoryRecognition> work = Task.Factory.StartNew(
                () => RecogniseBytes(handed, handedLength, opts, withImages),
                CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

            if (await Task.WhenAny(work, Task.Delay(budget)) != work)
            {
                // Native code cannot be cancelled. When the abandoned run finishes, its PNG is zeroed.
                _ = work.ContinueWith(finished =>
                {
                    if (finished.IsCompletedSuccessfully && finished.Result.Png is { } late)
                    {
                        CryptographicOperations.ZeroMemory(late);
                    }
                }, TaskScheduler.Default);
                throw new Refused(StatusCodes.Status504GatewayTimeout, "Recognition took too long", "error");
            }

            MemoryRecognition got;
            try
            {
                got = await work;
            }
            catch (ServiceException busy) when (busy.Kind == ErrorKind.PipelineBusy)
            {
                throw new Refused(StatusCodes.Status503ServiceUnavailable,
                    "All recognition pipelines are busy — retry shortly", "busy", 5);
            }
            catch (ServiceException notReady) when (notReady.Kind == ErrorKind.RuntimeNotReady)
            {
                throw new Refused(StatusCodes.Status503ServiceUnavailable,
                    "The recognition runtime is not ready yet", "busy", 10);
            }
            png = got.Png;

            int ms = ElapsedMs(started);
            Dictionary<string, object?> content = RecognitionBody(got.ViewModel, png, ms);
            Finish(got.ViewModel.Recognised ? "recognised" : "not_recognised", StatusCodes.Status200OK, started);
            request.HttpContext.Response.Headers.CacheControl = "no-store";
            return Results.Json(content);
        }
        catch (Refused refused)
        {
            Finish(refused.Outcome, refused.Status, started);
            if (refused.RetryAfter is int seconds)
            {
                request.HttpContext.Response.Headers.RetryAfter = seconds.ToString(
                    System.Globalization.CultureInfo.InvariantCulture);
            }
            return Results.Json(new ApiErrors.ErrorBody(refused.Detail), statusCode: refused.Status);
        }
        catch (Exception error)
        {
            Finish("error", StatusCodes.Status500InternalServerError, started);
            // The TYPE only: a message carries paths and values.
            log.LogWarning("[API] recognize failed: {Type}", error.GetType().Name);
            return Results.Json(new ApiErrors.ErrorBody("Recognition failed"),
                statusCode: StatusCodes.Status500InternalServerError);
        }
        finally
        {
            ReturnZeroed(ref body, bodyLength);
            ReturnZeroed(ref image, imageLength);
            if (png is not null)
            {
                CryptographicOperations.ZeroMemory(png);
            }
        }
    }

    /// <summary>The magic-byte sniffing the document upload uses: a PDF and a non-image are both 415.</summary>
    private static void RefuseNonImage(byte[] image, int length)
    {
        ReadOnlySpan<byte> bytes = image.AsSpan(0, length);
        if (Artifacts.IsPdf(bytes))
        {
            throw new Refused(StatusCodes.Status415UnsupportedMediaType,
                "PDF is not supported — send a JPEG, PNG, WEBP, BMP or TIFF image");
        }
        if (Artifacts.SniffImage(bytes) is null)
        {
            throw new Refused(StatusCodes.Status415UnsupportedMediaType,
                "Unsupported file type — expected an image");
        }
    }

    /// <summary>
    /// Decodes, recognises, zeroes. Runs on the recognition thread; every pixel buffer it creates is zeroed
    /// here, in <c>finally</c>, and so is the image buffer it was handed — even when recognition fails.
    /// </summary>
    private MemoryRecognition RecogniseBytes(byte[] image, int length, PipelineRuntime.RecogniseOptions opts,
        bool withCanvas)
    {
        Mat? bgr = null;
        Image? rgb = null;
        try
        {
            try
            {
                bgr = Cv2.ImDecode(image.AsSpan(0, length), ImreadModes.Color);
            }
            catch (OpenCVException)
            {
                bgr = null;
            }
            if (bgr is null || bgr.Empty())
            {
                throw new Refused(StatusCodes.Status422UnprocessableEntity,
                    "The image could not be decoded — it may be corrupt");
            }
            var converted = new Mat();
            rgb = Image.Wrap(converted);
            Cv2.CvtColor(bgr, converted, ColorConversionCodes.BGR2RGB);
            bgr.SetTo(OpenCvSharp.Scalar.All(0));
            return Recogniser.RecogniseImage(rgb, opts, withCanvas);
        }
        finally
        {
            if (bgr is not null)
            {
                if (!bgr.IsDisposed && !bgr.Empty())
                {
                    bgr.SetTo(OpenCvSharp.Scalar.All(0));
                }
                bgr.Dispose();
            }
            rgb?.Zero();
            rgb?.Dispose();
            byte[]? owned = image;
            ReturnZeroed(ref owned, length);
        }
    }

    /// <summary>
    /// The 200 body: the same transform the stored path uses (<see cref="Worker.SearchText.ToNode"/>, the
    /// blob a stored document's detail is served from), without anything that points at a stored thing —
    /// there is none. Keys and their order are the reference's <c>_response_body</c>.
    /// </summary>
    private static Dictionary<string, object?> RecognitionBody(Payload payload, byte[]? png, int ms)
    {
        JsonObject result = Worker.SearchText.ToNode(payload) as JsonObject ?? [];

        var canvas = new JsonObject();
        if (result["canvas"] is JsonObject stored)
        {
            foreach ((string key, JsonNode? value) in stored)
            {
                canvas[key] = value?.DeepClone();
            }
        }
        if (png is not null)
        {
            canvas["image_png_base64"] = Convert.ToBase64String(png);
        }

        // Same split as a stored document's row: DocConf on its own, the rest as quality.
        JsonNode? docConf = null;
        var quality = new JsonObject();
        if (result["quality"] is JsonObject q)
        {
            foreach ((string key, JsonNode? value) in q)
            {
                if (key == "DocConf")
                {
                    docConf = value?.DeepClone();
                    continue;
                }
                quality[key] = value?.DeepClone();
            }
        }

        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["stored"] = false,
            ["doc_type"] = result["doc_type"]?.DeepClone(),
            ["doc_type_base"] = result["doc_type_base"]?.DeepClone(),
            ["doc_type_era"] = result["doc_type_era"]?.DeepClone(),
            ["recognised"] = result["recognised"]?.DeepClone(),
            ["doc_conf"] = docConf,
            ["device"] = result["device"]?.DeepClone(),
            ["canvas"] = canvas,
            ["coord_space"] = result["coord_space"]?.DeepClone(),
            ["coord_space_note"] = result["coord_space_note"]?.DeepClone(),
            ["boxes"] = OrEmptyArray(result["boxes"]),
            ["fields"] = OrEmptyArray(result["fields"]),
            ["ocr"] = OrEmptyObject(result["ocr"]),
            ["quality"] = quality,
            ["address"] = result["address"]?.DeepClone(),
            ["timings"] = OrEmptyObject(result["timings"]),
            ["processing_ms"] = ms,
        };
    }

    /// <summary>The one journal line and the one log line of a recognition: outcome, status, time.</summary>
    private void Finish(string outcome, int status, long started)
    {
        int ms = ElapsedMs(started);
        Journal.Record(outcome, status, ms);
        log.LogInformation("[API] recognize: outcome={Outcome} status={Status} ms={Ms}", outcome, status, ms);
    }

    private static int ElapsedMs(long started) => (int)Stopwatch.GetElapsedTime(started).TotalMilliseconds;

    /// <summary>The journal: when, how it ended, how long. Nothing else exists to show.</summary>
    private IResult ListRecognitions(HttpRequest request)
    {
        int limit = QueryParams.Int(request.Query, "limit", 100, 1, 1000);
        int offset = QueryParams.Int(request.Query, "offset", 0, 0, 0);
        (List<RecognitionJournal.Entry> page, int total) = Journal.Recent(limit, offset);
        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["items"] = page.Select(e => e.Public()).ToList(),
            ["total"] = total,
            ["counts"] = Journal.Counts(),
        });
    }

    /// <summary>
    /// The whole request body into one pooled array, refusing more than <paramref name="cap"/> bytes — before
    /// reading when <c>Content-Length</c> already says so, and as soon as the count passes it otherwise.
    /// A grown buffer's old array is zeroed before it goes back to the pool.
    /// </summary>
    private async Task<(byte[] Buffer, int Length)> ReadBodyCapped(HttpRequest request, long cap)
    {
        // The server-wide body limit is the document upload's (MAX_UPLOAD_MB); this route allows the
        // multipart framing on top, and enforces its own cap below.
        IHttpMaxRequestBodySizeFeature? limitFeature =
            request.HttpContext.Features.Get<IHttpMaxRequestBodySizeFeature>();
        if (limitFeature is { IsReadOnly: false })
        {
            limitFeature.MaxRequestBodySize = cap + 1;
        }
        long? declared = request.ContentLength;
        if (declared > cap)
        {
            // Refused before a byte of it is kept. The rest is read and DISCARDED so the client receives
            // the 413: Kestrel resets a connection whose unread body is over its limit, and the client,
            // still sending, then sees a reset instead of the answer.
            await DiscardBody(request, limitFeature, declared.Value);
            throw new Refused(StatusCodes.Status413PayloadTooLarge, "Image exceeds the upload limit");
        }

        byte[]? buffer = BufferPool.Rent((int)(declared ?? 64 * 1024));
        int length = 0;
        try
        {
            while (true)
            {
                if (length == buffer.Length)
                {
                    if (declared == length)
                    {
                        break; // everything announced has arrived
                    }
                    byte[] bigger = BufferPool.Rent((int)Math.Min(Math.Max(1024L, buffer.LongLength * 2), cap + 1));
                    buffer.AsSpan(0, length).CopyTo(bigger);
                    ReturnZeroed(ref buffer, length);
                    buffer = bigger;
                }
                int read;
                try
                {
                    read = await request.Body.ReadAsync(buffer.AsMemory(length));
                }
                catch (BadHttpRequestException tooLarge)
                    when (tooLarge.StatusCode == StatusCodes.Status413PayloadTooLarge)
                {
                    throw new Refused(StatusCodes.Status413PayloadTooLarge, "Image exceeds the upload limit");
                }
                if (read == 0)
                {
                    break;
                }
                length += read;
                if (length > cap)
                {
                    ReturnZeroed(ref buffer, length);
                    await DiscardBody(request, limitFeature, null);
                    throw new Refused(StatusCodes.Status413PayloadTooLarge, "Image exceeds the upload limit");
                }
            }
            return (buffer, length);
        }
        catch
        {
            ReturnZeroed(ref buffer, length);
            throw;
        }
    }

    /// <summary>
    /// Past this the rest of an oversized body is not read at all: the connection is let go instead.
    /// </summary>
    private const long DiscardLimit = 256L * 1024 * 1024;

    /// <summary>
    /// Reads what is left of a refused body through one small scratch buffer and throws it away — nothing
    /// is kept, and the scratch is zeroed. Bounded by <see cref="DiscardLimit"/>; a reset there is the
    /// client's problem, not a reason to read a gigabyte.
    /// </summary>
    private static async Task DiscardBody(HttpRequest request, IHttpMaxRequestBodySizeFeature? limitFeature,
        long? declared)
    {
        if (declared > DiscardLimit)
        {
            return;
        }
        if (limitFeature is { IsReadOnly: false })
        {
            limitFeature.MaxRequestBodySize = DiscardLimit;
        }
        byte[] scratch = new byte[16 * 1024];
        try
        {
            long total = 0;
            int read;
            while (total <= DiscardLimit && (read = await request.Body.ReadAsync(scratch)) > 0)
            {
                total += read;
            }
        }
        catch (Exception ex) when (ex is BadHttpRequestException or IOException or InvalidOperationException)
        {
            // The client went away, or the body is over even this limit: nothing more to do.
        }
        finally
        {
            CryptographicOperations.ZeroMemory(scratch);
        }
    }

    /// <summary>Zeroes the used part of a pooled array and returns it cleared; nulls the reference.</summary>
    private void ReturnZeroed(ref byte[]? buffer, int length)
    {
        if (buffer is null)
        {
            return;
        }
        CryptographicOperations.ZeroMemory(buffer.AsSpan(0, Math.Min(length, buffer.Length)));
        if (buffer.Length > 0)
        {
            BufferPool.Return(buffer, clearArray: true);
        }
        buffer = null;
    }

    /// <summary>The boundary of a <c>multipart/form-data</c> content type, or null.</summary>
    internal static byte[]? Boundary(string contentType)
    {
        foreach (string part in contentType.Split(';').Skip(1))
        {
            string trimmed = part.Trim();
            int eq = trimmed.IndexOf('=');
            if (eq <= 0)
            {
                continue;
            }
            if (trimmed[..eq].Trim().Equals("boundary", StringComparison.OrdinalIgnoreCase))
            {
                string value = trimmed[(eq + 1)..].Trim().Trim('"');
                return value.Length > 0 ? Encoding.Latin1.GetBytes(value) : null;
            }
        }
        return null;
    }

    private static readonly byte[] Crlf2 = "\r\n\r\n"u8.ToArray();

    /// <summary>
    /// Where the image part lies in a multipart body: the first part with a <c>filename</c>, or the part
    /// named <c>file</c>. Offsets into <paramref name="body"/>; the caller copies the part out. The headers
    /// are searched as bytes, case-insensitively, and never turned into a string. Port of
    /// <c>_image_from_multipart</c>.
    /// </summary>
    internal static (int Start, int Length)? ImagePart(ReadOnlySpan<byte> body, byte[] boundary)
    {
        byte[] delimiter = [.. "--"u8, .. boundary];
        byte[] nextDelimiter = [.. "\r\n"u8, .. delimiter];
        int pos = body.IndexOf(delimiter);
        while (pos != -1)
        {
            int start = pos + delimiter.Length;
            if (body.Length >= start + 2 && body.Slice(start, 2).SequenceEqual("--"u8))
            {
                break;
            }
            int headerEnd = IndexFrom(body, Crlf2, start);
            if (headerEnd == -1)
            {
                break;
            }
            ReadOnlySpan<byte> headers = body[start..headerEnd];
            int dataStart = headerEnd + 4;
            int next = IndexFrom(body, nextDelimiter, dataStart);
            if (next == -1)
            {
                break;
            }
            if (ContainsAsciiIgnoreCase(headers, "filename="u8) || ContainsAsciiIgnoreCase(headers, "name=\"file\""u8))
            {
                return (dataStart, next - dataStart);
            }
            pos = next + 2;
        }
        return null;
    }

    private static int IndexFrom(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle, int from)
    {
        if (from > haystack.Length)
        {
            return -1;
        }
        int found = haystack[from..].IndexOf(needle);
        return found == -1 ? -1 : from + found;
    }

    /// <summary><paramref name="needle"/> must be lower-case ASCII.</summary>
    private static bool ContainsAsciiIgnoreCase(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle)
    {
        for (int i = 0; i + needle.Length <= haystack.Length; i++)
        {
            int j = 0;
            while (j < needle.Length)
            {
                byte c = haystack[i + j];
                if (c is >= (byte)'A' and <= (byte)'Z')
                {
                    c = (byte)(c + 32);
                }
                if (c != needle[j])
                {
                    break;
                }
                j++;
            }
            if (j == needle.Length)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary><c>?images=1</c> / <c>true</c> / <c>yes</c> / <c>on</c>; anything else is false.</summary>
    private static bool QueryFlag(IQueryCollection query, string name) =>
        QueryParams.Str(query, name).Trim().ToLowerInvariant() is "1" or "true" or "yes" or "on";
}
