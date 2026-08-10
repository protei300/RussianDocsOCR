using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Logging;
using RussianDocs.Service.Ml;
using RussianDocs.Service.Model;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Settings;
using RussianDocs.Service.Store;
using Results = Microsoft.AspNetCore.Http.Results;

namespace RussianDocs.Service.Api;

/// <summary>Auth, API keys, settings, logs, status and health.</summary>
public sealed partial class ApiServer
{
    // --- auth ---------------------------------------------------------------

    /// <summary>
    /// Exchanges the PIN for a session JWT.
    ///
    /// <para>
    /// The failure message deliberately does not distinguish "wrong PIN" from "malformed request":
    /// there is nothing useful for a legitimate user in the difference, and there is something useful
    /// in it for somebody guessing.
    /// </para>
    /// </summary>
    private IResult PinLogin(HttpRequest request)
    {
        try
        {
            JsonNode? body = JsonNode.Parse(ReadBody(request));
            string pin = body?["pin"]?.GetValue<string>() ?? "";
            if (!Tokens.VerifyPin(AuthConfig, pin))
            {
                // Logged, because repeated failures are the only signal available without rate
                // limiting — see the note in Tokens about what a PIN is and is not.
                log.LogWarning("[API] PIN login rejected");
                return Results.Json(new ApiErrors.ErrorBody("Incorrect PIN"),
                    statusCode: StatusCodes.Status401Unauthorized);
            }
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException)
        {
            return ApiErrors.Write(ServiceException.BadRequest("expected {\"pin\": \"...\"}"), log);
        }

        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["access_token"] = Tokens.CreateAccessToken(AuthConfig, "operator"),
            ["token_type"] = "bearer",
            ["user"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["name"] = Identity.Session.Name,
                ["role"] = Identity.Session.Role,
            },
        });
    }

    /// <summary>
    /// Reads the whole request body as text.
    ///
    /// <para>
    /// **The read is ASYNC and then waited on, not synchronous.** Kestrel sets
    /// <c>AllowSynchronousIO = false</c> by default, so <c>StreamReader.ReadToEnd</c> throws
    /// <see cref="InvalidOperationException"/> — which this file's own catch turned into a 400, so
    /// every PIN login failed with "expected a pin" and the login page just said the PIN was wrong.
    /// The bodies here are a few dozen bytes and the handlers are synchronous to match the Go port's
    /// shape, so blocking on the completed task is the honest trade; the alternative is a second,
    /// async copy of the whole guard chain for three endpoints that read a body.
    /// </para>
    /// </summary>
    private static string ReadBody(HttpRequest request)
    {
        using var reader = new StreamReader(request.Body);
        return reader.ReadToEndAsync().GetAwaiter().GetResult();
    }

    // --- api keys -----------------------------------------------------------

    /// <summary>
    /// Rendered verbatim by the keys page.
    ///
    /// <para>
    /// Surfaced so the UI can WARN rather than letting a restart quietly delete a key somebody pasted
    /// into a config somewhere. The text is copied from the reference so both services say the same
    /// thing.
    /// </para>
    /// </summary>
    private const string EphemeralKeyNote =
        "Keys created here live in ephemeral storage and are lost when the service restarts. " +
        "The default key comes from the environment and always exists.";

    private IResult ListKeys() => Results.Json(
        new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["items"] = ApiKeys.Public(db, AuthConfig),
            // api-keys/Index.vue reads `res.note` and renders it as a banner. Omitting it left an
            // empty warning div on the page.
            ["note"] = EphemeralKeyNote,
        });

    /// <summary>Mints a key and returns the PLAINTEXT exactly once.</summary>
    private IResult CreateKey(HttpRequest request)
    {
        string label = "";
        try
        {
            // A missing body is fine — the label is optional and defaults. Erroring here would make
            // "create a key" require a payload for no reason.
            label = JsonNode.Parse(ReadBody(request))?["label"]?.GetValue<string>() ?? "";
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException)
        {
            // deliberately ignored, see above
        }

        (ApiKey record, string plaintext) = ApiKeys.Create(db, label);
        log.LogInformation("[API] api key created: id={Id} label={Label}", record.Id, record.Label);

        Dictionary<string, object?> output = record.Public();
        // The ONLY response that ever carries the full key. After this it exists nowhere but the
        // caller's hands and a sha256 in the store.
        output["key"] = plaintext;
        output["warning"] = "Copy this key now — it will not be shown again.";
        return Results.Json(output, statusCode: StatusCodes.Status201Created);
    }

    /// <summary>
    /// Refuses to delete the default key.
    ///
    /// <para>
    /// 409, not 403: the request is well-formed and the caller is allowed to delete keys — it is the
    /// STATE that forbids this one. Deleting it would also be silently undone by the next restart,
    /// since it is derived from the environment rather than stored.
    /// </para>
    /// </summary>
    private IResult DeleteKey(int id)
    {
        if (id == ApiKeys.DefaultKeyId)
        {
            throw ServiceException.Conflict(
                "The default key comes from the environment and cannot be deleted");
        }
        if (!ApiKeys.Delete(db, id))
        {
            throw ServiceException.NotFound("Key not found");
        }
        return Results.NoContent();
    }

    // --- settings -----------------------------------------------------------

    private IResult GetSettings() => Results.Json(
        new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["schema"] = SettingsSchema.All,
            ["values"] = settings.AllSettings(db),
        });

    /// <summary>
    /// Validates and stores, reporting which changes need a restart.
    ///
    /// <para>
    /// <c>restart_required</c> is not decoration: <c>compute_device</c> and <c>ocr_mode</c> are baked
    /// into the pipeline at construction, so a UI that reported "saved" and left the runtime alone
    /// would be lying about something an operator can verify on the status page.
    /// </para>
    /// </summary>
    private IResult PutSettings(HttpRequest request)
    {
        // **The body is WRAPPED: {"values": {...}}.** settings/Index.vue posts
        // `Api.put('/settings', { values })`, and the reference declares a SettingsUpdate model with a
        // single `values` field. An earlier version of the Go port parsed the object FLAT, which was
        // the worst possible failure: `values` is not a schema key, so the whitelist dropped it,
        // nothing was stored, and the page reported success. Exactly the "reports saved while
        // discarding the value" outcome the settings layer is written to avoid.
        JsonObject? values;
        try
        {
            values = JsonNode.Parse(ReadBody(request))?["values"] as JsonObject;
        }
        catch (JsonException)
        {
            values = null;
        }
        if (values is null)
        {
            throw ServiceException.BadRequest("expected {\"values\": {...}}");
        }

        var incoming = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach ((string key, JsonNode? value) in values)
        {
            // Coerce takes the raw scalar; a JsonNode's own ToString would wrap a string in quotes and
            // turn "cpu" into "\"cpu\"", which then fails the choice check for a value that was fine.
            incoming[key] = value is JsonValue scalar ? Scalar(scalar) : value?.ToJsonString();
        }

        (Dictionary<string, string> stored, List<string> restart) =
            settings.BulkUpdate(db, incoming);
        if (restart.Count > 0)
        {
            log.LogInformation("[API] settings changed, restart required: {Keys}",
                string.Join(", ", restart));
        }
        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["values"] = stored,
            // The schema travels back with the values, matching the reference, so a client that only
            // ever calls PUT still has everything it needs to render the form.
            ["schema"] = SettingsSchema.All,
            // An empty ARRAY, not null: the page assigns it straight to a list it iterates.
            ["restart_required"] = restart,
        });
    }

    private static object? Scalar(JsonValue value) =>
        value.TryGetValue(out string? text) ? text
        : value.TryGetValue(out bool flag) ? flag
        : value.TryGetValue(out double number) ? number
        : value.ToJsonString();

    // --- logs ---------------------------------------------------------------

    /// <summary>
    /// Serves the ring buffer.
    ///
    /// <para>
    /// **The response key is <c>entries</c> and the count parameter is <c>n</c>.** Both are fixed by
    /// the shared frontend: logs/Index.vue sends <c>{ n: 400 }</c> and reads <c>res.entries</c>. An
    /// earlier version of the Go port returned <c>{"items": …}</c> and accepted <c>limit</c>, which
    /// produced a valid response the page could not read — an EMPTY logs page with a 200 and no error
    /// anywhere. The same class of mistake as the status block: when the UI is shared, the UI owns the
    /// wire format.
    /// </para>
    ///
    /// <para>
    /// <c>count</c> is sent alongside because the reference sends it. The page derives its "N lines"
    /// label from the array length, so nothing reads it today — but a client that trusted the
    /// documented shape would.
    /// </para>
    /// </summary>
    private IResult Logs(HttpRequest request)
    {
        // Bounds copied from the reference (1..2000), not from the buffer capacity: asking for more
        // than the buffer holds is not an error, it just returns everything there is.
        int n = QueryParams.Int(request.Query, "n", 200, 1, 2000);
        List<LogEntry> entries = LogRing.Recent(n, QueryParams.Str(request.Query, "level"),
            QueryParams.Str(request.Query, "search"));
        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["count"] = entries.Count,
            ["entries"] = entries,
        });
    }

    // --- status -------------------------------------------------------------

    /// <summary>
    /// Reports what the service is actually doing.
    ///
    /// <para>
    /// **The field names are fixed by the SHARED FRONTEND.** <c>web/</c> is reused unchanged by every
    /// port, so status/Index.vue is the contract — it reads <c>server.cpu_pct</c>,
    /// <c>gpu.vram_used_gb</c>, <c>service.data_is_ephemeral</c> and the rest BY NAME. An earlier
    /// version of the Go handler returned a thinner block, and the status page rendered completely
    /// empty.
    /// </para>
    ///
    /// <para>
    /// <c>device</c> and <c>ocr_device</c> come through SEPARATELY on purpose: with GPU detectors the
    /// OCR engines still run on CPU, and a page that just says "GPU active" invites a bug report the
    /// first time somebody watches nvidia-smi during recognition.
    /// </para>
    /// </summary>
    private IResult Status()
    {
        StoreStats stats = Documents.Stats(db);
        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["server"] = SysInfo.ReadServer(),
            // null when there is no GPU, no driver, or a CPU-only container. The status page then shows
            // the compute block alone, which is the part that answers whether the GPU is being used at
            // all.
            ["gpu"] = SysInfo.ReadGpu(),
            ["compute"] = runtime.Info(),
            ["service"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["uptime_sec"] = (int)Stopwatch.GetElapsedTime(_startedTicks).TotalSeconds,
                ["version"] = cfg.GitCommit,
                ["documents_queued"] = stats.Queued,
                ["documents_processing"] = stats.Processing,
                ["documents_done"] = stats.Done,
                ["documents_failed"] = stats.Failed,
                ["documents_total"] = stats.Total,
                ["recognised"] = stats.Recognised,
                ["avg_processing_ms"] = stats.AvgProcessingMs,
                ["data_dir_mb"] = Round1(db.DiskUsageBytes() / 1e6),
                // The SPA reads this from `service`, not from `storage`. The Python service puts it
                // only under `storage`, so its own status page always renders "Retained" — a real
                // defect on that side, recorded in the progress log rather than reproduced here.
                ["data_is_ephemeral"] = db.IsEphemeral,
            },
            // Which backend is live, so an operator can tell at a glance whether what they are looking
            // at survives a restart.
            ["storage"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["backend"] = db.Backend,
                ["ephemeral"] = db.IsEphemeral,
            },
        });
    }

    /// <summary>
    /// The container healthcheck. No auth, no store access, no runtime dependency.
    ///
    /// <para>
    /// It reports OK while the models are still loading, deliberately: the service IS healthy then — it
    /// accepts uploads and queues them. Gating health on the runtime would make Docker kill the
    /// container during the fifteen seconds it needs to start.
    /// </para>
    /// </summary>
    private IResult Health() => Results.Json(
        new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["status"] = "ok",
            ["runtime"] = runtime.Info().State,
        });
}
