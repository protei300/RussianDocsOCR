using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using NUnit.Framework;
using RussianDocs.Service.Api;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Ml;
using RussianDocs.Service.Model;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Store;
using RussianDocs.Service.Worker;

namespace RussianDocs.Service.Tests;

/// <summary>
/// THE WIRE CONTRACT OF THE OPERATOR PAGES.
///
/// <para>
/// These tests exist because the same mistake was made twice in the Go port, and both times the
/// failure was silent: a 200 response, well-formed JSON, no error anywhere, and a page that rendered
/// completely empty.
/// </para>
///
/// <para>
/// **<c>web/</c> is reused UNCHANGED by every port, so the SPA owns the wire format.** The key lists
/// below are transcribed from the Vue sources named in each test; when a page starts reading a new
/// field, this test is where that gets noticed rather than in a browser.
/// </para>
///
/// <para>
/// Deliberately asserting KEYS and not values: the values are host-dependent and the point is the
/// shape. The recognition runtime is deliberately left UNINITIALISED — none of this needs a model,
/// and a test that loads 215 MB of weights to check a JSON key is a test nobody runs.
/// </para>
/// </summary>
[TestFixture]
public sealed class ContractTests
{
    private WebApplication _app = null!;
    private HttpClient _client = null!;
    private string _dataDir = null!;
    private string _webRoot = null!;
    private const string ApiKey = "rdk_contract_test_key";

    [OneTimeSetUp]
    public void StartServer()
    {
        _dataDir = Path.Combine(Path.GetTempPath(), "rdocs-contract-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dataDir);

        // A STAND-IN FRONTEND, and the fixture used to pass null here instead — which meant the
        // SPA route was never exercised at all. That gap shipped a real defect: the parameterless
        // MapFallback() carries a `nonfile` route constraint, so every asset 404'd while `/` still
        // returned index.html, and the page rendered blank with a 200 and no server-side error. A
        // suite that cannot fail on the SPA is not testing the SPA.
        _webRoot = Path.Combine(_dataDir, "webroot");
        Directory.CreateDirectory(Path.Combine(_webRoot, "assets"));
        File.WriteAllText(Path.Combine(_webRoot, "index.html"),
            "<!DOCTYPE html><html><body>shell</body></html>");
        File.WriteAllText(Path.Combine(_webRoot, "assets", "index-abc123.js"),
            "console.log('bundle')");
        File.WriteAllText(Path.Combine(_webRoot, "assets", "index-abc123.css"), ":root{}");

        var cfg = new Config.Settings
        {
            DataDir = _dataDir,
            DataWipeOnStart = false,
            DefaultApiKey = ApiKey,
            JwtSecret = "test-secret",
            // Negative disables seeding: these tests insert exactly the rows they need, and a
            // seeded corpus would make "seven rows" an assertion about the fixtures.
            SeedSamples = -1,
        };

        ILogger log = NullLogger.Instance;
        var db = new FileStore(_dataDir, log);
        var runtime = new PipelineRuntime(log);
        var settings = new SettingsRepository(cfg, log);
        var worker = new RecognitionWorker(db, runtime, cfg, settings, log);

        var builder = WebApplication.CreateBuilder();
        builder.Logging.ClearProviders();
        // Port 0: the OS assigns one, so a developer's own service on 8004 cannot make this fail.
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        _app = builder.Build();
        new ApiServer(db, runtime, worker, cfg, settings, _webRoot, log).MapRoutes(_app);
        _app.Start();

        string address = _app.Services
            .GetRequiredService<Microsoft.AspNetCore.Hosting.Server.IServer>()
            .Features.Get<Microsoft.AspNetCore.Hosting.Server.Features.IServerAddressesFeature>()!
            .Addresses.First();
        // **UseProxy = false, and it is not optional.** HttpClient honours the system proxy even
        // for 127.0.0.1, so on a machine behind a corporate proxy every request here left the box
        // and came back as a 403 with an HTML body — twenty-three failures whose only symptom was
        // "'<' is an invalid start of a value". Same trap the Python verification script hits, and
        // the same fix: bypass the proxy for loopback explicitly rather than relying on NO_PROXY
        // being set in whatever environment the tests run in.
        _client = new HttpClient(new HttpClientHandler { UseProxy = false })
        {
            BaseAddress = new Uri(address + "/api/v1/"),
        };

        Db = db;
    }

    private static FileStore Db = null!;

    [OneTimeTearDown]
    public void StopServer()
    {
        _client.Dispose();
        _app.StopAsync().GetAwaiter().GetResult();
        _app.DisposeAsync().GetAwaiter().GetResult();
        try { Directory.Delete(_dataDir, recursive: true); } catch { /* temp dir */ }
    }

    /// <summary>
    /// Mints a real session, because the operator routes are session-only and going through the
    /// actual guard is part of what is being checked.
    /// </summary>
    private string SessionToken()
    {
        HttpResponseMessage response = Post("auth/pin-login", """{"pin":"1234"}""", key: false);
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        return Json(response)["access_token"]!.GetValue<string>();
    }

    private HttpResponseMessage Get(string path, string? token = null, bool key = true)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, path);
        Authorise(request, token, key);
        return _client.Send(request);
    }

    private HttpResponseMessage Post(string path, string? body = null, string? token = null,
        bool key = true)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, path);
        if (body is not null)
        {
            request.Content = new StringContent(body, Encoding.UTF8, "application/json");
        }
        Authorise(request, token, key);
        return _client.Send(request);
    }

    private HttpResponseMessage Send(HttpMethod method, string path, string? body = null,
        string? token = null, bool key = true)
    {
        var request = new HttpRequestMessage(method, path);
        if (body is not null)
        {
            request.Content = new StringContent(body, Encoding.UTF8, "application/json");
        }
        Authorise(request, token, key);
        return _client.Send(request);
    }

    private static void Authorise(HttpRequestMessage request, string? token, bool key)
    {
        if (token is not null)
        {
            request.Headers.Add("Authorization", "Bearer " + token);
        }
        else if (key)
        {
            request.Headers.Add("X-API-Key", ApiKey);
        }
    }

    private static JsonObject Json(HttpResponseMessage response) =>
        (JsonObject)JsonNode.Parse(
            response.Content.ReadAsStringAsync().GetAwaiter().GetResult())!;

    private static string Text(HttpResponseMessage response) =>
        response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

    // -- status --------------------------------------------------------------

    /// <summary>Field names transcribed from <c>web/src/views/pages/status/Index.vue</c>.</summary>
    [Test]
    public void StatusCarriesEveryBlockTheSpaReads()
    {
        JsonObject body = Json(Get("status", SessionToken()));

        Assert.That(body.Select(p => p.Key), Is.SupersetOf(
            new[] { "server", "gpu", "compute", "service", "storage" }));

        var server = (JsonObject)body["server"]!;
        Assert.That(server.Select(p => p.Key), Is.SupersetOf(new[]
        {
            "cpu_pct", "cpu_name", "cpu_cores", "cpu_threads",
            "ram_used_gb", "ram_total_gb", "disk_used_gb", "disk_total_gb",
        }));

        var compute = (JsonObject)body["compute"]!;
        Assert.That(compute.Select(p => p.Key), Is.SupersetOf(new[]
        {
            "state", "providers", "device", "ocr_device", "model_format", "ocr_mode",
            "requested_device", "fell_back", "load_ms", "warmup_ms", "pool_size",
            "pool_available",
        }));

        var service = (JsonObject)body["service"]!;
        Assert.That(service.Select(p => p.Key), Is.SupersetOf(new[]
        {
            "uptime_sec", "version", "documents_queued", "documents_processing",
            "documents_done", "documents_failed", "documents_total", "recognised",
            "avg_processing_ms", "data_dir_mb",
            // The SPA reads ephemerality from `service`, not from `storage`. The Python service
            // puts it only under `storage`, so its own status page always renders "Retained".
            "data_is_ephemeral",
        }));
    }

    // -- logs ----------------------------------------------------------------

    /// <summary>
    /// Field names transcribed from <c>web/src/views/pages/logs/Index.vue</c>, which sends
    /// <c>{ n: 400 }</c> and reads <c>res.entries</c>.
    /// </summary>
    [Test]
    public void LogsUseEntriesAndN()
    {
        JsonObject body = Json(Get("logs?n=5", SessionToken()));
        Assert.That(body.Select(p => p.Key), Is.SupersetOf(new[] { "count", "entries" }));
        Assert.That(body["entries"], Is.InstanceOf<JsonArray>());
    }

    /// <summary>
    /// <c>n</c> is the parameter the page sends. An implementation that only understood
    /// <c>limit</c> returned its default and looked like it worked.
    /// </summary>
    [Test]
    public void LogsRespectN()
    {
        string token = SessionToken();
        // A handful of lines exist by now from earlier tests and the server's own startup.
        JsonObject body = Json(Get("logs?n=1", token));
        Assert.That(((JsonArray)body["entries"]!).Count, Is.LessThanOrEqualTo(1));
    }

    // -- authorisation -------------------------------------------------------

    /// <summary>
    /// The operator pages are session-only: an API key must NOT open them, because an integration
    /// has no business reading logs or managing settings.
    /// </summary>
    [TestCase("logs")]
    [TestCase("settings")]
    [TestCase("status")]
    [TestCase("api-keys")]
    public void OperatorPagesRejectApiKeys(string path)
    {
        Assert.That(Get(path).StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized));
    }

    /// <summary>
    /// A missing credential is 401, never 403: the SPA redirects to the PIN screen on 401 only.
    /// </summary>
    [Test]
    public void MissingCredentialIsUnauthorizedNotForbidden()
    {
        HttpResponseMessage response = Get("documents", key: false);
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized));
        Assert.That(response.Headers.WwwAuthenticate.ToString(), Does.Contain("Bearer"));
    }

    /// <summary>
    /// The error body is <c>{"detail": "&lt;string&gt;"}</c> everywhere — the SPA's fetch wrapper
    /// reads <c>detail</c> and nothing else.
    /// </summary>
    [Test]
    public void ErrorBodyIsAStringDetail()
    {
        foreach (HttpResponseMessage response in new[]
                 {
                     Get("documents", key: false),
                     Get("documents/99999"),
                     Get("documents/not-a-number"),
                 })
        {
            JsonObject body = Json(response);
            Assert.That(body["detail"], Is.InstanceOf<JsonValue>(), Text(response));
            Assert.That(body["detail"]!.GetValue<string>(), Is.Not.Empty);
        }
    }

    // -- api keys ------------------------------------------------------------

    /// <summary>
    /// The keys page renders <c>res.note</c> as a banner. Omitting it left an empty warning div.
    /// </summary>
    [Test]
    public void ApiKeysListCarriesNote()
    {
        JsonObject body = Json(Get("api-keys", SessionToken()));
        Assert.That(body["note"]!.GetValue<string>(), Is.Not.Empty);
        var items = (JsonArray)body["items"]!;
        var first = (JsonObject)items[0]!;
        Assert.That(first["is_default"]!.GetValue<bool>(), Is.True);
        // A key supplied through the environment stays MASKED: whoever set it already has it.
        Assert.That(first.ContainsKey("key"), Is.False);
        Assert.That(first.Select(p => p.Key), Is.SupersetOf(new[]
        {
            "id", "label", "prefix", "masked", "is_default", "created_at", "last_used_at",
        }));
    }

    /// <summary>
    /// The created key is the one and only time the plaintext exists outside the caller's hands,
    /// and the page reads <c>res.key</c> to show it once.
    /// </summary>
    [Test]
    public void CreateKeyReturnsThePlaintextOnce()
    {
        string token = SessionToken();
        HttpResponseMessage created = Post("api-keys", """{"label":"contract"}""", token);
        Assert.That(created.StatusCode, Is.EqualTo(HttpStatusCode.Created));
        JsonObject body = Json(created);
        string plaintext = body["key"]!.GetValue<string>();
        Assert.That(plaintext, Does.StartWith("rdk_"));
        Assert.That(body["warning"]!.GetValue<string>(), Is.Not.Empty);

        // Never again, in the listing.
        JsonObject listing = Json(Get("api-keys", token));
        foreach (JsonNode? item in (JsonArray)listing["items"]!)
        {
            Assert.That(((JsonObject)item!).ContainsKey("key"), Is.False);
        }

        // DELETE is 204 with an EMPTY body: a JSON body on a 204 is a protocol error some clients
        // reject outright.
        HttpResponseMessage deleted = Send(HttpMethod.Delete,
            $"api-keys/{body["id"]!.GetValue<int>()}", token: token);
        Assert.That(deleted.StatusCode, Is.EqualTo(HttpStatusCode.NoContent));
        Assert.That(Text(deleted), Is.Empty);
    }

    /// <summary>
    /// Deleting the environment key is refused with 409 — the request is well-formed and the caller
    /// is allowed to delete keys, but the STATE forbids this one, and "deleting" it would only be
    /// undone by the next restart.
    ///
    /// <para>
    /// The body assertion is the one that matters: the Go port shipped a defect where wrapping the
    /// error put the sentinel's own name into the message, so a 409 read as
    /// <c>"conflict: The default key ..."</c>.
    /// </para>
    /// </summary>
    [Test]
    public void DeleteDefaultKeyIsConflictWithACleanMessage()
    {
        HttpResponseMessage response = Send(HttpMethod.Delete, "api-keys/0",
            token: SessionToken());
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Conflict));
        Assert.That(Json(response)["detail"]!.GetValue<string>(),
            Does.StartWith("The default key"));
    }

    // -- settings ------------------------------------------------------------

    /// <summary>
    /// **The PUT body is WRAPPED.** settings/Index.vue posts <c>{ values }</c>, and parsing it flat
    /// meant the whitelist dropped everything, nothing was stored, and the page reported success.
    /// </summary>
    [Test]
    public void SettingsPutTakesAWrappedBody()
    {
        string token = SessionToken();
        HttpResponseMessage response = Send(HttpMethod.Put, "settings",
            """{"values":{"img_size":1200}}""", token);
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));

        JsonObject body = Json(response);
        Assert.That(body.Select(p => p.Key),
            Is.SupersetOf(new[] { "values", "schema", "restart_required" }));
        Assert.That(body["values"]!["img_size"]!.GetValue<string>(), Is.EqualTo("1200"));
        // An empty ARRAY, not null: the page assigns it straight to a list it iterates.
        Assert.That(body["restart_required"], Is.InstanceOf<JsonArray>());

        // And it really was stored, which is the half the flat-parse bug got wrong.
        Assert.That(Json(Get("settings", token))["values"]!["img_size"]!.GetValue<string>(),
            Is.EqualTo("1200"));

        // A FLAT body must be refused rather than silently accepted-and-dropped.
        Assert.That(Send(HttpMethod.Put, "settings", """{"img_size":900}""", token).StatusCode,
            Is.EqualTo(HttpStatusCode.BadRequest));
    }

    /// <summary>
    /// A rejected setting is 400, matching the reference — not 422, which is what FastAPI uses for
    /// its OWN validation errors and is therefore the tempting wrong answer.
    /// </summary>
    [Test]
    public void SettingsValidationIsBadRequestAndNamesTheBound()
    {
        HttpResponseMessage response = Send(HttpMethod.Put, "settings",
            """{"values":{"docconf":5.0}}""", SessionToken());
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
        Assert.That(Json(response)["detail"]!.GetValue<string>(), Does.Contain("<="));
    }

    /// <summary>
    /// <c>restart_required</c> is not decoration: <c>ocr_mode</c> is baked into the pipeline at
    /// construction, so a UI reporting "saved" and leaving the runtime alone would be lying about
    /// something an operator can verify on the status page.
    /// </summary>
    [Test]
    public void SettingsReportRestartRequired()
    {
        string token = SessionToken();
        JsonObject body = Json(Send(HttpMethod.Put, "settings",
            """{"values":{"ocr_mode":"fast"}}""", token));
        Assert.That(((JsonArray)body["restart_required"]!).Select(n => n!.GetValue<string>()),
            Is.EquivalentTo(new[] { "ocr_mode" }));
        Send(HttpMethod.Put, "settings", """{"values":{"ocr_mode":"accurate"}}""", token);
    }

    // -- bounded query parameters --------------------------------------------

    /// <summary>
    /// Pins the pydantic-shaped 422.
    ///
    /// <para>
    /// The expected bodies were CAPTURED from the running reference, not written from memory — the
    /// whole class of defect this guards against is a plausible-looking hand-written shape. Before
    /// this existed the Go port CLAMPED silently, so <c>page_size=500</c> answered 200 with 100
    /// rows: a successful reply to a request the reference rejects, which no amount of server-side
    /// testing would reveal.
    /// </para>
    /// </summary>
    [Test]
    public void BoundedQueryParamsMatchTheReference()
    {
        HttpResponseMessage tooLarge = Get("documents?page_size=500");
        Assert.That(tooLarge.StatusCode, Is.EqualTo(HttpStatusCode.UnprocessableEntity));
        var item = (JsonObject)((JsonArray)Json(tooLarge)["detail"]!)[0]!;
        Assert.That(item["type"]!.GetValue<string>(), Is.EqualTo("less_than_equal"));
        Assert.That(item["msg"]!.GetValue<string>(),
            Is.EqualTo("Input should be less than or equal to 100"));
        // The RAW string, not the parsed number: pydantic echoes what it was given.
        Assert.That(item["input"]!.GetValue<string>(), Is.EqualTo("500"));
        Assert.That(item["ctx"]!["le"]!.GetValue<int>(), Is.EqualTo(100));
        Assert.That(((JsonArray)item["loc"]!).Select(n => n!.GetValue<string>()),
            Is.EqualTo(new[] { "query", "page_size" }));

        HttpResponseMessage unparseable = Get("documents?page_size=abc");
        Assert.That(unparseable.StatusCode, Is.EqualTo(HttpStatusCode.UnprocessableEntity));
        var parse = (JsonObject)((JsonArray)Json(unparseable)["detail"]!)[0]!;
        Assert.That(parse["type"]!.GetValue<string>(), Is.EqualTo("int_parsing"));
        // ctx is ABSENT for a parse failure and PRESENT for a bound. That asymmetry is the
        // reference's, and reproducing it is the point.
        Assert.That(parse.ContainsKey("ctx"), Is.False);

        // ge is checked BEFORE le, so a value violating both reports the lower bound.
        var below = (JsonObject)((JsonArray)Json(Get("documents?page=0"))["detail"]!)[0]!;
        Assert.That(below["type"]!.GetValue<string>(), Is.EqualTo("greater_than_equal"));
    }

    /// <summary>
    /// The other half: aligning on rejection must not turn a MISSING parameter into an error. The
    /// SPA omits page_size on its first load.
    /// </summary>
    [Test]
    public void AbsentQueryParamUsesTheDefault()
    {
        JsonObject body = Json(Get("documents"));
        Assert.That(body["page"]!.GetValue<int>(), Is.EqualTo(1));
        Assert.That(body["page_size"]!.GetValue<int>(), Is.EqualTo(20));
    }

    // -- documents -----------------------------------------------------------

    /// <summary>
    /// The list row is what the log page renders, and the upload response is the SAME shape so the
    /// SPA can insert the row without a second request.
    /// </summary>
    [Test]
    public void ListRowCarriesEveryColumnTheLogPageRenders()
    {
        // Inserted directly rather than uploaded: this is a shape assertion, and recognition is
        // neither needed nor available here.
        Document record = Document.New(Documents.ReserveId(Db), "contract.jpg", "image/jpeg",
            1234, ".jpg");
        record.DocType = "INTPASSPORT_2011";
        record.Status = DocumentStatus.Done;
        Documents.Create(Db, record);

        var row = (JsonObject)((JsonArray)Json(Get("documents"))["items"]!)
            .First(n => ((JsonObject)n!)["id"]!.GetValue<int>() == record.Id)!;

        Assert.That(row.Select(p => p.Key), Is.SupersetOf(new[]
        {
            "id", "filename", "size_bytes", "status", "doc_type", "doc_type_base",
            "doc_type_era", "recognised", "doc_conf", "quality", "field_count", "device",
            "processing_ms", "error", "error_code", "retry_count", "has_canvas", "created_at",
            "started_at", "finished_at",
        }));
        // The era is split server-side, because the log page shows the two separately.
        Assert.That(row["doc_type_base"]!.GetValue<string>(), Is.EqualTo("INTPASSPORT"));
        Assert.That(row["doc_type_era"]!.GetValue<string>(), Is.EqualTo("2011"));
        // A timestamp is the shared record format, not .NET's default spelling.
        Assert.That(row["created_at"]!.GetValue<string>(), Does.EndWith("Z"));
        Assert.That(row["created_at"]!.GetValue<string>(), Does.Contain("T"));
    }

    /// <summary>
    /// The detail response must carry containers rather than nulls where the SPA iterates: a null
    /// <c>boxes</c> is a runtime error in the browser, not an empty table.
    /// </summary>
    [Test]
    public void DetailNeverReturnsNullWhereTheSpaIterates()
    {
        Document record = Documents.Create(Db, Document.New(Documents.ReserveId(Db),
            "empty.jpg", "image/jpeg", 10, ".jpg"));

        JsonObject body = Json(Get($"documents/{record.Id}"));
        Assert.That(body["boxes"], Is.InstanceOf<JsonArray>());
        Assert.That(body["fields"], Is.InstanceOf<JsonArray>());
        Assert.That(body["ocr"], Is.InstanceOf<JsonObject>());
        Assert.That(body["quality"], Is.InstanceOf<JsonObject>());
        Assert.That(body["timings"], Is.InstanceOf<JsonObject>());
        // The canvas block always has a URL, even before there is a canvas to fetch.
        Assert.That(body["canvas"]!["url"]!.GetValue<string>(), Does.Contain("/image/canvas"));
        Assert.That(body["original"]!["url"]!.GetValue<string>(),
            Does.Contain("/image/original"));
    }

    /// <summary>
    /// **200 with a JSON null, never 404.** The polling client would otherwise raise an error toast
    /// every two seconds for a document that finished perfectly well.
    /// </summary>
    [Test]
    public void ProgressIsTwoHundredWithNullForAQueuedThenTerminalDocument()
    {
        Document record = Documents.Create(Db, Document.New(Documents.ReserveId(Db),
            "queued.jpg", "image/jpeg", 10, ".jpg"));

        HttpResponseMessage queued = Get($"documents/{record.Id}/progress");
        Assert.That(queued.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        Assert.That(Json(queued)["step"]!.GetValue<string>(), Is.EqualTo("queued"));

        Documents.UpdateStatus(Db, Documents.GetById(Db, record.Id)!, DocumentStatus.Failed,
            "nope", "error");
        HttpResponseMessage failed = Get($"documents/{record.Id}/progress");
        Assert.That(failed.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        Assert.That(Json(failed)["step"]!.GetValue<string>(), Is.EqualTo("failed"));
    }

    /// <summary>
    /// An upload with no file part is a 400 naming the part, not a 500 — and a PDF gets its own
    /// message, because people WILL try it.
    /// </summary>
    [Test]
    public void UploadRejectionsAreActionable()
    {
        var noFile = new HttpRequestMessage(HttpMethod.Post, "documents")
        {
            Content = new MultipartFormDataContent("boundary")
            {
                { new StringContent("x"), "notfile" },
            },
        };
        noFile.Headers.Add("X-API-Key", ApiKey);
        HttpResponseMessage response = _client.Send(noFile);
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
        Assert.That(Json(response)["detail"]!.GetValue<string>(), Does.Contain("file"));

        var pdf = new HttpRequestMessage(HttpMethod.Post, "documents")
        {
            Content = new MultipartFormDataContent("boundary")
            {
                { new ByteArrayContent(Encoding.ASCII.GetBytes("%PDF-1.7 nope")), "file", "a.pdf" },
            },
        };
        pdf.Headers.Add("X-API-Key", ApiKey);
        HttpResponseMessage pdfResponse = _client.Send(pdf);
        Assert.That(pdfResponse.StatusCode,
            Is.EqualTo(HttpStatusCode.UnsupportedMediaType));
        Assert.That(Json(pdfResponse)["detail"]!.GetValue<string>(), Does.Contain("PDF"));
    }

    /// <summary>
    /// Reprocessing a document that is already queued or processing is a 409: the state forbids it,
    /// and requeueing it twice would run it twice.
    /// </summary>
    [Test]
    public void ReprocessingAQueuedDocumentIsConflict()
    {
        Document record = Documents.Create(Db, Document.New(Documents.ReserveId(Db),
            "busy.jpg", "image/jpeg", 10, ".jpg"));
        HttpResponseMessage response = Post($"documents/{record.Id}/reprocess");
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Conflict));
    }

    // -- the SPA -------------------------------------------------------------

    /// <summary>
    /// **Assets with a dot in the filename must be served.** This is the test the fixture could
    /// not previously run, and the defect it now guards against was invisible from the server side:
    /// the parameterless <c>MapFallback()</c> uses the pattern <c>{*path:nonfile}</c>, whose
    /// <c>nonfile</c> constraint excludes any path whose last segment contains a dot. So <c>/</c>
    /// returned index.html with a 200 — the server looked healthy — while every
    /// <c>/assets/index-&lt;hash&gt;.js</c> 404'd and the page rendered BLANK.
    /// </summary>
    [TestCase("/assets/index-abc123.js", "text/javascript")]
    [TestCase("/assets/index-abc123.css", "text/css")]
    [TestCase("/index.html", "text/html")]
    public void SpaServesAssetsWhoseNamesContainDots(string path, string expectedType)
    {
        HttpResponseMessage response = _client.Send(new HttpRequestMessage(HttpMethod.Get,
            new Uri(_client.BaseAddress!, path)));
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK), path);
        Assert.That(response.Content.Headers.ContentType?.MediaType,
            Is.EqualTo(expectedType), path);
        Assert.That(Text(response), Is.Not.Empty);
    }

    /// <summary>
    /// A client-side route gets the shell, so the SPA router can resolve it — and the shell is
    /// <c>no-cache</c>, because a cached index.html pins the client to an old bundle after a deploy
    /// while the hashed assets beside it are already new.
    /// </summary>
    [Test]
    public void SpaFallsBackToTheShellForClientRoutes()
    {
        HttpResponseMessage response = _client.Send(new HttpRequestMessage(HttpMethod.Get,
            new Uri(_client.BaseAddress!, "/documents/42")));
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        Assert.That(Text(response), Does.Contain("shell"));
        Assert.That(response.Headers.CacheControl?.NoCache, Is.True);
    }

    /// <summary>
    /// An unknown path UNDER THE API PREFIX is a JSON 404, never the SPA. Serving HTML there makes
    /// a client's JSON parse fail with a message about '&lt;', which is a genuinely confusing way to
    /// learn a route was misspelled.
    /// </summary>
    [Test]
    public void UnknownApiRoutesAreJsonNotHtml()
    {
        HttpResponseMessage response = Get("no-such-endpoint");
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
        Assert.That(response.Content.Headers.ContentType?.MediaType,
            Is.EqualTo("application/json"));
        Assert.That(Json(response)["detail"]!.GetValue<string>(), Is.EqualTo("Not found"));
    }

    /// <summary>
    /// A path that climbs out of the web root is NOT FOUND rather than forbidden, so a prober
    /// learns nothing about the filesystem layout — and certainly is not served the file.
    /// </summary>
    [Test]
    public void SpaRefusesToEscapeTheWebRoot()
    {
        // Sent raw, without Uri normalisation, or the client would collapse the traversal before it
        // ever reached the server — which would make this test pass against a broken server.
        HttpResponseMessage response = _client.Send(new HttpRequestMessage(HttpMethod.Get,
            new Uri(_client.BaseAddress!.GetLeftPart(UriPartial.Authority)
                    + "/..%2f..%2fappsettings.json")));
        Assert.That(response.StatusCode,
            Is.AnyOf(HttpStatusCode.NotFound, HttpStatusCode.BadRequest, HttpStatusCode.OK));
        if (response.StatusCode == HttpStatusCode.OK)
        {
            // If anything is returned it must be the shell, never a file from outside the root.
            Assert.That(Text(response), Does.Contain("shell"));
        }
    }

    // -- the recognition runtime, before it is ready --------------------------

    /// <summary>
    /// The service must be USABLE while the models load — that is the whole point of the queue, and
    /// it is why /health reports OK during startup. This fixture never initialises the runtime, so
    /// every test above already runs in that state; this one states it directly.
    /// </summary>
    [Test]
    public void HealthIsOkWhileTheRuntimeIsStillLoading()
    {
        var request = new HttpRequestMessage(HttpMethod.Get,
            new Uri(_client.BaseAddress!, "/health"));
        HttpResponseMessage response = _client.Send(request);
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        JsonObject body = Json(response);
        Assert.That(body["status"]!.GetValue<string>(), Is.EqualTo("ok"));
        Assert.That(body["runtime"]!.GetValue<string>(),
            Is.EqualTo(PipelineRuntime.StateInitializing));
    }
}
