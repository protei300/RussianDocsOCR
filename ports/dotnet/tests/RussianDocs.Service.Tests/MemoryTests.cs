using System.Buffers;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NUnit.Framework;
using OpenCvSharp;
using RussianDocs.DocumentProcessing.ViewModel;
using RussianDocs.Service.Api;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Ml;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Store;
using RussianDocs.Service.Worker;
using Image = RussianDocs.DocumentProcessing.Imaging.Image;
using LogLevel = Microsoft.Extensions.Logging.LogLevel;
using PipelineResults = RussianDocs.DocumentProcessing.Pipeline.Results;

namespace RussianDocs.Service.Tests;

/// <summary>
/// <c>POST /recognize</c> and <c>PROCESSING_MODE=memory</c>: nothing about a document is kept. What the
/// black box (<c>conformance/memory_contract.py</c>) cannot see, per ports/MEMORY.md §7; mirrors
/// <c>tests/service/test_recognize.py</c>.
///
/// <para>
/// Every test checks a way something could be kept, not just that the route answers: files under the
/// data directory, ASP.NET's temporary upload files, log lines, the journal, and the buffers themselves —
/// which must be zeroed, not merely dropped. Recognition is replaced by a fake so the suite runs without
/// the models. **The no-spool, buffer and scrub tests were proven by deleting the production line each
/// guards, watching the test fail, and restoring it.**
/// </para>
/// </summary>
[TestFixture]
public sealed class MemoryTests
{
    private const string ApiKey = "rdk_memory_test_key";
    private const string SecretText = "ИВАНОВ ЛЕНИНА 17";         // what the fake "recognises"
    private const string FileName = "Ivanov_passport_scan.jpg";   // what the client calls the file

    private string _dir = null!;

    [SetUp]
    public void MakeDir()
    {
        _dir = Path.Combine(Path.GetTempPath(), "rdocs-memory-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    /// <summary>The bootstrap key is resolved once per process; forget ours so the next fixture sets its own.</summary>
    [OneTimeTearDown]
    public void ForgetKey() => Tokens.ResetDefaultKeyForTests();

    [TearDown]
    public void RemoveDir()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* temp dir */ }
    }

    // ------------------------------------------------------------------ the harness

    /// <summary>Collects every formatted log line — the log an operator would see.</summary>
    private sealed class ListLogger : ILogger
    {
        private readonly List<string> _lines = [];
        public List<string> Lines { get { lock (_lines) { return [.. _lines]; } } }
        public string Text => string.Join("\n", Lines);

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            lock (_lines)
            {
                _lines.Add(formatter(state, exception) + (exception is null ? "" : " " + exception));
            }
        }
    }

    /// <summary>The recogniser replaced: records what it was handed and keeps references to it.</summary>
    private sealed class FakeRecogniser : IImageRecogniser
    {
        public readonly List<Mat> Pixels = [];   // header copies sharing the handler's pixel data
        public readonly List<byte[]> Pngs = [];
        public bool PixelsWereSet;
        public Exception? Throw;
        public TimeSpan LeaseTimeout;
        public bool Recognised = true;

        public MemoryRecognition RecogniseImage(Image rgb, PipelineRuntime.RecogniseOptions opts, bool withCanvas)
        {
            LeaseTimeout = opts.LeaseTimeout;
            // A second header on the same data: it keeps the pixels alive after the handler disposes its
            // Image, so the test can look at what the handler left in them.
            Pixels.Add(new Mat(rgb.Mat, new Rect(0, 0, rgb.Width, rgb.Height)));
            PixelsWereSet = Cv2.CountNonZero(rgb.Mat.Reshape(1)) > 0;
            if (Throw is not null)
            {
                throw Throw;
            }
            var payload = new Payload
            {
                DocType = Recognised ? "INTPASSPORT_2011" : "NONE",
                DocTypeBase = Recognised ? "INTPASSPORT" : null,
                DocTypeEra = Recognised ? "2011" : null,
                Recognised = Recognised,
                Device = "cpu",
                Canvas = new Canvas { Width = 10, Height = 8, IsFallback = false },
                Fields = [new Field { Name = "Address", Value = SecretText }],
                Ocr = new Dictionary<string, string> { ["Address"] = SecretText },
                Quality = new Dictionary<string, object> { ["DocConf"] = 0.97, ["Blur"] = "good" },
            };
            byte[]? png = null;
            if (withCanvas)
            {
                png = [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4];
                Pngs.Add(png);
            }
            return new MemoryRecognition(payload, png);
        }
    }

    /// <summary>
    /// A pool that records every array it hands out and how each came back. It does NOT clear on return,
    /// so an array found zero afterwards was zeroed by the handler, not by the pool.
    /// </summary>
    private sealed class RecordingPool : ArrayPool<byte>
    {
        public readonly List<byte[]> Rented = [];
        public readonly List<(byte[] Array, bool Cleared)> Returned = [];

        public override byte[] Rent(int minimumLength)
        {
            var array = new byte[minimumLength];
            lock (Rented) { Rented.Add(array); }
            return array;
        }

        public override void Return(byte[] array, bool clearArray = false)
        {
            lock (Returned) { Returned.Add((array, clearArray)); }
        }
    }

    private sealed class Server : IDisposable
    {
        public required WebApplication App { get; init; }
        public required HttpClient Http { get; init; }
        public required ApiServer Api { get; init; }
        public required ListLogger Log { get; init; }
        public required string DataDir { get; init; }

        public void Dispose()
        {
            Http.Dispose();
            App.StopAsync().GetAwaiter().GetResult();
            App.DisposeAsync().GetAwaiter().GetResult();
        }
    }

    private Server Start(string? mode = null, int maxUploadMb = 20, int recognizeWaitSec = 30,
        IImageRecogniser? recogniser = null, ArrayPool<byte>? pool = null)
    {
        string dataDir = Path.Combine(_dir, "data");
        Directory.CreateDirectory(dataDir);
        var cfg = new Config.Settings
        {
            DataDir = dataDir,
            DataWipeOnStart = false,
            DefaultApiKey = ApiKey,
            JwtSecret = "memory-test-secret",
            SeedSamples = -1,
            ProcessingMode = mode ?? "stored",
            MaxUploadMB = maxUploadMb,
            RecognizeWaitSec = recognizeWaitSec,
        };
        var log = new ListLogger();
        Tokens.ResetDefaultKeyForTests(); // resolved once per process: make it this fixture's key
        var db = new FileStore(dataDir, log);
        var runtime = new PipelineRuntime(log);
        var settings = new SettingsRepository(cfg, log);
        var worker = new RecognitionWorker(db, runtime, cfg, settings, log);
        var api = new ApiServer(db, runtime, worker, cfg, settings, AuthRuntime.Create(cfg, db), null, log);
        if (recogniser is not null)
        {
            api.Recogniser = recogniser;
        }
        if (pool is not null)
        {
            api.BufferPool = pool;
        }

        var builder = WebApplication.CreateBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        WebApplication app = builder.Build();
        api.MapRoutes(app);
        app.Start();
        string address = app.Services
            .GetRequiredService<Microsoft.AspNetCore.Hosting.Server.IServer>()
            .Features.Get<Microsoft.AspNetCore.Hosting.Server.Features.IServerAddressesFeature>()!
            .Addresses.First();
        // UseProxy = false: see ContractTests — a system proxy would otherwise take 127.0.0.1 off the box.
        var http = new HttpClient(new HttpClientHandler { UseProxy = false })
        {
            BaseAddress = new Uri(address + "/api/v1/"),
        };
        return new Server { App = app, Http = http, Api = api, Log = log, DataDir = dataDir };
    }

    private static HttpRequestMessage Req(HttpMethod method, string path, HttpContent? content = null,
        bool key = true, string? token = null)
    {
        var request = new HttpRequestMessage(method, path) { Content = content };
        if (token is not null)
        {
            request.Headers.Add("Authorization", "Bearer " + token);
        }
        else if (key)
        {
            request.Headers.Add("X-API-Key", ApiKey);
        }
        return request;
    }

    private static ByteArrayContent Raw(byte[] data, string contentType = "image/jpeg")
    {
        var content = new ByteArrayContent(data);
        content.Headers.ContentType = MediaTypeHeaderValue.Parse(contentType);
        return content;
    }

    private static MultipartFormDataContent Multipart(byte[] data, string field = "file", string name = FileName,
        string contentType = "image/png")
    {
        var part = new ByteArrayContent(data);
        part.Headers.ContentType = MediaTypeHeaderValue.Parse(contentType);
        return new MultipartFormDataContent { { part, field, name } };
    }

    private static JsonObject Json(HttpResponseMessage response) =>
        (JsonObject)JsonNode.Parse(response.Content.ReadAsStringAsync().GetAwaiter().GetResult())!;

    private static string Session(Server s)
    {
        HttpResponseMessage r = s.Http.Send(Req(HttpMethod.Post, "auth/pin-login",
            new StringContent("""{"pin":"1234"}""", Encoding.UTF8, "application/json"), key: false));
        return Json(r)["access_token"]!.GetValue<string>();
    }

    private static HashSet<string> FilesUnder(string root) =>
        Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).ToHashSet(StringComparer.Ordinal);

    private static byte[] Jpeg(int w = 320, int h = 200)
    {
        using var img = new Mat(h, w, MatType.CV_8UC3, new Scalar(200, 180, 160));
        Cv2.ImEncode(".jpg", img, out byte[] jpeg);
        return jpeg;
    }

    /// <summary>A PNG of random noise: incompressible, so it is as big as asked.</summary>
    private static byte[] NoisePng(int w = 1400, int h = 1000)
    {
        using var img = new Mat(h, w, MatType.CV_8UC3);
        Cv2.Randu(img, new Scalar(0, 0, 0), new Scalar(256, 256, 256));
        Cv2.ImEncode(".png", img, out byte[] png);
        return png;
    }

    /// <summary>
    /// Watches the directory ASP.NET buffers form uploads into (<c>ASPNETCORE_TEMP</c>, else the system temp
    /// directory) for the files it creates there (<c>ASPNETCORE_*.tmp</c>, deleted on close — so they have
    /// to be caught being created).
    /// </summary>
    private sealed class SpoolWatcher : IDisposable
    {
        private readonly FileSystemWatcher _watcher;
        private readonly List<string> _created = [];

        public SpoolWatcher()
        {
            string dir = Environment.GetEnvironmentVariable("ASPNETCORE_TEMP") is { Length: > 0 } t
                ? t : Path.GetTempPath();
            _watcher = new FileSystemWatcher(dir, "ASPNETCORE_*")
            {
                NotifyFilter = NotifyFilters.FileName,
                IncludeSubdirectories = false,
            };
            _watcher.Created += (_, e) => { lock (_created) { _created.Add(e.Name ?? ""); } };
            // On Linux ASP.NET creates the spool as /tmp/tmpXXXX.tmp and RENAMES it to ASPNETCORE_*.tmp
            // (seen with inotifywait); a Created-only watcher misses it there and both checks go blind.
            _watcher.Renamed += (_, e) => { lock (_created) { _created.Add(e.Name ?? ""); } };
            _watcher.EnableRaisingEvents = true;
        }

        public List<string> Created
        {
            get
            {
                Thread.Sleep(500); // the watcher's events arrive on its own thread
                lock (_created) { return [.. _created]; }
            }
        }

        public void Dispose() => _watcher.Dispose();
    }

    // ------------------------------------------------------------------ the route

    /// <summary>The 200 answer: the reference's keys exactly, stored=false, no-store, nothing on disk.</summary>
    [Test]
    public void RecognizeAnswersAndStoresNothing()
    {
        var fake = new FakeRecogniser();
        using Server s = Start(recogniser: fake);
        HashSet<string> before = FilesUnder(s.DataDir);

        HttpResponseMessage resp = s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(Jpeg())));
        Assert.That(resp.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        JsonObject body = Json(resp);
        Assert.Multiple(() =>
        {
            Assert.That(body.Select(p => p.Key), Is.EqualTo(new[]
            {
                "stored", "doc_type", "doc_type_base", "doc_type_era", "recognised", "doc_conf", "device",
                "canvas", "coord_space", "coord_space_note", "boxes", "fields", "ocr", "quality", "address",
                "timings", "processing_ms",
            }), "the keys of service/api/recognize.py::_response_body, in its order");
            Assert.That(body["stored"]!.GetValue<bool>(), Is.False);
            Assert.That(body["recognised"]!.GetValue<bool>(), Is.True);
            Assert.That(body["ocr"]!["Address"]!.GetValue<string>(), Is.EqualTo(SecretText));
            Assert.That(body["doc_conf"]!.GetValue<double>(), Is.EqualTo(0.97));
            Assert.That(((JsonObject)body["quality"]!).ContainsKey("DocConf"), Is.False);
            Assert.That(((JsonObject)body["canvas"]!).ContainsKey("image_png_base64"), Is.False);
            Assert.That(resp.Headers.CacheControl?.NoStore, Is.True);
            Assert.That(FilesUnder(s.DataDir), Is.EquivalentTo(before), "a recognition left a file");
            Assert.That(s.Log.Text, Does.Not.Contain(SecretText).And.Not.Contain("INTPASSPORT"));
            List<string> lines = s.Log.Lines.Where(l => l.StartsWith("[API] recognize", StringComparison.Ordinal))
                .ToList();
            Assert.That(lines, Has.Count.EqualTo(1));
            Assert.That(lines[0], Does.Match(@"^\[API\] recognize: outcome=recognised status=200 ms=\d+$"));
        });

        JsonObject journal = Json(s.Http.Send(Req(HttpMethod.Get, "recognitions")));
        Assert.That(journal["total"]!.GetValue<int>(), Is.EqualTo(1));
        Assert.That(((JsonObject)journal["items"]![0]!).Select(p => p.Key),
            Is.EquivalentTo(new[] { "id", "at", "outcome", "status_code", "processing_ms", "stored" }));
        Assert.That(journal["counts"]!["recognised"]!.GetValue<int>(), Is.EqualTo(1));
    }

    /// <summary>
    /// §7.1 — a multipart body well past ASP.NET's 64 KB form-buffering threshold goes through /recognize
    /// and no <c>ASPNETCORE_*.tmp</c> appears. The positive control — the SAME watcher catching the
    /// document upload, which does go through ReadFormAsync — proves the watcher can see a spool at all.
    /// </summary>
    [Test]
    public void MultipartIsSplitInMemoryNotSpooled()
    {
        byte[] big = NoisePng();
        Assert.That(big.Length, Is.GreaterThan(2 * 1024 * 1024));
        using Server s = Start(recogniser: new FakeRecogniser());

        using (var watcher = new SpoolWatcher())
        {
            HttpResponseMessage resp = s.Http.Send(Req(HttpMethod.Post, "recognize", Multipart(big)));
            Assert.That(resp.StatusCode, Is.EqualTo(HttpStatusCode.OK));
            Assert.That(watcher.Created, Is.Empty, "a temporary file was created while parsing the upload");
        }
        Assert.That(s.Log.Text, Does.Not.Contain(FileName));

        using (var control = new SpoolWatcher())
        {
            HttpResponseMessage resp = s.Http.Send(Req(HttpMethod.Post, "documents", Multipart(big)));
            Assert.That(resp.StatusCode, Is.EqualTo(HttpStatusCode.Accepted));
            Assert.That(control.Created, Is.Not.Empty,
                "positive control: the stored upload's ReadFormAsync must be seen spooling");
        }
    }

    /// <summary>
    /// §7.2 — with a fake that keeps references to what it got, after the request every buffer is all zeros:
    /// the body, the image part, the decoded pixels, the PNG. And every pooled array went back cleared.
    /// </summary>
    [Test]
    public void BuffersAreZeroedAfterTheAnswer()
    {
        var fake = new FakeRecogniser();
        var pool = new RecordingPool();
        using Server s = Start(recogniser: fake, pool: pool);

        HttpResponseMessage resp = s.Http.Send(Req(HttpMethod.Post, "recognize?images=1", Multipart(Jpeg())));
        Assert.That(resp.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        string b64 = Json(resp)["canvas"]!["image_png_base64"]!.GetValue<string>();
        Assert.That(Convert.FromBase64String(b64)[..4], Is.EqualTo(new byte[] { 0x89, (byte)'P', (byte)'N', (byte)'G' }));

        Assert.Multiple(() =>
        {
            Assert.That(pool.Rented, Has.Count.GreaterThanOrEqualTo(2), "the body and the image part");
            Assert.That(pool.Rented.Sum(a => a.Length), Is.GreaterThan(0));
            foreach (byte[] array in pool.Rented)
            {
                Assert.That(array.All(b => b == 0), Is.True, $"a {array.Length}-byte buffer was not wiped");
                Assert.That(pool.Returned.Any(r => ReferenceEquals(r.Array, array) && r.Cleared), Is.True,
                    "a rented buffer was not returned with clearArray: true");
            }
            Assert.That(fake.PixelsWereSet, Is.True, "sanity: the recogniser saw real pixels");
            Assert.That(fake.Pixels, Has.Count.EqualTo(1));
            Assert.That(Cv2.CountNonZero(fake.Pixels[0].Reshape(1)), Is.Zero, "the decoded pixels were not wiped");
            Assert.That(fake.Pngs[0].All(b => b == 0), Is.True, "the canvas PNG was not wiped");
        });
        fake.Pixels.ForEach(m => m.Dispose());
    }

    /// <summary>The raw-body path wipes the same way: its body buffer IS the image buffer.</summary>
    [Test]
    public void RawBodyIsZeroedToo()
    {
        var fake = new FakeRecogniser();
        var pool = new RecordingPool();
        using Server s = Start(recogniser: fake, pool: pool);

        Assert.That(s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(Jpeg()))).StatusCode,
            Is.EqualTo(HttpStatusCode.OK));
        Assert.That(pool.Rented, Is.Not.Empty);
        Assert.That(pool.Rented.All(a => a.All(b => b == 0)), Is.True);
        Assert.That(Cv2.CountNonZero(fake.Pixels[0].Reshape(1)), Is.Zero);
        fake.Pixels.ForEach(m => m.Dispose());
    }

    [TestCase(new byte[0], "image/jpeg", 400, "Empty upload")]
    [TestCase(new byte[] { 0x25, 0x50, 0x44, 0x46, 0x2d, 0x31 }, "application/pdf", 415,
        "PDF is not supported — send a JPEG, PNG, WEBP, BMP or TIFF image")]
    [TestCase(new byte[] { 0x6a, 0x75, 0x73, 0x74, 0x20, 0x74, 0x65, 0x78, 0x74 }, "text/plain", 415,
        "Unsupported file type — expected an image")]
    [TestCase(new byte[] { 0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0, 0, 0, 0, 0 }, "image/jpeg", 422,
        "The image could not be decoded — it may be corrupt")]
    public void BadInputIsRefusedAndJournalled(byte[] payload, string contentType, int status, string detail)
    {
        var fake = new FakeRecogniser();
        var pool = new RecordingPool();
        using Server s = Start(recogniser: fake, pool: pool);
        HttpResponseMessage resp = s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(payload, contentType)));
        Assert.That((int)resp.StatusCode, Is.EqualTo(status));
        Assert.That(Json(resp)["detail"]!.GetValue<string>(), Is.EqualTo(detail));
        JsonObject item = (JsonObject)Json(s.Http.Send(Req(HttpMethod.Get, "recognitions")))["items"]![0]!;
        Assert.That(item["outcome"]!.GetValue<string>(), Is.EqualTo("rejected"));
        Assert.That(item["status_code"]!.GetValue<int>(), Is.EqualTo(status));
        Assert.That(fake.Pixels, Is.Empty, "a refused image never reaches recognition");
        Assert.That(pool.Rented.All(a => a.All(b => b == 0)), Is.True, "a refused upload is wiped too");
    }

    [Test]
    public void MalformedMultipartIsRefused()
    {
        using Server s = Start(recogniser: new FakeRecogniser());
        HttpResponseMessage noBoundary = s.Http.Send(Req(HttpMethod.Post, "recognize",
            Raw("--x\r\n\r\n"u8.ToArray(), "multipart/form-data")));
        Assert.That((int)noBoundary.StatusCode, Is.EqualTo(400));
        Assert.That(Json(noBoundary)["detail"]!.GetValue<string>(), Is.EqualTo("Malformed multipart body"));

        var other = new MultipartFormDataContent { { new StringContent("hello"), "note" } };
        HttpResponseMessage noImage = s.Http.Send(Req(HttpMethod.Post, "recognize", other));
        Assert.That((int)noImage.StatusCode, Is.EqualTo(400));
        Assert.That(Json(noImage)["detail"]!.GetValue<string>(),
            Is.EqualTo("No image in the multipart body — send it as the 'file' part"));
    }

    [Test]
    public void OversizedUploadIsRefused()
    {
        using Server s = Start(maxUploadMb: 1, recogniser: new FakeRecogniser());
        byte[] big = new byte[2 * 1024 * 1024];
        big[0] = 0xff; big[1] = 0xd8; big[2] = 0xff;
        HttpResponseMessage resp = s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(big)));
        Assert.That((int)resp.StatusCode, Is.EqualTo(413));
        JsonObject item = (JsonObject)Json(s.Http.Send(Req(HttpMethod.Get, "recognitions")))["items"]![0]!;
        Assert.That(item["status_code"]!.GetValue<int>(), Is.EqualTo(413));
    }

    [Test]
    public void BusyPipelineIs503WithRetryAfter()
    {
        var fake = new FakeRecogniser { Throw = ServiceException.Busy("no pipeline") };
        using Server s = Start(recognizeWaitSec: 7, recogniser: fake);
        HttpResponseMessage resp = s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(Jpeg())));
        Assert.That((int)resp.StatusCode, Is.EqualTo(503));
        Assert.That(resp.Headers.RetryAfter?.Delta, Is.EqualTo(TimeSpan.FromSeconds(5)));
        Assert.That(fake.LeaseTimeout, Is.EqualTo(TimeSpan.FromSeconds(7)), "RECOGNIZE_WAIT_SEC is the lease wait");
        JsonObject item = (JsonObject)Json(s.Http.Send(Req(HttpMethod.Get, "recognitions")))["items"]![0]!;
        Assert.That(item["outcome"]!.GetValue<string>(), Is.EqualTo("busy"));
        Assert.That(Cv2.CountNonZero(fake.Pixels[0].Reshape(1)), Is.Zero, "pixels are wiped on failure too");
        fake.Pixels.ForEach(m => m.Dispose());
    }

    [Test]
    public void RuntimeNotReadyIs503WithRetryAfter()
    {
        // The real runtime, never initialised: exactly the state during the first seconds of a start.
        using Server s = Start();
        HttpResponseMessage resp = s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(Jpeg())));
        Assert.That((int)resp.StatusCode, Is.EqualTo(503));
        Assert.That(resp.Headers.RetryAfter?.Delta, Is.EqualTo(TimeSpan.FromSeconds(10)));
        Assert.That(Json(resp)["detail"]!.GetValue<string>(), Is.EqualTo("The recognition runtime is not ready yet"));
    }

    [Test]
    public void UnexpectedErrorIs500AndLogsOnlyTheType()
    {
        var fake = new FakeRecogniser { Throw = new InvalidOperationException($"failed while reading {SecretText}") };
        using Server s = Start(recogniser: fake);
        HttpResponseMessage resp = s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(Jpeg())));
        string text = resp.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        Assert.That((int)resp.StatusCode, Is.EqualTo(500));
        Assert.That(text, Does.Not.Contain(SecretText));
        Assert.That(s.Log.Text, Does.Not.Contain(SecretText));
        Assert.That(s.Log.Text, Does.Contain("InvalidOperationException"));
        fake.Pixels.ForEach(m => m.Dispose());
    }

    [Test]
    public void NotRecognisedIsAnOutcome()
    {
        using Server s = Start(recogniser: new FakeRecogniser { Recognised = false });
        Assert.That(s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(Jpeg()))).StatusCode, Is.EqualTo(HttpStatusCode.OK));
        JsonObject journal = Json(s.Http.Send(Req(HttpMethod.Get, "recognitions")));
        Assert.That(journal["items"]![0]!["outcome"]!.GetValue<string>(), Is.EqualTo("not_recognised"));
    }

    /// <summary>Auth runs first: an unauthenticated call is a 401 and leaves no journal line.</summary>
    [Test]
    public void UnauthenticatedIsNotJournalled()
    {
        using Server s = Start(recogniser: new FakeRecogniser());
        Assert.That((int)s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(Jpeg()), key: false)).StatusCode,
            Is.EqualTo(401));
        Assert.That(Json(s.Http.Send(Req(HttpMethod.Get, "recognitions")))["total"]!.GetValue<int>(), Is.Zero);
    }

    // ------------------------------------------------------------------ memory mode

    [Test]
    public void MemoryModeTurnsDocumentStorageOff()
    {
        var fake = new FakeRecogniser();
        using Server s = Start("memory", recogniser: fake);
        string token = Session(s);   // signing in writes the audit log: that is not a document
        HashSet<string> before = FilesUnder(s.DataDir);

        Assert.That(Json(s.Http.Send(Req(HttpMethod.Get, "auth/config", key: false)))["processing_mode"]!
            .GetValue<string>(), Is.EqualTo("memory"));
        foreach ((HttpMethod method, string path) in new[]
                 {
                     (HttpMethod.Get, "documents"), (HttpMethod.Get, "documents/1"),
                     (HttpMethod.Get, "documents/1/image/canvas"), (HttpMethod.Delete, "documents/1"),
                     (HttpMethod.Post, "documents/purge"),
                 })
        {
            HttpResponseMessage r = s.Http.Send(Req(method, path, token: token));
            Assert.That((int)r.StatusCode, Is.EqualTo(404), $"{method} {path}");
            Assert.That(Json(r)["detail"]!.GetValue<string>(), Does.Contain("PROCESSING_MODE=memory"));
        }
        // Answered before authentication, too: nothing about storage is a secret worth a file on disk.
        Assert.That((int)s.Http.Send(Req(HttpMethod.Get, "documents", key: false)).StatusCode, Is.EqualTo(404));

        using (var watcher = new SpoolWatcher())
        {
            HttpResponseMessage upload = s.Http.Send(Req(HttpMethod.Post, "documents", Multipart(NoisePng())));
            Assert.That((int)upload.StatusCode, Is.EqualTo(404));
            Assert.That(watcher.Created, Is.Empty, "the refused upload was parsed into a temporary file");
        }

        Assert.That(s.Http.Send(Req(HttpMethod.Post, "recognize", Raw(Jpeg()))).StatusCode, Is.EqualTo(HttpStatusCode.OK));
        Assert.That(FilesUnder(s.DataDir), Is.EquivalentTo(before), "memory mode wrote a file");
        JsonObject processing = (JsonObject)Json(s.Http.Send(Req(HttpMethod.Get, "status", token: token)))["processing"]!;
        Assert.That(processing["mode"]!.GetValue<string>(), Is.EqualTo("memory"));
        Assert.That(processing["reason"], Is.Null);
        Assert.That(processing["recognitions"]!["recognised"]!.GetValue<int>(), Is.EqualTo(1));
        fake.Pixels.ForEach(m => m.Dispose());
    }

    /// <summary>
    /// §7.4 — the gate answers 404 WITHOUT touching the body: a body stream that fails on any read, a
    /// declared length far over every limit, and a next() that would read it. Neither is reached.
    /// </summary>
    [Test]
    public void MemoryModeGateNeverReadsTheBody()
    {
        using Server s = Start("memory");
        var body = new TrapStream();
        var context = new DefaultHttpContext();
        context.Request.Method = "POST";
        context.Request.Path = "/api/v1/documents";
        context.Request.ContentLength = 500L * 1024 * 1024;
        context.Request.ContentType = "multipart/form-data; boundary=x";
        context.Request.Body = body;
        context.Response.Body = new MemoryStream();
        bool nextCalled = false;

        s.Api.MemoryModeGate(context, async () =>
        {
            nextCalled = true;
            await context.Request.ReadFormAsync();
        }).GetAwaiter().GetResult();

        Assert.That(context.Response.StatusCode, Is.EqualTo(404));
        Assert.That(nextCalled, Is.False, "the request went on to the upload handler");
        Assert.That(body.Reads, Is.Zero, "the body was read");
        context.Response.Body.Position = 0;
        Assert.That(new StreamReader(context.Response.Body).ReadToEnd(), Does.Contain("PROCESSING_MODE=memory"));
    }

    /// <summary>In stored mode the same gate lets every route through.</summary>
    [Test]
    public void StoredModeGateLetsDocumentsThrough()
    {
        using Server s = Start("stored");
        var context = new DefaultHttpContext();
        context.Request.Path = "/api/v1/documents";
        bool nextCalled = false;
        s.Api.MemoryModeGate(context, () => { nextCalled = true; return Task.CompletedTask; })
            .GetAwaiter().GetResult();
        Assert.That(nextCalled, Is.True);
        Assert.That((int)s.Http.Send(Req(HttpMethod.Get, "documents")).StatusCode, Is.EqualTo(200));
        Assert.That(Json(s.Http.Send(Req(HttpMethod.Get, "auth/config", key: false)))["processing_mode"]!
            .GetValue<string>(), Is.EqualTo("stored"));
    }

    private sealed class TrapStream : Stream
    {
        public int Reads;
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => 0; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count)
        {
            Interlocked.Increment(ref Reads);
            throw new InvalidOperationException("the body was read");
        }
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref Reads);
            throw new InvalidOperationException("the body was read");
        }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    [Test]
    public void AMistypedModeMeansMemoryAndSaysWhy()
    {
        using Server s = Start("memroy");
        Assert.That(Json(s.Http.Send(Req(HttpMethod.Get, "auth/config", key: false)))["processing_mode"]!
            .GetValue<string>(), Is.EqualTo("memory"));
        JsonObject processing = (JsonObject)Json(s.Http.Send(Req(HttpMethod.Get, "status", token: Session(s))))["processing"]!;
        Assert.That(processing["mode"]!.GetValue<string>(), Is.EqualTo("memory"));
        Assert.That(processing["reason"]!.GetValue<string>(), Does.Contain("memroy"));
        Assert.That((int)s.Http.Send(Req(HttpMethod.Get, "documents")).StatusCode, Is.EqualTo(404));
    }

    /// <summary>§7.6 — ports/MEMORY.md §1, row by row. A wrong value fails toward privacy.</summary>
    [TestCase(null, "stored", false)]
    [TestCase("", "stored", false)]
    [TestCase("   ", "stored", false)]
    [TestCase("stored", "stored", false)]
    [TestCase(" Stored ", "stored", false)]
    [TestCase("memory", "memory", false)]
    [TestCase("MEMORY", "memory", false)]
    [TestCase("memroy", "memory", true)]
    [TestCase("off", "memory", true)]
    [TestCase("store", "memory", true)]
    public void ModeResolution(string? raw, string mode, bool hasReason)
    {
        (string resolved, string? reason) = Config.ProcessingMode.Resolve(raw);
        Assert.That(resolved, Is.EqualTo(mode));
        if (hasReason)
        {
            Assert.That(reason, Does.Contain(raw!.Trim().ToLowerInvariant()));
            Assert.That(reason, Is.EqualTo($"PROCESSING_MODE='{raw.Trim().ToLowerInvariant()}' is not one of " +
                "stored, memory — using 'memory' (documents are not stored), the safe reading of a " +
                "mistyped privacy setting"));
        }
        else
        {
            Assert.That(reason, Is.Null);
        }
    }

    [Test]
    public void JournalIsBoundedAndNewestFirst()
    {
        var journal = new RecognitionJournal(3);
        for (int i = 0; i < 5; i++)
        {
            journal.Record(i % 2 == 0 ? "recognised" : "rejected", 200, i);
        }
        (List<RecognitionJournal.Entry> page, int total) = journal.Recent(10, 0);
        Assert.That(total, Is.EqualTo(3));
        Assert.That(page.Select(e => e.Id), Is.EqualTo(new[] { 5, 4, 3 }));
        Assert.That(journal.Counts(), Is.EquivalentTo(new Dictionary<string, int>
        {
            ["recognised"] = 2, ["not_recognised"] = 0, ["rejected"] = 1, ["busy"] = 0, ["error"] = 0, ["total"] = 3,
        }));
        Assert.That(page[0].Public()["at"], Does.Match(@"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z$"));
    }

    // ------------------------------------------------------------------ the pipeline scrub (§7.3)

    /// <summary>
    /// A fake run that fills a Results the way the real one does — owned intermediates and a canvas — and
    /// keeps header copies of them so the test can look after the wipe. The "weights" are a Mat the run
    /// reads but does not own: they must survive.
    /// </summary>
    private sealed class FakeRun : IDisposable
    {
        public readonly Mat Weights = new(4, 4, MatType.CV_32FC1, new Scalar(3));
        public readonly List<Mat> Kept = [];
        public PipelineResults? Seen;

        public void Run(PipelineResults results, bool fail)
        {
            Seen = results;
            Image original = Image.Wrap(new Mat(50, 60, MatType.CV_8UC3, new Scalar(7, 7, 7)));
            Image crop = Image.Wrap(new Mat(8, 30, MatType.CV_8UC3, new Scalar(9, 9, 9)));
            Image canvas = Image.Wrap(new Mat(40, 50, MatType.CV_8UC3, new Scalar(5, 5, 5)));
            foreach (Image image in new[] { original, crop, canvas })
            {
                Kept.Add(new Mat(image.Mat, new Rect(0, 0, image.Width, image.Height)));
            }
            results.Own(original);
            results.Own(crop);
            results.Canvas = canvas;
            if (fail)
            {
                // What the real run does on any failure (Recognizer.ReadDocument's catch): release
                // everything it allocated, then rethrow — so a wipe that only happened in the caller's
                // finally would find nothing left to zero.
                results.Dispose();
                throw new InvalidOperationException("boom");
            }
        }

        public void Dispose()
        {
            Weights.Dispose();
            Kept.ForEach(m => m.Dispose());
        }
    }

    [TestCase(false, false)]
    [TestCase(true, false)]
    [TestCase(false, true)]
    public void TheRunsImagesAreZeroedOnSuccessAndOnFailure(bool runFails, bool transformFails)
    {
        using var fake = new FakeRun();
        MemoryRecognition? got = null;
        TestDelegate call = () => got = PipelineRuntime.Scrubbed(
            results => fake.Run(results, runFails),
            _ => transformFails ? throw new InvalidOperationException("transform") : new Payload { DocType = "X" },
            withCanvas: true);

        if (runFails || transformFails)
        {
            Assert.Throws<InvalidOperationException>(call);
        }
        else
        {
            Assert.DoesNotThrow(call);
            Assert.That(got!.ViewModel.DocType, Is.EqualTo("X"));
            Assert.That(got.Png![..4], Is.EqualTo(new byte[] { 0x89, (byte)'P', (byte)'N', (byte)'G' }));
        }

        Assert.Multiple(() =>
        {
            for (int i = 0; i < fake.Kept.Count; i++)
            {
                Assert.That(Cv2.CountNonZero(fake.Kept[i].Reshape(1)), Is.Zero, $"image {i} was not wiped");
            }
            Assert.That(fake.Seen!.Canvas, Is.Null, "the results still hold a canvas");
            Assert.That(Cv2.CountNonZero(fake.Weights), Is.EqualTo(16), "model state was wiped");
        });
    }
}
