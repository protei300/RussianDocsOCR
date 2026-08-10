using System.Globalization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using RussianDocs.Service.Api;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Logging;
using RussianDocs.Service.Ml;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Seed;
using RussianDocs.Service.Store;
using RussianDocs.Service.Worker;

namespace RussianDocs.Service;

/// <summary>
/// The HTTP service entry point.
///
/// <para>
/// **Startup order matters and is not arbitrary:**
/// </para>
/// <list type="number">
/// <item>logging, so everything after it is captured;</item>
/// <item>config, so a bad value fails before anything expensive;</item>
/// <item>the store, wiping first if configured — that has to precede anything that reads it;</item>
/// <item>the worker, which starts model loading in the BACKGROUND and returns immediately;</item>
/// <item>the HTTP listener, which is therefore serving within milliseconds.</item>
/// </list>
///
/// <para>
/// The service accepts uploads while the models are still loading. That is the entire point of the
/// queue, and it is why <c>/health</c> reports OK during the seconds startup takes — gating health on
/// the runtime would make Docker kill the container before it finished booting.
/// </para>
///
/// <para>Port of <c>service/main.py</c>.</para>
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        string addr = ":8004";
        bool healthcheck = false;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--addr" when i + 1 < args.Length:
                    addr = args[++i];
                    break;
                // Turns the binary into its own health probe, so the image needs no curl and no shell.
                // A HEALTHCHECK that depends on tools installed purely to run it is a larger image and
                // one more thing to keep patched.
                case "--healthcheck":
                    healthcheck = true;
                    break;
                case "--help" or "-h":
                    Console.Out.WriteLine(
                        "rdocs-service [--addr <[host]:port>] [--healthcheck]");
                    return 0;
                default:
                    Console.Error.WriteLine($"unknown argument {args[i]}");
                    return 2;
            }
        }

        if (healthcheck)
        {
            return Healthcheck(addr);
        }

        try
        {
            return Run(addr);
        }
        catch (Exception ex)
        {
            // Printed as well as logged: if logging setup itself failed, the log line goes nowhere.
            Console.Error.WriteLine($"fatal: {ex.Message}");
            return 1;
        }
    }

    /// <summary>
    /// Probes the local instance. Exit 0 healthy, 1 otherwise.
    ///
    /// <para>
    /// It talks to LOOPBACK rather than to the configured address, because <c>--addr</c> is a BIND spec:
    /// <c>:8004</c> means every interface, and using it verbatim as a URL host does not resolve. Only
    /// the port is taken from it.
    /// </para>
    /// </summary>
    private static int Healthcheck(string addr)
    {
        string port = addr[(addr.LastIndexOf(':') + 1)..];
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(4) };
            HttpResponseMessage response =
                client.GetAsync($"http://127.0.0.1:{port}/health").GetAwaiter().GetResult();
            if (!response.IsSuccessStatusCode)
            {
                Console.Error.WriteLine($"healthcheck: HTTP {(int)response.StatusCode}");
                return 1;
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"healthcheck: {ex.Message}");
            return 1;
        }
    }

    private static int Run(string addr)
    {
        Config.Settings cfg = Config.Settings.Load(out IReadOnlyList<string> configErrors);

        var builder = WebApplication.CreateBuilder();
        builder.Logging.ClearProviders();
        // Logging is installed even when the config failed, so the errors below are captured in the ring
        // buffer that /logs serves.
        builder.Logging.AddProvider(new RingLoggerProvider(LogRing.ParseLevel(cfg.LogLevel)));
        builder.Logging.SetMinimumLevel(LogLevel.Debug);

        // Kestrel's own request logging is silenced: one JSON line per HTTP request would bury the
        // service's own lines, and the ring buffer is what the logs page reads.
        builder.Logging.AddFilter("Microsoft", LogLevel.Warning);

        builder.WebHost.ConfigureKestrel(options =>
        {
            options.Limits.MaxRequestBodySize = cfg.MaxUploadBytes + 1;
            // Read and write timeouts are deliberately ASYMMETRIC in the Go port; Kestrel expresses the
            // same intent through the request-body rate limits it already applies by default, plus this
            // header timeout. Neither covers recognition, which happens in the worker and not in a
            // request.
            options.Limits.RequestHeadersTimeout = TimeSpan.FromSeconds(10);
        });
        builder.WebHost.UseUrls(ListenUrl(addr));

        WebApplication app = builder.Build();
        ILogger log = app.Services.GetRequiredService<ILoggerFactory>()
            .CreateLogger("service");

        foreach (string error in configErrors)
        {
            log.LogError("[MAIN] config: {Error}", error);
        }
        if (configErrors.Count > 0)
        {
            return 1;
        }

        log.LogInformation("[MAIN] starting: version={Version} data_dir={Dir} device={Device}",
            cfg.GitCommit, cfg.DataDir, cfg.ComputeDevice);

        // **The data directory must live OUTSIDE the repository.** It holds uploaded documents, which
        // are personal data; the default is relative, so a deployment that leaves it unset gets a
        // directory next to the binary rather than inside a source tree.
        string dataDir = Path.GetFullPath(cfg.DataDir);

        if (cfg.DataWipeOnStart)
        {
            try
            {
                long size = FileStore.Wipe(dataDir);
                log.LogInformation("[MAIN] wiped data directory on startup: {Mb} MB in {Dir}",
                    size / (1024 * 1024), dataDir);
            }
            catch (Exception ex)
            {
                log.LogWarning("[MAIN] could not wipe the data directory: {Error}", ex.Message);
            }
        }

        var db = new FileStore(dataDir, log);

        // Resolved and logged HERE rather than at first use, because a generated key that nobody ever
        // sees is a service nobody can call. The masked/unmasked decision is in the repository layer;
        // this is only the log line.
        var authCfg = new Tokens.Config
        {
            Pin = cfg.AuthPin, JwtSecret = cfg.JwtSecret, JwtAlgorithm = cfg.JwtAlgorithm,
            JwtExpireMinutes = cfg.JwtExpireMinutes, DefaultApiKey = cfg.DefaultApiKey,
        };
        (string key, bool generated) = Tokens.ResolveDefaultKey(authCfg);
        if (generated)
        {
            log.LogWarning(
                "[MAIN] DEFAULT_API_KEY was not set; generated one for this process only: {Key}",
                key);
        }
        else
        {
            log.LogInformation("[MAIN] using DEFAULT_API_KEY from the environment");
        }

        if (cfg.JwtSecret == new Config.Settings().JwtSecret)
        {
            log.LogWarning("[MAIN] JWT_SECRET is the built-in default — set it before exposing " +
                           "this service to anything you care about");
        }
        if (db.IsEphemeral)
        {
            log.LogWarning("[MAIN] storage is TEMPORARY: everything is lost on restart. " +
                           "Set a database connection string for anything real.");
        }

        string? repoRoot = cfg.RepoRoot();

        // Seeded BEFORE the worker starts, so the drain loop never sees a half-inserted fixture. A
        // negative SEED_SAMPLES disables it; 0 means all available.
        if (cfg.SeedSamples >= 0 && repoRoot is not null)
        {
            SeedData.IfEmpty(db, repoRoot, cfg.SeedSamples, log);
        }

        var runtime = new PipelineRuntime(log);
        var settings = new SettingsRepository(cfg, log);
        var worker = new RecognitionWorker(db, runtime, cfg, settings, log);

        using var shutdown = new CancellationTokenSource();
        worker.Start(shutdown.Token);

        string? webRoot = ApiServer.FindWebRoot(repoRoot);
        if (webRoot is null)
        {
            log.LogWarning("[MAIN] no frontend build found under {Root}; the API works but " +
                           "there is no UI", repoRoot);
        }
        else
        {
            log.LogInformation("[MAIN] serving frontend from {Dir}", webRoot);
        }

        new ApiServer(db, runtime, worker, cfg, settings, webRoot, log).MapRoutes(app);

        // CORS is applied by hand rather than through the middleware package: exact-origin matching is
        // four lines, and a wildcard reflected back with credentials enabled is the classic CORS mistake
        // in a service that authenticates every route that matters.
        string[] corsOrigins = cfg.CorsOrigins();
        if (corsOrigins.Length > 0)
        {
            app.Use(async (context, next) =>
            {
                string origin = context.Request.Headers.Origin.ToString();
                if (origin.Length > 0 && Array.IndexOf(corsOrigins, origin) >= 0)
                {
                    context.Response.Headers.AccessControlAllowOrigin = origin;
                    context.Response.Headers.Vary = "Origin";
                    context.Response.Headers.AccessControlAllowHeaders =
                        "Authorization, Content-Type, X-API-Key";
                    context.Response.Headers.AccessControlAllowMethods =
                        "GET, POST, PUT, DELETE, OPTIONS";
                }
                if (HttpMethods.IsOptions(context.Request.Method))
                {
                    context.Response.StatusCode = StatusCodes.Status204NoContent;
                    return;
                }
                await next();
            });
        }

        app.Lifetime.ApplicationStopping.Register(() =>
        {
            log.LogInformation("[MAIN] shutdown signal received");
            shutdown.Cancel();
        });

        log.LogInformation("[MAIN] listening on {Addr}", addr);
        app.Run();

        runtime.Dispose();
        log.LogInformation("[MAIN] stopped");
        return 0;
    }

    /// <summary>
    /// Turns a bind spec into the URL Kestrel wants.
    ///
    /// <para>
    /// <c>:8004</c> means every interface, which Kestrel spells <c>http://*:8004</c>; an explicit host
    /// passes through. Accepting the Go port's <c>--addr</c> spelling matters because the two services
    /// are deployed from compose files that differ only in the image.
    /// </para>
    /// </summary>
    private static string ListenUrl(string addr)
    {
        int colon = addr.LastIndexOf(':');
        if (colon < 0)
        {
            return $"http://*:{addr}";
        }
        string host = addr[..colon];
        string port = addr[(colon + 1)..];
        if (!int.TryParse(port, NumberStyles.Integer, CultureInfo.InvariantCulture, out _))
        {
            throw new ArgumentException($"--addr: \"{addr}\" has no port");
        }
        return host.Length == 0 ? $"http://*:{port}" : $"http://{host}:{port}";
    }
}
