using System.Globalization;

namespace RussianDocs.Service.Config;

/// <summary>
/// The environment tier of configuration, resolved once at startup.
///
/// <para>
/// Two tiers, split by WHO changes them and how often: this one holds secrets and anything that
/// cannot change without a restart (the PIN, the JWT secret, the default API key, the data
/// directory); the SETTINGS STORE holds operator-tunable knobs the worker re-reads every loop, so a
/// change applies without bouncing the service.
/// </para>
///
/// <para>
/// Secrets never appear in the settings store and are never returned by any endpoint. That split is
/// the entire reason <see cref="AuthPin"/>, <see cref="JwtSecret"/> and <see cref="DefaultApiKey"/>
/// live here and nowhere else.
/// </para>
///
/// <para>
/// **No configuration-binding framework.** The reference uses pydantic-settings and ASP.NET offers
/// <c>IConfiguration</c>, but a hand-written <see cref="Load"/> is what the Go and Kotlin ports can
/// read line for line, and it keeps each DEFAULT visible next to the field it belongs to instead of
/// buried in a provider chain. Port of <c>service/core/config.py</c>.
/// </para>
/// </summary>
public sealed record Settings
{
    // --- Auth ---------------------------------------------------------------

    /// <summary>Protects the WEBSITE only. API endpoints authenticate with API keys.</summary>
    public string AuthPin { get; init; } = "1234";

    public string JwtSecret { get; init; } = "changeme-in-production";
    public string JwtAlgorithm { get; init; } = "HS256";

    /// <summary>480 minutes — eight hours, one working day.</summary>
    public int JwtExpireMinutes { get; init; } = 480;

    /// <summary>
    /// The bootstrap key: always present, never deletable — without it a restart (which wipes
    /// runtime-created keys) would leave the API with no way in.
    ///
    /// <para>
    /// **EMPTY BY DEFAULT, on purpose.** A hardcoded fallback would mean every deployment that
    /// forgot to set this shares one publicly-known key. Empty means a random key is generated at
    /// startup and logged; set the variable when an integration needs one that survives restarts.
    /// </para>
    /// </summary>
    public string DefaultApiKey { get; init; } = "";

    // --- Storage ------------------------------------------------------------

    /// <summary>
    /// Selects the metadata store. Empty falls back to a temporary on-disk store wiped at every
    /// start — fine for a demo, useless for anything real, so startup says so loudly.
    /// </summary>
    public string DatabaseConnectionString { get; init; } = "";

    /// <summary>
    /// Holds uploaded originals, canvases and thumbnails. Artifacts stay on the filesystem in BOTH
    /// modes: multi-megabyte PNGs do not belong in a database row.
    /// </summary>
    public string DataDir { get; init; } = "data";

    /// <summary>
    /// Wipes <see cref="DataDir"/> at startup.
    ///
    /// <para>
    /// Only meaningful in temporary mode — with a database configured the rows outlive the process,
    /// so wiping the images would leave every stored document pointing at a missing file.
    /// </para>
    ///
    /// <para>
    /// Note that "no Docker volume" alone does NOT make the directory ephemeral: `docker restart`
    /// keeps the writable layer. This flag is what makes the promise true.
    /// </para>
    /// </summary>
    public bool DataWipeOnStart { get; init; } = true;

    public int MaxUploadMB { get; init; } = 20;

    /// <summary>
    /// Queues the anonymised documents from <c>samples/</c> when the store is empty, so the log
    /// demonstrates real results instead of showing nothing.
    ///
    /// <para>
    /// ONLY ever repository samples, never user uploads. Negative disables; 0 means all available.
    /// </para>
    /// </summary>
    public int SeedSamples { get; init; }

    // --- Recognition --------------------------------------------------------

    /// <summary>auto | cpu | gpu</summary>
    public string ComputeDevice { get; init; } = "auto";

    /// <summary>ONNX | OpenVINO</summary>
    public string ModelFormat { get; init; } = "ONNX";

    /// <summary>accurate | fast. ('legacy' was removed in 3.0.0 and raises.)</summary>
    public string OcrMode { get; init; } = "accurate";

    public int PipelinePoolSize { get; init; } = 1;

    /// <summary>
    /// Must be an anonymised repository sample. **NEVER a real user document**: warmup re-reads
    /// this file at every start.
    /// </summary>
    public string WarmupImage { get; init; } = "";

    // --- Worker -------------------------------------------------------------

    public int JobTimeoutSec { get; init; } = 120;
    public int MaxRetries { get; init; } = 2;
    public double Docconf { get; init; } = 0.5;
    public int ImgSize { get; init; } = 1500;

    // --- Ops ----------------------------------------------------------------

    public string LogLevel { get; init; } = "INFO";
    public string CorsAllowedOrigins { get; init; } = "";
    public string GitCommit { get; init; } = "unknown";

    /// <summary>
    /// Applies the environment over the defaults, collecting ALL parse errors before failing.
    ///
    /// <para>
    /// Variable names are UPPER_SNAKE of the property, which is what pydantic-settings derives on
    /// the Python side — so one compose file works unchanged against any implementation. The
    /// connection string is the single exception: the reference aliases it explicitly so it reads
    /// unambiguously in a compose file, and both spellings are accepted here.
    /// </para>
    /// </summary>
    public static Settings Load(out IReadOnlyList<string> errors)
    {
        var errs = new List<string>();
        var s = new Settings();

        string Str(string key, string current) =>
            Environment.GetEnvironmentVariable(key) is { } v ? v : current;

        int Num(string key, int current)
        {
            string? v = Environment.GetEnvironmentVariable(key);
            if (v is null)
            {
                return current;
            }
            if (int.TryParse(v.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture,
                    out int n))
            {
                return n;
            }
            errs.Add($"{key}=\"{v}\" is not an integer");
            return current;
        }

        double Flt(string key, double current)
        {
            string? v = Environment.GetEnvironmentVariable(key);
            if (v is null)
            {
                return current;
            }
            if (double.TryParse(v.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture,
                    out double f))
            {
                return f;
            }
            errs.Add($"{key}=\"{v}\" is not a number");
            return current;
        }

        bool Boolean(string key, bool current)
        {
            string? v = Environment.GetEnvironmentVariable(key);
            if (v is null)
            {
                return current;
            }
            // The spellings pydantic accepts, so a compose file stays portable.
            return v.Trim().ToLowerInvariant() switch
            {
                "1" or "true" or "yes" or "on" => true,
                "0" or "false" or "no" or "off" => false,
                _ => Fail(),
            };

            bool Fail()
            {
                errs.Add($"{key}=\"{v}\" is not a boolean");
                return current;
            }
        }

        s = s with
        {
            AuthPin = Str("AUTH_PIN", s.AuthPin),
            JwtSecret = Str("JWT_SECRET", s.JwtSecret),
            JwtAlgorithm = Str("JWT_ALGORITHM", s.JwtAlgorithm),
            JwtExpireMinutes = Num("JWT_EXPIRE_MINUTES", s.JwtExpireMinutes),
            DefaultApiKey = Str("DEFAULT_API_KEY", s.DefaultApiKey),

            DatabaseConnectionString = Str("RUSSIANDOCS_DATABASE_CONNECTIONSTRING",
                Str("DATABASE_CONNECTIONSTRING", s.DatabaseConnectionString)),
            DataDir = Str("DATA_DIR", s.DataDir),
            DataWipeOnStart = Boolean("DATA_WIPE_ON_START", s.DataWipeOnStart),
            MaxUploadMB = Num("MAX_UPLOAD_MB", s.MaxUploadMB),
            SeedSamples = Num("SEED_SAMPLES", s.SeedSamples),

            ComputeDevice = Str("COMPUTE_DEVICE", s.ComputeDevice),
            ModelFormat = Str("MODEL_FORMAT", s.ModelFormat),
            OcrMode = Str("OCR_MODE", s.OcrMode),
            PipelinePoolSize = Num("PIPELINE_POOL_SIZE", s.PipelinePoolSize),
            WarmupImage = Str("WARMUP_IMAGE", s.WarmupImage),

            JobTimeoutSec = Num("JOB_TIMEOUT_SEC", s.JobTimeoutSec),
            MaxRetries = Num("MAX_RETRIES", s.MaxRetries),
            Docconf = Flt("DOCCONF", s.Docconf),
            ImgSize = Num("IMG_SIZE", s.ImgSize),

            LogLevel = Str("LOG_LEVEL", s.LogLevel),
            CorsAllowedOrigins = Str("CORS_ALLOWED_ORIGINS", s.CorsAllowedOrigins),
            GitCommit = Str("GIT_COMMIT", s.GitCommit),
        };

        errors = errs;
        return s;
    }

    public string[] CorsOrigins() => CorsAllowedOrigins
        .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);

    public long MaxUploadBytes => (long)MaxUploadMB * 1024 * 1024;

    /// <summary>
    /// Locates the repository root, for finding <c>samples/</c> and <c>web/</c>.
    ///
    /// <para>
    /// Resolved from the executable's location and then from the working directory, because the
    /// service runs from three places with three different layouts: a container (<c>/app</c>), a
    /// <c>dotnet run</c> from the project directory, and a published binary. Returns <c>null</c>
    /// when nothing matches, and callers degrade rather than fail — a missing <c>samples/</c> costs
    /// a cold first document, not a broken service.
    /// </para>
    /// </summary>
    public string? RepoRoot()
    {
        if (Environment.GetEnvironmentVariable("RDOCS_REPO_ROOT") is { Length: > 0 } explicitRoot)
        {
            return explicitRoot;
        }

        var candidates = new List<string> { "." };
        string dir = AppContext.BaseDirectory;
        candidates.Add(dir);
        for (int up = 1; up <= 6; up++)
        {
            dir = Path.Combine(dir, "..");
            candidates.Add(dir);
        }
        candidates.Add(Path.Combine("..", ".."));

        foreach (string candidate in candidates)
        {
            if (Directory.Exists(Path.Combine(candidate, "document_processing", "models")))
            {
                return Path.GetFullPath(candidate);
            }
        }
        return null;
    }
}
