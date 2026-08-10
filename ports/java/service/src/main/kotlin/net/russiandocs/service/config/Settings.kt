package net.russiandocs.service.config

import java.io.File

/**
 * The environment tier of configuration, resolved once at startup.
 *
 * Two tiers, split by WHO changes them and how often: this one holds secrets and anything that cannot change
 * without a restart (the PIN, the JWT secret, the default API key, the data directory); the SETTINGS STORE
 * holds operator-tunable knobs the worker re-reads every loop, so a change applies without bouncing the
 * service.
 *
 * Secrets never appear in the settings store and are never returned by any endpoint. That split is the entire
 * reason [authPin], [jwtSecret] and [defaultApiKey] live here and nowhere else.
 *
 * **No configuration-binding framework.** The reference uses pydantic-settings and Spring offers
 * `@ConfigurationProperties`, but a hand-written [load] is what the Go and .NET ports can read line for line,
 * and it keeps each DEFAULT visible next to the field it belongs to instead of buried in a property source.
 * Port of `service/core/config.py`.
 */
public data class Settings(
    // --- Auth ---------------------------------------------------------------
    /** Protects the WEBSITE only. API endpoints authenticate with API keys. */
    val authPin: String = "1234",
    val jwtSecret: String = "changeme-in-production",
    val jwtAlgorithm: String = "HS256",
    /** 480 minutes — eight hours, one working day. */
    val jwtExpireMinutes: Int = 480,
    /**
     * The bootstrap key: always present, never deletable — without it a restart (which wipes runtime-created
     * keys) would leave the API with no way in.
     *
     * **EMPTY BY DEFAULT, on purpose.** A hardcoded fallback would mean every deployment that forgot to set
     * this shares one publicly-known key. Empty means a random key is generated at startup and logged.
     */
    val defaultApiKey: String = "",

    // --- Storage ------------------------------------------------------------
    /**
     * Selects the metadata store. Empty falls back to a temporary on-disk store wiped at every start — fine
     * for a demo, useless for anything real, so startup says so loudly.
     */
    val databaseConnectionString: String = "",
    /**
     * Holds uploaded originals, canvases and thumbnails. Artifacts stay on the filesystem in BOTH modes:
     * multi-megabyte PNGs do not belong in a database row.
     */
    val dataDir: String = "data",
    /**
     * Wipes [dataDir] at startup.
     *
     * Note that "no Docker volume" alone does NOT make the directory ephemeral: `docker restart` keeps the
     * writable layer. This flag is what makes the promise true.
     */
    val dataWipeOnStart: Boolean = true,
    val maxUploadMb: Int = 20,
    /**
     * Queues the anonymised documents from `samples/` when the store is empty, so the log demonstrates real
     * results instead of showing nothing.
     *
     * ONLY ever repository samples, never user uploads. Negative disables; 0 means all available.
     */
    val seedSamples: Int = 0,

    // --- Recognition --------------------------------------------------------
    /** auto | cpu | gpu */
    val computeDevice: String = "auto",
    /** ONNX | OpenVINO */
    val modelFormat: String = "ONNX",
    /** accurate | fast. ('legacy' was removed in 3.0.0 and raises.) */
    val ocrMode: String = "accurate",
    val pipelinePoolSize: Int = 1,
    /**
     * Must be an anonymised repository sample. **NEVER a real user document**: warmup re-reads this file at
     * every start.
     */
    val warmupImage: String = "",

    // --- Worker -------------------------------------------------------------
    val jobTimeoutSec: Int = 120,
    val maxRetries: Int = 2,
    val docconf: Double = 0.5,
    val imgSize: Int = 1500,

    // --- Ops ----------------------------------------------------------------
    val logLevel: String = "INFO",
    val corsAllowedOrigins: String = "",
    val gitCommit: String = "unknown",
) {
    public fun corsOrigins(): List<String> =
        corsAllowedOrigins.split(',').map { it.trim() }.filter { it.isNotEmpty() }

    public val maxUploadBytes: Long get() = maxUploadMb.toLong() * 1024 * 1024

    /**
     * Locates the repository root, for finding `samples/` and `web/`.
     *
     * Returns `null` when nothing matches, and callers degrade rather than fail — a missing `samples/` costs a
     * cold first document, not a broken service.
     */
    public fun repoRoot(): String? {
        System.getenv("RDOCS_REPO_ROOT")?.takeIf { it.isNotEmpty() }?.let { return it }

        var dir: File? = File(".").canonicalFile
        while (dir != null) {
            if (File(dir, "document_processing/models").isDirectory) {
                return dir.path
            }
            dir = dir.parentFile
        }
        return null
    }

    public companion object {
        /**
         * Applies the environment over the defaults, collecting ALL parse errors before failing.
         *
         * Variable names are UPPER_SNAKE of the property, which is what pydantic-settings derives on the
         * Python side — so one compose file works unchanged against any implementation. The connection string
         * is the single exception: the reference aliases it explicitly, and both spellings are accepted.
         */
        public fun load(errors: MutableList<String>): Settings {
            fun str(key: String, current: String): String = System.getenv(key) ?: current

            fun num(key: String, current: Int): Int {
                val v = System.getenv(key) ?: return current
                return v.trim().toIntOrNull()
                    ?: run { errors += "$key=\"$v\" is not an integer"; current }
            }

            fun flt(key: String, current: Double): Double {
                val v = System.getenv(key) ?: return current
                return v.trim().toDoubleOrNull()
                    ?: run { errors += "$key=\"$v\" is not a number"; current }
            }

            fun bool(key: String, current: Boolean): Boolean {
                val v = System.getenv(key) ?: return current
                // The spellings pydantic accepts, so a compose file stays portable.
                return when (v.trim().lowercase()) {
                    "1", "true", "yes", "on" -> true
                    "0", "false", "no", "off" -> false
                    else -> { errors += "$key=\"$v\" is not a boolean"; current }
                }
            }

            val d = Settings()
            return Settings(
                authPin = str("AUTH_PIN", d.authPin),
                jwtSecret = str("JWT_SECRET", d.jwtSecret),
                jwtAlgorithm = str("JWT_ALGORITHM", d.jwtAlgorithm),
                jwtExpireMinutes = num("JWT_EXPIRE_MINUTES", d.jwtExpireMinutes),
                defaultApiKey = str("DEFAULT_API_KEY", d.defaultApiKey),
                databaseConnectionString = str("RUSSIANDOCS_DATABASE_CONNECTIONSTRING",
                    str("DATABASE_CONNECTIONSTRING", d.databaseConnectionString)),
                dataDir = str("DATA_DIR", d.dataDir),
                dataWipeOnStart = bool("DATA_WIPE_ON_START", d.dataWipeOnStart),
                maxUploadMb = num("MAX_UPLOAD_MB", d.maxUploadMb),
                seedSamples = num("SEED_SAMPLES", d.seedSamples),
                computeDevice = str("COMPUTE_DEVICE", d.computeDevice),
                modelFormat = str("MODEL_FORMAT", d.modelFormat),
                ocrMode = str("OCR_MODE", d.ocrMode),
                pipelinePoolSize = num("PIPELINE_POOL_SIZE", d.pipelinePoolSize),
                warmupImage = str("WARMUP_IMAGE", d.warmupImage),
                jobTimeoutSec = num("JOB_TIMEOUT_SEC", d.jobTimeoutSec),
                maxRetries = num("MAX_RETRIES", d.maxRetries),
                docconf = flt("DOCCONF", d.docconf),
                imgSize = num("IMG_SIZE", d.imgSize),
                logLevel = str("LOG_LEVEL", d.logLevel),
                corsAllowedOrigins = str("CORS_ALLOWED_ORIGINS", d.corsAllowedOrigins),
                gitCommit = str("GIT_COMMIT", d.gitCommit),
            )
        }
    }
}
