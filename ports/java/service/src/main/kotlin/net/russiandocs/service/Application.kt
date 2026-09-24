package net.russiandocs.service

import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import net.russiandocs.docproc.NativeLibraries
import net.russiandocs.service.api.ApiRoutes
import net.russiandocs.service.api.ApiServer
import net.russiandocs.service.auth.AuthMode
import net.russiandocs.service.auth.Tokens
import net.russiandocs.service.repositories.Users
import net.russiandocs.service.config.Settings
import net.russiandocs.service.logging.LogRing
import net.russiandocs.service.logging.ServiceLog
import net.russiandocs.service.ml.PipelineRuntime
import net.russiandocs.service.repositories.SettingsRepository
import net.russiandocs.service.seed.SeedData
import net.russiandocs.service.store.FileStore
import net.russiandocs.service.worker.RecognitionWorker
import org.springframework.boot.WebApplicationType
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.builder.SpringApplicationBuilder

/**
 * The HTTP service entry point.
 *
 * **Startup order matters and is not arbitrary:**
 * 1. logging, so everything after it is captured;
 * 2. config, so a bad value fails before anything expensive;
 * 3. the store, wiping first if configured — that has to precede anything that reads it;
 * 4. the worker, which starts model loading in the BACKGROUND and returns immediately;
 * 5. the HTTP listener, which is therefore serving within milliseconds.
 *
 * The service accepts uploads while the models are still loading. That is the entire point of the queue, and
 * it is why `/health` reports OK during the seconds startup takes — gating health on the runtime would make
 * Docker kill the container before it finished booting.
 *
 * **Every collaborator is constructed HERE, by hand, and registered as a singleton.** No component scanning
 * of the service graph, no `@Autowired`: the order above is a correctness constraint, and a DI container that
 * chose its own order would satisfy the type system while breaking the invariant. Only [ApiRoutes] is a
 * Spring bean, because Spring MVC has to find the controller.
 *
 * Port of `service/main.py`.
 */
@SpringBootApplication(scanBasePackageClasses = [ApiRoutes::class])
public open class Application

private const val DEFAULT_ADDR = ":8005"

public fun main(args: Array<String>) {
    var addr = DEFAULT_ADDR
    var healthcheck = false
    var i = 0
    while (i < args.size) {
        when (args[i]) {
            "--addr" -> if (i + 1 < args.size) {
                addr = args[++i]
            }
            // Turns the binary into its own health probe, so the image needs no curl and no shell. A
            // HEALTHCHECK that depends on tools installed purely to run it is a larger image and one more
            // thing to keep patched.
            "--healthcheck" -> healthcheck = true
            "--help", "-h" -> {
                println("rdocs-service [--addr <[host]:port>] [--healthcheck]")
                return
            }
            else -> {
                System.err.println("unknown argument ${args[i]}")
                kotlin.system.exitProcess(2)
            }
        }
        i++
    }

    if (healthcheck) {
        kotlin.system.exitProcess(healthcheckProbe(addr))
    }

    try {
        // **Only a FAILURE exits.** `SpringApplication.run` RETURNS once the context is up — it does not
        // block the way ASP.NET Core's `app.Run()` does — so calling `exitProcess` on its result would kill
        // the process the moment startup finished. It did, and the symptom read as a healthy service: the
        // full startup log, "listening on :8005", Tomcat reporting itself started, and then a clean
        // shutdown two seconds later with no error anywhere. Tomcat's own non-daemon thread keeps the JVM
        // alive; the shutdown hook below is what stops it.
        val code = run(addr)
        if (code != 0) {
            kotlin.system.exitProcess(code)
        }
    } catch (e: Throwable) {
        // Printed as well as logged: if logging setup itself failed, the log line goes nowhere.
        System.err.println("fatal: ${e.message}")
        kotlin.system.exitProcess(1)
    }
}

/**
 * Probes the local instance. Exit 0 healthy, 1 otherwise.
 *
 * It talks to LOOPBACK rather than to the configured address, because `--addr` is a BIND spec: `:8005` means
 * every interface, and using it verbatim as a URL host does not resolve. Only the port is taken from it.
 */
private fun healthcheckProbe(addr: String): Int {
    val port = addr.substring(addr.lastIndexOf(':') + 1)
    return try {
        val client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(4)).build()
        val response = client.send(
            HttpRequest.newBuilder(URI.create("http://127.0.0.1:$port/health"))
                .timeout(Duration.ofSeconds(4)).GET().build(),
            HttpResponse.BodyHandlers.discarding(),
        )
        if (response.statusCode() !in 200..299) {
            System.err.println("healthcheck: HTTP ${response.statusCode()}")
            1
        } else {
            0
        }
    } catch (e: Exception) {
        System.err.println("healthcheck: ${e.message}")
        1
    }
}

private fun run(addr: String): Int {
    val configErrors = mutableListOf<String>()
    val cfg = Settings.load(configErrors)

    // Installed before anything else, so the config errors below land in the ring buffer that /logs serves.
    LogRing.stdoutLevel = LogRing.parseLevel(cfg.logLevel)
    val log = ServiceLog("service")

    for (error in configErrors) {
        log.error("[MAIN] config: $error")
    }
    if (configErrors.isNotEmpty()) {
        return 1
    }

    // J-01: the OpenCV and ONNX Runtime natives need their MinGW dependencies preloaded in order on
    // Windows. Done here rather than lazily, so a broken deployment fails at startup with a diagnosis
    // instead of on the first document with "the specified procedure could not be found".
    NativeLibraries.load()

    log.info("[MAIN] starting: version=${cfg.gitCommit} data_dir=${cfg.dataDir} " +
        "device=${cfg.computeDevice}")

    // **The data directory must live OUTSIDE the repository.** It holds uploaded documents, which are
    // personal data; the default is relative, so a deployment that leaves it unset gets a directory next to
    // the jar rather than inside a source tree.
    val dataDir = File(cfg.dataDir).absolutePath

    if (cfg.dataWipeOnStart) {
        try {
            val size = FileStore.wipe(dataDir)
            log.info("[MAIN] wiped data directory on startup: ${size / (1024 * 1024)} MB in $dataDir")
        } catch (e: Exception) {
            log.warn("[MAIN] could not wipe the data directory: ${e.message}")
        }
    }

    val db = FileStore(dataDir, log.sink())

    // Resolved and logged HERE rather than at first use, because a generated key that nobody ever sees is a
    // service nobody can call. The masked/unmasked decision is in the repository layer; this is only the log
    // line.
    val authCfg = Tokens.Config(
        pin = cfg.authPin, jwtSecret = cfg.jwtSecret, jwtAlgorithm = cfg.jwtAlgorithm,
        jwtExpireMinutes = cfg.jwtExpireMinutes, defaultApiKey = cfg.defaultApiKey,
    )
    val (key, generated) = Tokens.resolveDefaultKey(authCfg)
    if (generated) {
        log.warn("[MAIN] DEFAULT_API_KEY was not set; generated one for this process only: $key")
    } else {
        log.info("[MAIN] using DEFAULT_API_KEY from the environment")
    }

    if (Tokens.secretIsEphemeral(authCfg)) {
        // Not a warning to ignore any more: the default is no longer USED. A random secret is generated per
        // process instead, so the published default can never sign a token — the cost is only that
        // sessions end at restart.
        log.warn("[BOOT] JWT_SECRET is unset or still the published default — using a random per-process " +
            "secret instead; every session ends when the service restarts. Set JWT_SECRET to keep sessions " +
            "across restarts.")
    }
    if (db.isEphemeral) {
        log.warn("[MAIN] storage is TEMPORARY: everything is lost on restart. Set a database " +
            "connection string for anything real.")
    }

    val repoRoot = cfg.repoRoot()

    // Seeded BEFORE the worker starts, so the drain loop never sees a half-inserted fixture. A negative
    // SEED_SAMPLES disables it; 0 means all available.
    if (cfg.seedSamples >= 0 && repoRoot != null) {
        SeedData.ifEmpty(db, repoRoot, cfg.seedSamples, log)
    }

    val runtime = PipelineRuntime(log.sink())
    val settings = SettingsRepository(cfg, log.sink())
    val worker = RecognitionWorker(db, runtime, cfg, settings, log)
    worker.start()

    val webRoot = ApiServer.findWebRoot(repoRoot)
    if (webRoot == null) {
        log.warn("[MAIN] no frontend build found under $repoRoot; the API works but there is no UI")
    } else {
        log.info("[MAIN] serving frontend from $webRoot")
    }

    val api = ApiServer(db, runtime, worker, cfg, settings, webRoot, log)
    announceAuthMode(api.mode, db, cfg, log)

    val (host, port) = splitAddr(addr)
    val properties = mutableMapOf<String, Any>(
        "server.port" to port,
        // The multipart limits are the upload cap plus a byte, so an oversized body is refused by the
        // container instead of being buffered whole and then rejected.
        "spring.servlet.multipart.max-file-size" to "${cfg.maxUploadBytes + 1}B",
        "spring.servlet.multipart.max-request-size" to "${cfg.maxUploadBytes + 1}B",
    )
    if (host.isNotEmpty()) {
        properties["server.address"] = host
    }

    val app = SpringApplicationBuilder(Application::class.java)
        .web(WebApplicationType.SERVLET)
        .bannerMode(org.springframework.boot.Banner.Mode.OFF)
        .properties(properties)
        // The hand-built graph is handed to Spring as ready singletons. Registering them rather than
        // declaring @Bean factories keeps the construction order above authoritative.
        .initializers({ ctx ->
            ctx.beanFactory.registerSingleton("apiServer", api)
            // Registered as a bean so Spring installs it as a servlet filter, which is the one
            // place framework wiring is unavoidable: a filter has to be in the chain before the
            // DispatcherServlet, and nothing in application code can put it there.
            val origins = cfg.corsOrigins()
            if (origins.isNotEmpty()) {
                log.info("[MAIN] CORS enabled for: ${origins.joinToString(", ")}")
                ctx.beanFactory.registerSingleton(
                    "corsFilter",
                    org.springframework.boot.web.servlet.FilterRegistrationBean(
                        net.russiandocs.service.api.CorsFilter(origins),
                    ).apply { order = Int.MIN_VALUE },
                )
            }
        })
        .build()

    Runtime.getRuntime().addShutdownHook(Thread {
        log.info("[MAIN] shutdown signal received")
        worker.stop()
        runtime.close()
        log.info("[MAIN] stopped")
    })

    log.info("[MAIN] listening on $addr")
    app.run()
    return 0
}

/**
 * Says which authentication is in force, and seeds the first account in users mode.
 *
 * Loud on purpose, and loudest when the configuration was NOT honoured: a downgrade from named accounts to a
 * shared four-digit PIN is precisely the thing nobody notices from the outside — the service works, the login
 * page looks plausible, and the operator believes the accounts they configured are in effect.
 *
 * **Never fatal.** A service that will not boot because it could not seed a demo account is worse than one
 * that boots with no accounts and says so. And the password is logged only when it is the documented demo
 * value: the reference once logged `ADMIN_PASSWORD` unconditionally, writing a real secret into every log
 * collector the service feeds.
 *
 * Port of `_announce_auth_mode` in `service/main.py`.
 */
private fun announceAuthMode(mode: AuthMode.Resolved, db: FileStore, cfg: Settings, log: ServiceLog) {
    mode.downgradeReason?.let { log.warn("[AUTH] $it") }
    if (!mode.usersEnabled) {
        log.info("[AUTH] PIN sign-in; user accounts are disabled (set AUTH_MODE=users to enable them)")
        return
    }

    val created = try {
        // The decoy is computed here, once, so the first sign-in for an unknown username does not pay for it
        // and become measurably slower than a wrong password.
        Users.warmDecoy()
        Users.seedAdmin(db, cfg.adminUsername, cfg.adminPassword)
    } catch (e: Exception) {
        log.error("[AUTH] could not seed the first administrator", e)
        return
    }

    log.info("[AUTH] named accounts (AUTH_MODE=users)")
    if (created != null) {
        val shown = if (cfg.adminPassword == Settings.DEFAULT_ADMIN_PASSWORD) {
            "'${cfg.adminPassword}'"
        } else {
            "from ADMIN_PASSWORD (not logged)"
        }
        log.warn("[AUTH] seeded the first administrator '${created.username}', password $shown — it must be " +
            "changed at first sign-in, and it is re-created after every restart because this store is " +
            "temporary")
    }
}

/**
 * Splits a bind spec into host and port.
 *
 * `:8005` means every interface, which Spring expresses by leaving `server.address` unset; an explicit host
 * passes through. Accepting the Go and .NET ports' `--addr` spelling matters because the services are
 * deployed from compose files that differ only in the image.
 */
private fun splitAddr(addr: String): Pair<String, Int> {
    val colon = addr.lastIndexOf(':')
    if (colon < 0) {
        val port = addr.toIntOrNull()
            ?: throw IllegalArgumentException("--addr: \"$addr\" has no port")
        return "" to port
    }
    val host = addr.substring(0, colon)
    val port = addr.substring(colon + 1).toIntOrNull()
        ?: throw IllegalArgumentException("--addr: \"$addr\" has no port")
    return host to port
}
