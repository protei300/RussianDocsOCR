package net.russiandocs.service.logging

import java.io.FileDescriptor
import java.io.FileOutputStream
import java.io.PrintStream
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.Locale
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** One buffered record. The JSON shape is what the logs page reads. */
@Serializable
public data class LogEntry(
    @SerialName("ts") val ts: Double,
    @SerialName("level") val level: String,
    @SerialName("logger") val logger: String,
    @SerialName("message") val message: String,
    @SerialName("exc") val exc: String? = null,
)

/**
 * The in-memory log buffer behind `GET /logs`, plus the stdout writer.
 *
 * **Two sinks, and the split matters**: stdout carries structured lines at the configured level, while the
 * ring buffer captures EVERYTHING regardless of that level. The reason is the operator workflow — when
 * something goes wrong you want the debug lines that were already emitted, and raising the level afterwards
 * cannot retrieve them.
 *
 * Hand-written rather than routed through SLF4J/Logback, for the same reason the JWT is hand-rolled: the
 * ring buffer is the deliverable here, an appender wrapping it would be more configuration than code, and
 * the two-sink rule stays visible in one file instead of spread across a logback.xml. Spring's own startup
 * lines still go through Logback and look different — that is why `logging.level.root` is turned down in
 * `application.properties`, so the service's own JSON stream is the log an operator reads.
 *
 * Port of `service/core/logging.py`.
 */
public object LogRing {

    /**
     * How many entries the ring holds.
     *
     * 5000, matching the reference, so all implementations keep the same amount of history and an operator
     * comparing them is not misled by one having forgotten more. Bounded on purpose: this is an in-memory
     * diagnostic aid, not a log store, and an unbounded buffer in a long-running service is a slow leak
     * nobody planned. At roughly 150 bytes per entry that is under a megabyte.
     */
    public const val CAPACITY: Int = 5000

    public const val DEBUG: Int = 0
    public const val INFO: Int = 1
    public const val WARNING: Int = 2
    public const val ERROR: Int = 3
    public const val CRITICAL: Int = 4

    private val LEVEL_ORDER: Map<String, Int> = mapOf(
        "DEBUG" to DEBUG, "INFO" to INFO, "WARN" to WARNING, "WARNING" to WARNING,
        "ERROR" to ERROR, "CRITICAL" to CRITICAL,
    )

    // An array used circularly rather than a queue: fixed allocation, no per-entry garbage, and the read
    // path is a single pass.
    private val gate = Any()
    private val entries = arrayOfNulls<LogEntry>(CAPACITY)
    private var next = 0
    private var filled = false

    /**
     * The stdout stream, forced to UTF-8.
     *
     * `System.out` on Windows uses the console codepage, which turns every Cyrillic character in a log
     * message into `?`. The conformance CLI hit exactly this and needs the same fix — DEVIATIONS J-10.
     */
    private val stdout = PrintStream(FileOutputStream(FileDescriptor.out), true, "UTF-8")

    private val TIMESTAMP: DateTimeFormatter =
        DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss'Z'").withZone(ZoneOffset.UTC)

    /** The minimum severity written to stdout. The ring ignores it — see the type note. */
    @Volatile
    public var stdoutLevel: Int = INFO

    public fun add(entry: LogEntry): Unit = synchronized(gate) {
        entries[next] = entry
        next = (next + 1) % entries.size
        if (next == 0) {
            filled = true
        }
    }

    /**
     * Records one line: into the ring always, to stdout when it clears [stdoutLevel].
     *
     * The stdout write is inside the same lock as the ring update, because two threads writing partial
     * lines to one stream interleave them — and an interleaved JSON line is worse than a dropped one, since
     * a log parser rejects the whole file.
     */
    public fun log(level: Int, logger: String, message: String, error: Throwable? = null) {
        val name = levelName(level)
        val entry = LogEntry(
            ts = System.currentTimeMillis() / 1000.0,
            level = name,
            logger = logger,
            message = message,
            exc = error?.stackTraceToString(),
        )
        add(entry)
        if (level < stdoutLevel) {
            return
        }
        val line = StringBuilder()
        line.append('{')
        line.append("\"timestamp\":").append(quote(TIMESTAMP.format(Instant.now())))
        line.append(",\"level\":").append(quote(name))
        line.append(",\"logger\":").append(quote(logger))
        line.append(",\"message\":").append(quote(message))
        if (error != null) {
            line.append(",\"exc\":").append(quote(error.stackTraceToString()))
        }
        line.append('}')
        synchronized(gate) { stdout.println(line) }
    }

    /** Escapes a JSON string. Small enough to write out, and it keeps the log path allocation-light. */
    private fun quote(value: String): String {
        val sb = StringBuilder(value.length + 2)
        sb.append('"')
        for (ch in value) {
            when (ch) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                else -> if (ch < ' ') {
                    sb.append("\\u").append("%04x".format(ch.code))
                } else {
                    sb.append(ch)
                }
            }
        }
        sb.append('"')
        return sb.toString()
    }

    /** Entries NEWEST FIRST. */
    private fun snapshot(): List<LogEntry> = synchronized(gate) {
        val count = if (filled) entries.size else next
        val output = ArrayList<LogEntry>(count)
        for (i in 0 until count) {
            val index = (next - 1 - i + entries.size * 2) % entries.size
            entries[index]?.let { output.add(it) }
        }
        output
    }

    /**
     * The most recent entries, optionally filtered.
     *
     * [level] is a MINIMUM severity, not an exact match: asking for warnings should show errors too, which
     * is what an operator means by "show me warnings".
     */
    public fun recent(n: Int, level: String, search: String): List<LogEntry> {
        val floor = LEVEL_ORDER[level.uppercase(Locale.ROOT)] ?: DEBUG
        val needle = search.lowercase(Locale.ROOT)

        val output = ArrayList<LogEntry>(n)
        for (entry in snapshot()) {
            val rank = LEVEL_ORDER[entry.level]
            if (rank != null && rank < floor) {
                continue
            }
            if (needle.isNotEmpty() && !entry.message.lowercase(Locale.ROOT).contains(needle)) {
                continue
            }
            output.add(entry)
            if (output.size >= n) {
                break
            }
        }
        return output
    }

    public fun parseLevel(name: String): Int = when (name.trim().uppercase(Locale.ROOT)) {
        "DEBUG" -> DEBUG
        "WARNING", "WARN" -> WARNING
        "ERROR" -> ERROR
        "CRITICAL" -> CRITICAL
        else -> INFO
    }

    /** The names the Python service uses, so one log pipeline can ingest either. */
    public fun levelName(level: Int): String = when (level) {
        DEBUG -> "DEBUG"
        WARNING -> "WARNING"
        ERROR -> "ERROR"
        CRITICAL -> "CRITICAL"
        else -> "INFO"
    }
}

/**
 * A named logger over [LogRing].
 *
 * The service passes `::info`-style function references around as `(String) -> Unit` — the store and the
 * repositories take exactly that, so nothing below the API layer knows what logging is.
 */
public class ServiceLog(private val name: String) {
    public fun debug(message: String): Unit = LogRing.log(LogRing.DEBUG, name, message)
    public fun info(message: String): Unit = LogRing.log(LogRing.INFO, name, message)
    public fun warn(message: String): Unit = LogRing.log(LogRing.WARNING, name, message)
    public fun error(message: String, cause: Throwable? = null): Unit =
        LogRing.log(LogRing.ERROR, name, message, cause)

    public fun critical(message: String): Unit = LogRing.log(LogRing.CRITICAL, name, message)

    /** The sink form, for the layers that must not know about logging at all. */
    public fun sink(): (String) -> Unit = ::info
}
