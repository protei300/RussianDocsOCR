package net.russiandocs.service.repositories

import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * The recognition journal: one line per `POST /recognize` call, and nothing about the document.
 *
 * A line says **that** something was recognised, how it ended and how long it took: id, time, outcome, HTTP
 * status, milliseconds. No filename, no document type, no size, no field, no caller, no client address — it
 * must be safe to show to anyone who can see the service at all, and the UI cannot open it: there is nothing
 * behind it to open.
 *
 * Kept **in memory only**, capped, lost at restart. A journal file would be the one thing an in-memory
 * recognition leaves on disk, and an exception is what an auditor asks about first.
 *
 * One instance per process, held by the API server (the service is pinned to a single process). Port of
 * `service/repositories/recognitions.py`; ports/MEMORY.md §4.
 */
public class Recognitions(capacity: Int) {

    public companion object {
        /**
         * `recognised` = a document type was found; `not_recognised` = the image was read but no supported
         * document was in it; `rejected` = refused before recognition; `busy` = no pipeline became free in
         * time (or the runtime was not ready); `error` = anything else.
         */
        public val OUTCOMES: List<String> = listOf("recognised", "not_recognised", "rejected", "busy", "error")

        /** Python's `datetime.isoformat()` with microseconds, `+00:00` written as `Z`. */
        private val AT: DateTimeFormatter =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'").withZone(ZoneOffset.UTC)
    }

    /** One journal line. */
    public data class Event(
        val id: Int,
        val outcome: String,
        val statusCode: Int,
        val processingMs: Int,
        val at: Instant,
    ) {
        /** Exactly these six keys — the contract checks the set. */
        public fun public(): JsonObject = JsonObject(linkedMapOf(
            "id" to JsonPrimitive(id),
            "at" to JsonPrimitive(AT.format(at)),
            "outcome" to JsonPrimitive(outcome),
            "status_code" to JsonPrimitive(statusCode),
            "processing_ms" to JsonPrimitive(processingMs),
            // The UI keys "no drill-down" off this.
            "stored" to JsonPrimitive(false),
        ))
    }

    private val cap = maxOf(1, capacity)
    private val gate = Any()
    private val events = ArrayDeque<Event>()
    private var nextId = 1

    public fun record(outcome: String, statusCode: Int, processingMs: Int): Event = synchronized(gate) {
        val event = Event(nextId++, outcome, statusCode, maxOf(0, processingMs), Instant.now())
        events.addLast(event)
        while (events.size > cap) {
            events.removeFirst()
        }
        event
    }

    /** Newest first. Returns `(page, total)`. */
    public fun recent(limit: Int, offset: Int): Pair<List<Event>, Int> {
        val rows = synchronized(gate) { events.toList() }.asReversed()
        val page = rows.drop(maxOf(0, offset)).take(maxOf(0, limit))
        return page to rows.size
    }

    /** Count per outcome, every outcome present, plus `total`. */
    public fun counts(): Map<String, Int> {
        val rows = synchronized(gate) { events.toList() }
        val out = LinkedHashMap<String, Int>()
        OUTCOMES.forEach { out[it] = 0 }
        rows.forEach { out[it.outcome] = (out[it.outcome] ?: 0) + 1 }
        out["total"] = rows.size
        return out
    }
}
