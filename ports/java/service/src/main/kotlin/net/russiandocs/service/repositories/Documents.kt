package net.russiandocs.service.repositories

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import net.russiandocs.service.model.Document
import net.russiandocs.service.model.DocumentStatus
import net.russiandocs.service.model.Timestamps
import net.russiandocs.service.store.DocumentQuery
import net.russiandocs.service.store.DocumentStore
import net.russiandocs.service.store.StoreStats

/**
 * The repository functions for documents: query, create, mutate.
 *
 * **These signatures ARE the migration contract.** They are copied from the reference's
 * `service/repositories` package, deliberately, and the whole point of the layer is that swapping the store
 * implementation underneath changes nothing above it.
 *
 * (The path is spelled without a trailing wildcard on purpose: **Kotlin NESTS block comments**, so a `/`
 * followed by `*` inside KDoc opens a comment that never closes, and the file fails to parse at its last
 * line with no hint about the real location. C# and Go do not nest, so this trap belongs to this port —
 * DEVIATIONS J-09.)
 *
 * Thin by design. Every function takes the store first and delegates the actual query to it, because the
 * backends must express the same question differently — in-memory filtering over JSON files versus real
 * SQL. What lives here is the genuinely shared part: validation, timestamp rules, and the denormalisation
 * performed when a result is saved.
 *
 * Mutating functions return a NEW record; callers rebind (`record = Documents.update(db, record) { … }`).
 * On the JVM this is free rather than defensive: [Document] is a data class, so every mutation is a
 * `copy()` and the record a caller holds can never be edited into storage by accident.
 */
public object Documents {

    /** The statuses that mean "the list page should keep polling". */
    public val ACTIVE_STATUSES: Set<String> = setOf(DocumentStatus.QUEUED, DocumentStatus.PROCESSING)

    /** One page of matching records plus the unpaged total. */
    public fun getAll(db: DocumentStore, query: DocumentQuery): Pair<List<Document>, Int> =
        db.queryDocuments(query)

    /** The full record, including the recognition result. */
    public fun getById(db: DocumentStore, id: Int): Document? = db.getRecord(id)

    /**
     * Claims an id WITHOUT inserting a row yet.
     *
     * This exists so a caller can write the upload's bytes BEFORE the document becomes visible to the
     * worker. Inserting first looks harmless and is a real race: the row lands in `queued`, the drain loop
     * runs on its own schedule, and if it claims the document in the window before the file is written the
     * job fails with "has no stored original" — a good upload reported as a failed document.
     */
    public fun reserveId(db: DocumentStore): Int = db.nextDocumentId()

    /** Inserts a record. Pass an id from [reserveId] when artifacts came first. */
    public fun create(db: DocumentStore, record: Document): Document = db.putRecord(record)

    /**
     * Applies a mutation to a copy and persists it.
     *
     * `updatedAt` is stamped here, once, so no caller can forget it. `result` is carried across because it
     * is stored separately: a plain field update must not look like a request to clear it.
     */
    public fun update(
        db: DocumentStore,
        record: Document,
        mutate: (Document) -> Document,
    ): Document {
        val next = mutate(record).copy(updatedAt = Timestamps.now())
        next.result = record.result
        return db.putRecord(next)
    }

    /**
     * Moves a document between statuses and stamps the matching timestamp.
     *
     * The status is VALIDATED rather than trusted: it reaches the store, the wire and the SPA's badge
     * classes, and an invented value would render as an unstyled row somebody then reports as a UI bug.
     */
    public fun updateStatus(
        db: DocumentStore,
        record: Document,
        status: String,
        errorText: String?,
        errorCode: String?,
    ): Document {
        require(status in DocumentStatus.VALID) { "repo: invalid status \"$status\"" }
        return update(db, record) { d ->
            val stamped = d.copy(status = status, error = errorText, errorCode = errorCode)
            when (status) {
                DocumentStatus.PROCESSING -> stamped.copy(startedAt = Timestamps.now())
                DocumentStatus.DONE, DocumentStatus.FAILED ->
                    stamped.copy(finishedAt = Timestamps.now())
                else -> stamped
            }
        }
    }

    /**
     * Stores the view model and denormalises the columns the list page needs.
     *
     * **The denormalisation IS the point**: without it, filtering or sorting the log means opening every
     * result blob on every keystroke.
     */
    public fun saveResult(
        db: DocumentStore,
        record: Document,
        payload: JsonElement,
        searchText: String,
        processingMs: Int,
    ): Document {
        db.saveResultPayload(record.id, payload)

        val root = payload as? JsonObject ?: JsonObject(emptyMap())
        val quality = root["quality"] as? JsonObject
        val canvas = root["canvas"] as? JsonObject

        // DocConf is lifted OUT of the quality map into its own column, because the list page sorts by
        // it. The remaining keys stay together: they are verdict strings with no single vocabulary
        // ('good'/'bad' and 'REAL'/'FAKE'), so a column each would invite a client to assume otherwise.
        var docConf: Double? = null
        val trimmedQuality = LinkedHashMap<String, JsonElement>()
        quality?.forEach { (key, value) ->
            if (key == "DocConf") {
                docConf = asDouble(value)
            } else {
                trimmedQuality[key] = value
            }
        }

        val fieldCount = (root["fields"] as? kotlinx.serialization.json.JsonArray)?.size ?: 0
        val recognised = asBoolean(root["recognised"]) ?: false
        val docType = asStringOrNull(root["doc_type"])
        val device = asStringOrNull(root["device"])
        val canvasW = canvas?.let { asInt(it["width"]) }
        val canvasH = canvas?.let { asInt(it["height"]) }

        return update(db, record) { d ->
            d.copy(
                status = DocumentStatus.DONE,
                error = null,
                errorCode = null,
                docType = docType,
                docConf = docConf,
                quality = trimmedQuality,
                recognised = recognised,
                fieldCount = fieldCount,
                device = device,
                processingMs = processingMs,
                canvasW = canvasW,
                canvasH = canvasH,
                hasCanvas = canvasW != null,
                searchText = searchText,
                finishedAt = Timestamps.now(),
            )
        }
    }

    /**
     * Resets a document for another attempt, clearing the previous outcome.
     *
     * `retryCount` goes back to zero because this is an OPERATOR action, not an automatic retry: a human
     * asking for a reprocess should get the full retry budget, not whatever was left.
     */
    public fun requeue(db: DocumentStore, record: Document): Document =
        update(db, record) { d ->
            d.copy(
                status = DocumentStatus.QUEUED,
                retryCount = 0,
                error = null,
                errorCode = null,
                startedAt = null,
                finishedAt = null,
            )
        }

    public fun delete(db: DocumentStore, record: Document): Unit = db.dropRecord(record.id)

    public fun nextQueued(db: DocumentStore): Int? = db.nextQueuedId()

    public fun queuePosition(db: DocumentStore, id: Int): Int? = db.queuePosition(id)

    /**
     * Recovers jobs interrupted mid-flight by a restart.
     *
     * Without it a document caught in `processing` when the process died sits there forever: the drain
     * loop only ever claims `queued` rows. Called once at startup.
     */
    public fun resetStaleProcessing(db: DocumentStore): Int {
        var count = 0
        for (record in db.allRecords()) {
            if (record.status != DocumentStatus.PROCESSING) {
                continue
            }
            update(db, record) { it.copy(status = DocumentStatus.QUEUED, startedAt = null) }
            count++
        }
        return count
    }

    public fun countByStatus(db: DocumentStore): Map<String, Int> = db.countByStatus()

    public fun stats(db: DocumentStore): StoreStats = db.aggregateStats()

    // -- JSON coercion ------------------------------------------------------
    // The view model arrives as a JsonElement because it round-trips through JSON, where a number has no
    // declared type. These helpers are the one place that knows it, so no caller has to guess whether a
    // field is int or double.

    private fun asDouble(node: JsonElement?): Double? =
        runCatching { node?.jsonPrimitive?.doubleOrNull }.getOrNull()

    private fun asInt(node: JsonElement?): Int? = asDouble(node)?.toInt()

    private fun asBoolean(node: JsonElement?): Boolean? =
        runCatching { node?.jsonPrimitive?.booleanOrNull }.getOrNull()

    /** An empty string reads as absent, matching the reference. */
    private fun asStringOrNull(node: JsonElement?): String? =
        runCatching { node?.jsonPrimitive?.content }.getOrNull()?.takeIf { it.isNotEmpty() }
}
