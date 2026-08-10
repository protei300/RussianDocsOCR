package net.russiandocs.service.store

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import net.russiandocs.service.model.ApiKey
import net.russiandocs.service.model.Document
import net.russiandocs.service.model.DocumentStatus
import java.io.File
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset

/**
 * The whitelist of sortable columns, shared by every backend so they cannot drift apart.
 *
 * A whitelist rather than dynamic member lookup: in a SQL backend that difference is an injection vector, and
 * here it is what stops a typo in a query string from silently sorting by nothing.
 */
public object SortColumns {
    public val ALL: Set<String> = setOf(
        "created_at", "filename", "status", "doc_type", "doc_conf", "processing_ms", "size_bytes")
}

/**
 * The filter/sort/page request for a document listing.
 *
 * A type rather than a long parameter list because it crosses the store boundary and grows: a positional call
 * is where "swap date_from and date_to" hides.
 */
public data class DocumentQuery(
    val status: String = "",
    val docType: String = "",
    val search: String = "",
    val dateFrom: String = "",
    val dateTo: String = "",
    val page: Int = 1,
    val pageSize: Int = 20,
    val sortBy: String = "created_at",
    val sortDir: String = "desc",
)

/** The aggregate summary the status page shows. */
@Serializable
public data class StoreStats(
    @SerialName("queued") val queued: Int = 0,
    @SerialName("processing") val processing: Int = 0,
    @SerialName("done") val done: Int = 0,
    @SerialName("failed") val failed: Int = 0,
    @SerialName("total") val total: Int = 0,
    @SerialName("recognised") val recognised: Int = 0,
    @SerialName("avg_processing_ms") val avgProcessingMs: Int? = null,
)

/**
 * Everything the service needs from a storage backend.
 *
 * **SQL SWAP POINT.** Implementing this interface over a real database, and constructing that instead of
 * [FileStore], is the whole migration as far as callers are concerned. Controller and worker code does not
 * change.
 *
 * **Query methods live HERE rather than in the repositories**, deliberately: filtering a list in memory is
 * correct for a few hundred JSON files and wrong for a table, so each backend has to express "the newest
 * twenty matching rows" in its own terms.
 *
 * Port of `service/core/store.py` and `service/core/database.py`.
 */
public interface DocumentStore {
    /**
     * "files" or "sql" — surfaced on the status page, because "why did my data vanish" is answered by this one
     * word.
     */
    public val backend: String

    /** Whether the contents survive a restart. */
    public val isEphemeral: Boolean

    public fun nextDocumentId(): Int
    public fun getRecord(id: Int): Document?
    public fun putRecord(record: Document): Document
    public fun dropRecord(id: Int)
    public fun queryDocuments(query: DocumentQuery): Pair<List<Document>, Int>
    public fun allRecords(): List<Document>
    public fun nextQueuedId(): Int?
    public fun queuePosition(id: Int): Int?
    public fun countByStatus(): Map<String, Int>
    public fun aggregateStats(): StoreStats

    public fun saveResultPayload(id: Int, payload: JsonElement)
    public fun loadResultPayload(id: Int): JsonElement?

    public fun allApiKeys(): List<ApiKey>
    public fun nextApiKeyId(): Int
    public fun putApiKey(key: ApiKey): ApiKey
    public fun dropApiKey(id: Int): Boolean

    public fun allSettings(): Map<String, String>
    public fun setSettings(values: Map<String, String>): Map<String, String>

    /**
     * A plain directory in every backend: binary artifacts stay on the filesystem regardless of where the
     * metadata lives.
     */
    public fun docDir(id: Int): String

    public fun diskUsageBytes(): Long
}

/**
 * The filesystem backend.
 *
 * On-disk layout:
 * ```
 * $DATA_DIR/
 *   documents/42/
 *     record.json     the "row"
 *     original.jpg    exactly the bytes uploaded
 *     canvas.png      the deskewed/rectified canvas
 *     result.json     the full recognition view model
 *   api_keys.json
 *   settings.json
 * ```
 *
 * Four design notes worth reading before changing anything here:
 *
 * - **The index lives in memory; disk is scanned once at startup.** The service is pinned to ONE process — the
 *   pipeline singleton and this index both are — so a shared in-memory index is legitimate rather than a
 *   shortcut.
 * - **Writes are atomic** (temp file plus rename, atomic on NTFS and ext4). A half-written record.json would
 *   survive a crash and poison the next boot.
 * - **Reads return COPIES.** Kotlin data classes make that free — every mutation is a `copy()` — so a caller
 *   editing what it read cannot corrupt the index, and update returns a NEW record the caller rebinds.
 * - **result is not held in the index** — it can be 100 KB of boxes per document. Get-by-id loads it lazily;
 *   list queries never touch it.
 *
 * **Concurrency: ONE lock guards the index and all mutations**, because both the worker and the request
 * handlers write here. Long I/O — writing a 2 MB PNG — happens OUTSIDE the lock; only the rename and the index
 * update are inside. Every public method takes it at most once and calls only `…Locked` helpers beneath it.
 */
public class FileStore(root: String, private val log: (String) -> Unit) : DocumentStore {

    private val root: File = File(root).absoluteFile
    private val docsDir: File = File(this.root, "documents")

    private val gate = Any()
    private val records = LinkedHashMap<Int, Document>()
    private val apiKeys = LinkedHashMap<Int, ApiKey>()
    private var settings = LinkedHashMap<String, String>()
    private var nextDocId = 1
    private var nextKeyId = 1

    init {
        docsDir.mkdirs()
        scan()
    }

    override val backend: String get() = "files"
    override val isEphemeral: Boolean get() = true

    /**
     * Rebuilds the in-memory index from disk. Cheap: N small JSON reads.
     *
     * A corrupt record is SKIPPED with a log line rather than failing the scan. The rest of the scratch data is
     * still perfectly usable, and a service that refuses to start because one of two hundred files is
     * truncated is worse than one that starts with 199.
     */
    private fun scan() {
        val dirs = docsDir.listFiles()?.filter { it.isDirectory }?.sortedBy { it.name } ?: emptyList()

        var loaded = 0
        for (dir in dirs) {
            val file = File(dir, "record.json")
            if (!file.isFile) {
                continue
            }
            try {
                val record = json.decodeFromString(Document.serializer(), file.readText())
                records[record.id] = record
                nextDocId = maxOf(nextDocId, record.id + 1)
                loaded++
            } catch (e: Exception) {
                log("[STORE] skipping unreadable record $file: ${e.message}")
            }
        }

        if (apiKeysPath.isFile) {
            try {
                for (key in json.decodeFromString(
                    kotlinx.serialization.builtins.ListSerializer(ApiKey.serializer()),
                    apiKeysPath.readText(),
                )) {
                    apiKeys[key.id] = key
                    nextKeyId = maxOf(nextKeyId, key.id + 1)
                }
            } catch (e: Exception) {
                log("[STORE] api_keys.json unreadable — starting with none: ${e.message}")
            }
        }

        if (settingsPath.isFile) {
            try {
                settings = LinkedHashMap(json.decodeFromString(
                    kotlinx.serialization.builtins.MapSerializer(
                        String.serializer(),
                        String.serializer(),
                    ),
                    settingsPath.readText(),
                ))
            } catch (e: Exception) {
                log("[STORE] settings.json unreadable — using defaults: ${e.message}")
            }
        }

        if (loaded > 0) {
            log("[STORE] recovered $loaded documents from $docsDir")
        }
    }

    private val apiKeysPath: File get() = File(root, "api_keys.json")
    private val settingsPath: File get() = File(root, "settings.json")

    override fun docDir(id: Int): String = File(docsDir, id.toString()).path

    // -- documents ----------------------------------------------------------

    override fun nextDocumentId(): Int = synchronized(gate) { nextDocId++ }

    override fun allRecords(): List<Document> = synchronized(gate) {
        // Deterministic order regardless of map internals. Callers sort by their own key afterwards, but an
        // unstable base order makes equal keys shuffle between requests — visible in the UI as rows jumping.
        records.values.sortedBy { it.id }
    }

    /**
     * Returns the record with the lazily-stored result attached.
     *
     * A data class is already immutable, so no defensive copy is needed here — which is the one place the JVM
     * port is simpler than Go and .NET rather than more complex.
     */
    override fun getRecord(id: Int): Document? {
        val record = synchronized(gate) { records[id] } ?: return null
        // Loaded OUTSIDE the lock: it is a file read of up to 100 KB, and holding the index lock across it
        // would serialise every other reader for no benefit.
        return record.copy().also { it.result = loadResultPayload(id) }
    }

    /**
     * Persists a record and indexes it.
     *
     * **The FILE IS WRITTEN BEFORE THE INDEX ENTRY**, and the order matters: a record is what makes a document
     * visible to the worker, so indexing it before the bytes exist lets the drain loop claim a document whose
     * file is not written yet.
     */
    override fun putRecord(record: Document): Document {
        val dir = File(docDir(record.id))
        try {
            dir.mkdirs()
            atomicWriteText(File(dir, "record.json"),
                json.encodeToString(Document.serializer(), record))
        } catch (e: Exception) {
            log("[STORE] cannot write record ${record.id}: ${e.message}")
            return record
        }
        synchronized(gate) {
            records[record.id] = record
            nextDocId = maxOf(nextDocId, record.id + 1)
        }
        return record
    }

    override fun dropRecord(id: Int) {
        synchronized(gate) { records.remove(id) }
        // Outside the lock: removing a directory of multi-megabyte artifacts is slow, and nothing else can
        // reach the record now that it is out of the index.
        try {
            File(docDir(id)).deleteRecursively()
        } catch (e: Exception) {
            log("[STORE] could not remove artifacts for $id: ${e.message}")
        }
    }

    // -- api keys -----------------------------------------------------------

    override fun allApiKeys(): List<ApiKey> = synchronized(gate) { apiKeys.values.sortedBy { it.id } }

    override fun nextApiKeyId(): Int = synchronized(gate) { nextKeyId++ }

    override fun putApiKey(key: ApiKey): ApiKey {
        synchronized(gate) {
            apiKeys[key.id] = key
            nextKeyId = maxOf(nextKeyId, key.id + 1)
            flushApiKeysLocked()
        }
        return key
    }

    override fun dropApiKey(id: Int): Boolean = synchronized(gate) {
        if (apiKeys.remove(id) == null) {
            return false
        }
        flushApiKeysLocked()
        true
    }

    /** Assumes the lock is held — see the note on [FileStore] about non-reentrancy. */
    private fun flushApiKeysLocked() {
        atomicWriteText(apiKeysPath, json.encodeToString(
            kotlinx.serialization.builtins.ListSerializer(ApiKey.serializer()),
            apiKeys.values.sortedBy { it.id },
        ))
    }

    // -- settings -----------------------------------------------------------

    override fun allSettings(): Map<String, String> = synchronized(gate) { LinkedHashMap(settings) }

    override fun setSettings(values: Map<String, String>): Map<String, String> = synchronized(gate) {
        settings.putAll(values)
        atomicWriteText(settingsPath, json.encodeToString(
            kotlinx.serialization.builtins.MapSerializer(
                String.serializer(),
                String.serializer(),
            ),
            settings,
        ))
        LinkedHashMap(settings)
    }

    // -- results ------------------------------------------------------------

    override fun saveResultPayload(id: Int, payload: JsonElement) {
        val dir = File(docDir(id))
        dir.mkdirs()
        atomicWriteText(File(dir, "result.json"),
            json.encodeToString(JsonElement.serializer(), payload))
    }

    override fun loadResultPayload(id: Int): JsonElement? {
        val file = File(docDir(id), "result.json")
        if (!file.isFile) {
            return null
        }
        return try {
            json.parseToJsonElement(file.readText())
        } catch (e: Exception) {
            log("[STORE] unreadable result.json for $id: ${e.message}")
            null
        }
    }

    // -- queries ------------------------------------------------------------
    // Implemented over the in-memory index. Correct at this scale (a few hundred records) and honest about it:
    // a SQL backend answers the same questions with real queries.

    override fun queryDocuments(query: DocumentQuery): Pair<List<Document>, Int> {
        var rows: List<Document> = allRecords()

        if (query.status.isNotEmpty()) {
            rows = rows.filter { it.status == query.status }
        }
        // '__none__' means "unrecognised", which is not the same as "no doc_type": a failed document has
        // neither, and the UI offers one filter for both.
        if (query.docType == "__none__") {
            rows = rows.filter { !it.recognised }
        } else if (query.docType.isNotEmpty()) {
            rows = rows.filter { it.docType?.startsWith(query.docType) == true }
        }
        parseDay(query.dateFrom)?.let { start ->
            rows = rows.filter { it.createdAt != null && !it.createdAt.isBefore(start) }
        }
        parseDay(query.dateTo)?.let { end ->
            // Inclusive of the whole named day, which is what a date picker means by "to".
            val limit = end.plusSeconds(86_400)
            rows = rows.filter { it.createdAt != null && it.createdAt.isBefore(limit) }
        }
        val needle = query.search.trim().lowercase()
        if (needle.isNotEmpty()) {
            rows = rows.filter { it.searchText.contains(needle) }
        }

        val total = rows.size

        val column = if (query.sortBy in SortColumns.ALL) query.sortBy else "created_at"
        rows = sortRows(rows, column, desc = query.sortDir != "asc")

        val pageSize = if (query.pageSize > 0) query.pageSize else 20
        val page = if (query.page < 1) 1 else query.page
        val offset = minOf((page - 1) * pageSize, rows.size)
        return rows.subList(offset, minOf(offset + pageSize, rows.size)) to total
    }

    /**
     * Orders by one whitelisted column.
     *
     * **NULLS LAST IN BOTH DIRECTIONS**, matching what a SQL backend must do: a queued document has no
     * doc_conf and must not lead an ascending sort. That is why the ordering is on an (isNull, value) pair
     * rather than on the value alone, and why the null test is never reversed.
     *
     * `sortedWith` is STABLE, which matters: equal keys must keep their previous relative order, or rows jump
     * between refreshes of the list page for no visible reason.
     */
    private fun sortRows(rows: List<Document>, column: String, desc: Boolean): List<Document> {
        val comparator = Comparator<Document> { a, b ->
            val (aNull, aKey) = sortKey(a, column)
            val (bNull, bKey) = sortKey(b, column)
            when {
                aNull != bNull -> if (aNull) 1 else -1   // a null sorts last whichever direction
                else -> if (desc) bKey.compareTo(aKey) else aKey.compareTo(bKey)
            }
        }
        return rows.sortedWith(comparator)
    }

    /**
     * Returns (isNull, comparable) for a column.
     *
     * Every column reduces to a STRING key so one comparator covers dates, numbers and text alike, instead of
     * a type switch inside the comparator. Numbers go through [numKey], which renders them in a
     * lexicographically ordered fixed width; timestamps use the shared format, which sorts correctly as text
     * by construction.
     */
    private fun sortKey(r: Document, column: String): Pair<Boolean, String> = when (column) {
        "filename" -> false to r.filename.lowercase()
        "status" -> false to r.status
        "doc_type" -> if (r.docType == null) true to "" else false to r.docType
        "doc_conf" -> if (r.docConf == null) true to "" else false to numKey(r.docConf)
        "processing_ms" ->
            if (r.processingMs == null) true to "" else false to numKey(r.processingMs.toDouble())
        "size_bytes" -> false to numKey(r.sizeBytes.toDouble())
        else -> if (r.createdAt == null) {
            true to ""
        } else {
            false to (net.russiandocs.service.model.Timestamps.format(r.createdAt) ?: "")
        }
    }

    /**
     * Renders a number as a fixed-width, lexicographically ordered string.
     *
     * This exists so ONE string comparator can order every column, numeric and textual alike. The offset keeps
     * negatives ordered correctly; the width covers every value these columns can hold.
     */
    private fun numKey(v: Double): String = "%020.6f".format(java.util.Locale.ROOT, v + 1e9)

    override fun nextQueuedId(): Int? {
        var best: Document? = null
        for (r in allRecords()) {
            if (r.status != DocumentStatus.QUEUED) {
                continue
            }
            if (best == null || earlier(r, best)) {
                best = r
            }
        }
        return best?.id
    }

    override fun queuePosition(id: Int): Int? {
        val queued = allRecords()
            .filter { it.status == DocumentStatus.QUEUED }
            .sortedWith(compareBy({ it.createdAt ?: Instant.MAX }, { it.id }))
        val index = queued.indexOfFirst { it.id == id }
        return if (index < 0) null else index
    }

    /**
     * FIFO by creation, with the id as the tie-breaker.
     *
     * The tie-break is not decoration: two uploads inside the same clock tick would otherwise have an
     * unspecified order, and the queue would not be FIFO in exactly the case where somebody is testing it by
     * uploading twice quickly.
     */
    private fun earlier(a: Document, b: Document): Boolean {
        val at = a.createdAt
        val bt = b.createdAt
        if (at != null && bt != null && at != bt) {
            return at.isBefore(bt)
        }
        return a.id < b.id
    }

    override fun countByStatus(): Map<String, Int> {
        val counts = linkedMapOf(
            DocumentStatus.QUEUED to 0, DocumentStatus.PROCESSING to 0,
            DocumentStatus.DONE to 0, DocumentStatus.FAILED to 0,
        )
        for (r in allRecords()) {
            counts[r.status] = (counts[r.status] ?: 0) + 1
        }
        return counts
    }

    override fun aggregateStats(): StoreStats {
        val rows = allRecords()
        val counts = countByStatus()

        var sum = 0
        var n = 0
        var recognised = 0
        for (r in rows) {
            if (r.recognised) {
                recognised++
            }
            if (r.status == DocumentStatus.DONE && (r.processingMs ?: 0) > 0) {
                sum += r.processingMs!!
                n++
            }
        }

        return StoreStats(
            queued = counts.getValue(DocumentStatus.QUEUED),
            processing = counts.getValue(DocumentStatus.PROCESSING),
            done = counts.getValue(DocumentStatus.DONE),
            failed = counts.getValue(DocumentStatus.FAILED),
            total = rows.size,
            recognised = recognised,
            avgProcessingMs = if (n > 0) (sum.toDouble() / n + 0.5).toInt() else null,
        )
    }

    override fun diskUsageBytes(): Long = dirSize(docsDir)

    public companion object {
        internal val json = Json {
            prettyPrint = true
            encodeDefaults = true
            explicitNulls = true
            ignoreUnknownKeys = true
        }

        /**
         * Writes text so a crash can never leave a partial file behind.
         *
         * Temp file plus rename, which is atomic on NTFS and ext4. Not a nicety: a truncated record.json
         * survives the crash and poisons the next boot, and the failure then looks like data corruption rather
         * than an interrupted write.
         */
        public fun atomicWriteText(path: File, text: String) {
            atomicWriteBytes(path, text.toByteArray(Charsets.UTF_8))
        }

        public fun atomicWriteBytes(path: File, data: ByteArray) {
            val tmp = File(path.path + ".tmp")
            tmp.writeBytes(data)
            try {
                java.nio.file.Files.move(tmp.toPath(), path.toPath(),
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING)
            } catch (e: Exception) {
                tmp.delete()
                throw e
            }
        }

        /**
         * Empties the data directory. Called before construction when configured.
         *
         * **The CONTENTS go, the directory stays** — and that is not a detail. Removing the directory itself
         * needs write permission on its PARENT, which a non-root container does not have for `/app`, and a
         * directory that is a MOUNT POINT can never be unlinked at all, by anyone. The Go port found this the
         * first time its image ran, as `unlinkat /app/data: permission denied`, with the store then unusable.
         */
        public fun wipe(root: String): Long {
            val abs = File(root).absoluteFile
            if (!abs.isDirectory) {
                return 0
            }
            val size = dirSize(abs)
            abs.listFiles()?.forEach { it.deleteRecursively() }
            return size
        }

        private fun dirSize(root: File): Long {
            if (!root.isDirectory) {
                return 0
            }
            var total = 0L
            root.walkTopDown().forEach {
                // A vanished file mid-walk is normal here (the worker writes while the status page reads), so
                // a failure skips the entry rather than the walk.
                try {
                    if (it.isFile) {
                        total += it.length()
                    }
                } catch (e: Exception) {
                    // skipped on purpose, see above
                }
            }
            return total
        }

        /**
         * Accepts YYYY-MM-DD.
         *
         * **A HALF-TYPED DATE DISABLES THE FILTER** rather than erroring: the list page sends the field on
         * every keystroke, and rejecting "2026-0" would make the page flash an error while somebody is still
         * typing.
         */
        internal fun parseDay(value: String): Instant? {
            if (value.isEmpty()) {
                return null
            }
            return try {
                LocalDate.parse(value).atStartOfDay(ZoneOffset.UTC).toInstant()
            } catch (e: Exception) {
                null
            }
        }
    }
}
