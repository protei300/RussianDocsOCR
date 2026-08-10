package net.russiandocs.service.seed

import java.io.File
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive
import net.russiandocs.service.logging.ServiceLog
import net.russiandocs.service.model.Document
import net.russiandocs.service.model.Timestamps
import net.russiandocs.service.repositories.Artifacts
import net.russiandocs.service.repositories.Documents
import net.russiandocs.service.store.DocumentStore

/** One manifest row. Names match the committed `manifest.json` exactly. */
@Serializable
public data class SeedEntry(
    @SerialName("slug") val slug: String = "",
    @SerialName("sample") val sample: String = "",
    @SerialName("filename") val filename: String = "",
    @SerialName("original_ext") val originalExt: String = "",
    @SerialName("content_type") val contentType: String = "",
    @SerialName("size_bytes") val sizeBytes: Long = 0,
    @SerialName("search_text") val searchText: String = "",
)

/**
 * Populates an empty store with pre-computed sample documents.
 *
 * A blank log is a bad first impression and an unhelpful one: there is nothing to click, so nothing
 * demonstrates what the service does. Seeding means the box overlay, the field table and the timings are
 * visible the moment the page loads, across every supported document type.
 *
 * **The results are pre-computed, not re-derived.** `service/seed_data/` holds one finished recognition per
 * document type — the view model, the rendered canvas and a thumbnail — generated once by
 * `service/tools/build_seed_data.py` and committed. Seeding is therefore a FILE COPY: no GPU, no model load,
 * no minute of startup latency, and the same rows every time regardless of the host's hardware.
 *
 * **THIS PORT READS THE SAME DIRECTORY AS THE PYTHON SERVICE.** That is the point: the seeded corpus is ONE
 * artifact with one generator, consumed by every port. A second copy under `ports/java/` would drift from
 * the first the moment recognition changed, and then two services would disagree about what the reference
 * behaviour is while both looked internally consistent.
 *
 * Three rules keep this from becoming a nuisance, all carried over from the reference:
 * - Only into an EMPTY store, so nothing piles up and a deleted document stays deleted.
 * - Only ANONYMISED repository samples. Never a user upload, never a local personal file — everything seeded
 *   here is visible to anyone who can reach the UI.
 * - ONE PER DOCUMENT TYPE, in the manifest's order, so the log shows the breadth of what the library handles
 *   rather than nineteen driving licences.
 *
 * Re-run the builder after any change to recognition, or the seeded rows quietly describe an older version's
 * behaviour. Port of `service/core/seed.py`.
 */
public object SeedData {

    private val json = Json { ignoreUnknownKeys = true }

    public fun dir(repoRoot: String): String = File(File(repoRoot, "service"), "seed_data").path

    /**
     * Inserts the pre-computed samples when the store holds nothing.
     *
     * [limit] caps how many are inserted; 0 means all available. Returns how many were added.
     *
     * **NEVER THROWS**: a service that cannot seed its demo data must still start and accept real uploads.
     * Every failure path logs and continues, and one bad fixture does not stop the others.
     */
    public fun ifEmpty(db: DocumentStore, repoRoot: String?, limit: Int, log: ServiceLog): Int {
        if (repoRoot == null) {
            return 0
        }
        if (db.countByStatus().values.sum() > 0) {
            return 0
        }

        val seedDir = dir(repoRoot)
        var entries: List<SeedEntry> = try {
            json.decodeFromString(
                ListSerializer(SeedEntry.serializer()),
                File(seedDir, "manifest.json").readText(),
            )
        } catch (e: Exception) {
            log.warn("[SEED] no pre-computed data in $seedDir — the log starts empty; run " +
                "`python service/tools/build_seed_data.py` (${e.message})")
            emptyList()
        }
        if (entries.isEmpty()) {
            return 0
        }
        if (limit > 0 && limit < entries.size) {
            entries = entries.subList(0, limit)
        }

        var added = 0
        for (entry in entries) {
            try {
                seedOne(db, repoRoot, seedDir, entry, log)
                added++
            } catch (e: Exception) {
                log.warn("[SEED] skipping fixture ${entry.slug}: ${e.message}")
            }
        }
        log.info("[SEED] inserted $added pre-computed sample document(s)")
        return added
    }

    private fun seedOne(
        db: DocumentStore,
        repoRoot: String,
        seedDir: String,
        entry: SeedEntry,
        log: ServiceLog,
    ) {
        val entryDir = File(seedDir, entry.slug)

        val payload: JsonElement = json.parseToJsonElement(File(entryDir, "result.json").readText())

        // The original is NOT duplicated into the fixture set — it is the repository sample the result was
        // computed from, which is also what keeps the seed data committable.
        val data = File(repoRoot, entry.sample).readBytes()

        // Same BYTES-BEFORE-ROW ordering as an upload. Safe either way here, because seeding finishes before
        // the worker starts — but two orderings for one invariant is how the unsafe one survives a refactor.
        val id = Documents.reserveId(db)
        Artifacts.saveOriginal(db, id, data, entry.originalExt)

        val size = Artifacts.decodeDimensions(data)
        // Timestamps are NOW rather than the build time, so the log's relative dates ("2 minutes ago") stay
        // sane however old the committed fixtures are.
        val now = Timestamps.now()
        var record = Document.new(
            id, entry.filename, entry.contentType, entry.sizeBytes, entry.originalExt)
            .copy(
                originalW = size?.first,
                originalH = size?.second,
                searchText = entry.searchText,
                createdAt = now,
                startedAt = now,
            )
        record = Documents.create(db, record)

        val destination = Artifacts.docDir(db, record.id)
        for (name in listOf("canvas.jpg", "thumb.jpg")) {
            // A missing preview is not fatal: the fields are the product and the picture is a convenience,
            // exactly as in the worker's own canvas-write path.
            val source = File(entryDir, name)
            if (!source.isFile) {
                continue
            }
            try {
                source.copyTo(File(destination, name), overwrite = true)
            } catch (e: Exception) {
                log.warn("[SEED] could not copy $name for ${entry.slug}: ${e.message}")
            }
        }

        // `timings.total` is the library's own value, in SECONDS (spec/viewmodel.md), while the record stores
        // milliseconds.
        val seconds = runCatching {
            ((payload as? JsonObject)?.get("timings") as? JsonObject)
                ?.get("total")?.jsonPrimitive?.doubleOrNull
        }.getOrNull() ?: 0.0
        val processingMs = (seconds * 1000 + 0.5).toInt()
        Documents.saveResult(db, record, payload, entry.searchText, processingMs)
    }
}
