package net.russiandocs.service.model

import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonElement
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

public object DocumentStatus {
    public const val QUEUED: String = "queued"
    public const val PROCESSING: String = "processing"
    public const val DONE: String = "done"
    public const val FAILED: String = "failed"

    public val VALID: Set<String> = setOf(QUEUED, PROCESSING, DONE, FAILED)
}

/**
 * The ONE place that spells a timestamp.
 *
 * UTC, up to nine fractional digits, trailing `Z`. **The two implementations share a data directory** — the
 * services can be pointed at the same one, and the seed corpus is written by Python — so a record written by
 * either must be readable by the other.
 *
 * `DateTimeFormatter` is used rather than a pattern string, and that is the JVM's version of the trap the
 * .NET port hit twice: a hand-written pattern needs `'T'` and `'Z'` QUOTED (unquoted letters are format
 * specifiers) and would silently accept a precision the platform cannot express. `ISO_INSTANT` with an
 * explicit UTC zone cannot get either wrong.
 */
public object Timestamps {
    private val FORMATTER: DateTimeFormatter =
        DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSSSSS'Z'").withZone(ZoneOffset.UTC)

    public fun format(instant: Instant?): String? = instant?.let { FORMATTER.format(it) }

    /**
     * Parses either spelling.
     *
     * Lenient about the number of fractional digits on the way IN, because a record written by another
     * implementation may carry a different count — and strict on the way out, so this port's own output is
     * always the same shape.
     */
    public fun parse(text: String?): Instant? =
        text?.takeIf { it.isNotEmpty() }?.let { Instant.parse(it) }

    public fun now(): Instant = Instant.now()
}

/** Serialises an [Instant] through [Timestamps], or `null`. */
public object InstantSerializer : KSerializer<Instant?> {
    override val descriptor: SerialDescriptor =
        PrimitiveSerialDescriptor("Instant?", PrimitiveKind.STRING)

    override fun serialize(encoder: Encoder, value: Instant?) {
        if (value == null) {
            encoder.encodeNull()
        } else {
            encoder.encodeString(Timestamps.format(value)!!)
        }
    }

    override fun deserialize(decoder: Decoder): Instant? = Timestamps.parse(decoder.decodeString())
}

/**
 * One uploaded document and everything known about it.
 *
 * **The JSON names are the on-disk record format AND the future SQL column names**, which is why every one is
 * written by hand: four languages have four default naming policies, and a record written by one
 * implementation has to be readable by the others.
 */
@Serializable
public data class Document(
    @SerialName("id") val id: Int = 0,
    /**
     * Sanitised, and for DISPLAY ONLY — never used as a path.
     *
     * On disk the file is always `original.<ext>`, which is what makes a hostile filename harmless rather
     * than a directory-traversal vector.
     */
    @SerialName("filename") val filename: String = "",
    @SerialName("content_type") val contentType: String = "",
    @SerialName("size_bytes") val sizeBytes: Long = 0,
    @SerialName("status") val status: String = DocumentStatus.QUEUED,

    @SerialName("doc_type") val docType: String? = null,
    @SerialName("doc_conf") val docConf: Double? = null,
    @SerialName("recognised") val recognised: Boolean = false,
    @SerialName("field_count") val fieldCount: Int = 0,
    /**
     * Denormalised quality verdicts, so the list page can show them without loading each result blob.
     *
     * Values are whatever the library reports — `good`/`bad` for glare and blur but `REAL`/`FAKE` for the
     * spoofing checks. Clients must NOT assume one vocabulary; the inconsistency is in the library and the
     * wire carries it.
     */
    @SerialName("quality") val quality: Map<String, JsonElement> = emptyMap(),

    @SerialName("device") val device: String? = null,
    @SerialName("processing_ms") val processingMs: Int? = null,
    /** Human-readable failure text. May be in Russian even though the UI is English. */
    @SerialName("error") val error: String? = null,
    /**
     * A machine-readable failure code beside the message.
     *
     * Present precisely because the message may arrive in Russian from the library while the UI is English — a
     * client that needs to branch on the failure cannot parse prose.
     */
    @SerialName("error_code") val errorCode: String? = null,
    @SerialName("retry_count") val retryCount: Int = 0,

    @SerialName("original_ext") val originalExt: String = "",
    @SerialName("original_w") val originalW: Int? = null,
    @SerialName("original_h") val originalH: Int? = null,
    @SerialName("canvas_w") val canvasW: Int? = null,
    @SerialName("canvas_h") val canvasH: Int? = null,
    @SerialName("has_canvas") val hasCanvas: Boolean = false,

    /**
     * Pre-computed lowercase haystack: filename, document type and every OCR value.
     *
     * Denormalised so the list filter never parses a result blob. In SQL this becomes an indexable column,
     * which is the whole point of computing it at write time.
     */
    @SerialName("search_text") val searchText: String = "",

    @SerialName("created_at") @Serializable(with = InstantSerializer::class)
    val createdAt: Instant? = null,
    @SerialName("started_at") @Serializable(with = InstantSerializer::class)
    val startedAt: Instant? = null,
    @SerialName("finished_at") @Serializable(with = InstantSerializer::class)
    val finishedAt: Instant? = null,
    @SerialName("updated_at") @Serializable(with = InstantSerializer::class)
    val updatedAt: Instant? = null,
) {
    /**
     * The full recognition view model.
     *
     * Kept OUT of the record file — it can be 100 KB of boxes per document — and loaded lazily by the
     * repository's get-by-id. `@Transient` because it lives in its own file.
     */
    @Transient
    public var result: JsonElement? = null

    public companion object {
        public fun new(
            id: Int,
            filename: String,
            contentType: String,
            sizeBytes: Long,
            ext: String,
        ): Document {
            val now = Timestamps.now()
            return Document(
                id = id,
                filename = filename,
                contentType = contentType,
                sizeBytes = sizeBytes,
                originalExt = ext,
                status = DocumentStatus.QUEUED,
                createdAt = now,
                updatedAt = now,
            )
        }
    }
}

/**
 * An API key. Only the HASH is stored; the plaintext is shown once, at creation.
 *
 * The same reasoning as any password store: a leaked data directory must not hand over working credentials.
 */
@Serializable
public data class ApiKey(
    @SerialName("id") val id: Int = 0,
    @SerialName("label") val label: String = "",
    /** A short display prefix, so a key can be recognised without being revealed. */
    @SerialName("prefix") val prefix: String = "",
    /** sha256 of the key. **Never the key itself.** */
    @SerialName("key_hash") val keyHash: String = "",
    /**
     * The key from the environment, which cannot be deleted.
     *
     * DELETE on it answers 409. Without that rule the service could be left with no way in at all.
     */
    @SerialName("is_default") val isDefault: Boolean = false,
    @SerialName("created_at") @Serializable(with = InstantSerializer::class)
    val createdAt: Instant? = null,
    @SerialName("last_used_at") @Serializable(with = InstantSerializer::class)
    val lastUsedAt: Instant? = null,
) {
    /**
     * What the UI may see. **Never the hash.**
     *
     * A separate projection rather than an annotation on the hash field, because the same type is persisted
     * WITH the hash — one type with two audiences needs two explicit projections, not one annotation that has
     * to be right in both directions.
     */
    public fun public(): Map<String, Any?> = linkedMapOf(
        "id" to id,
        "label" to label,
        "prefix" to prefix,
        "masked" to prefix + "••••••••",
        "is_default" to isDefault,
        "created_at" to Timestamps.format(createdAt),
        "last_used_at" to Timestamps.format(lastUsedAt),
    )
}
