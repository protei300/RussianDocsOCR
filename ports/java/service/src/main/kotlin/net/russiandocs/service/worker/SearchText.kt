package net.russiandocs.service.worker

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import net.russiandocs.docproc.viewmodel.Payload

public object SearchText {

    /**
     * The serialiser that produces the wire form.
     *
     * `encodeDefaults` and `explicitNulls` are both on because the contract says a missing value is
     * `null`, not an absent key: sixty named fields are read by the SPA, and an omitted one becomes
     * `undefined` in a template rather than an empty cell.
     */
    private val json = Json {
        encodeDefaults = true
        explicitNulls = true
    }

    /**
     * The lowercased haystack for the list page's free-text search.
     *
     * Precomputed at write time so filtering never has to parse the stored result blob. In a SQL backend
     * this becomes an indexed computed column.
     *
     * **The OCR values are appended in SORTED KEY ORDER.** Although the haystack is only ever
     * substring-matched — so order cannot change a search RESULT — an order that depends on map internals
     * would differ between two runs over the same document, which makes the stored records
     * non-reproducible and any diff of them noise.
     */
    public fun build(filename: String, payload: Payload): String {
        val parts = mutableListOf(filename)
        payload.docType?.let { parts.add(it) }

        for (key in payload.ocr.keys.sorted()) {
            payload.ocr[key]?.let { parts.add(it) }
        }

        payload.address?.lines?.forEach { line ->
            line.text?.let { parts.add(it) }
        }
        return parts.joinToString(" ").lowercase()
    }

    /**
     * Converts the view model into the generic element the store persists.
     *
     * Via JSON rather than by hand, deliberately: the stored blob must be EXACTLY what the API serves, and
     * a hand-written projection would be a second definition of the wire format, free to drift from the
     * `@SerialName` annotations.
     */
    public fun toElement(payload: Payload): JsonElement {
        val tree = json.encodeToJsonElement(Payload.serializer(), payload) as JsonObject
        // **`debug` is the ONE key that must be ABSENT when null**, while every other null stays present.
        // The reference omits it rather than writing null, and kotlinx-serialization has no per-property
        // override for that, so the key is removed after encoding. The conformance CLI does the same, and
        // this is the reason the two agree: the stored blob is byte-identical to what `recognize` emits.
        return if (payload.debug == null) {
            JsonObject(tree.filterKeys { it != "debug" })
        } else {
            tree
        }
    }

    /** The wire text, for the API layer. Same serialiser, so the two cannot disagree. */
    public fun toJson(payload: Payload): String = json.encodeToString(
        JsonElement.serializer(), toElement(payload))
}
