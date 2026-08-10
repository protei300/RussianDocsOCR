package net.russiandocs.docproc.viewmodel

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

/**
 * The view model — the detail response the SPA reads.
 *
 * **Every JSON name is written by hand and nothing is omitted.** No naming policy, and `explicitNulls = true`
 * on the serialiser: the SPA reads about sixty named fields, and a missing key is a real defect rather than a
 * tolerance question — it makes a page render blank. A policy that gets fifty-nine names right is worse than
 * none, because the one it misses looks like a typo somewhere else entirely.
 *
 * **kotlinx.serialization OMITS nulls by default**, which is the exact opposite of what this contract needs
 * (J-04). Nullable properties are nullable ON PURPOSE: "absent" must serialise as `null`, not vanish and not
 * become 0 or "" — the SPA distinguishes a field that was not read from one that read as empty.
 *
 * Exactly FOURTEEN top-level keys when debug is off. The count is asserted by the harness, because an extra
 * key is as much a contract break as a missing one.
 */
@Serializable
public data class Payload(
    @SerialName("doc_type") val docType: String? = null,
    @SerialName("doc_type_base") val docTypeBase: String? = null,
    @SerialName("doc_type_era") val docTypeEra: String? = null,
    @SerialName("recognised") val recognised: Boolean = false,
    @SerialName("device") val device: String? = null,
    @SerialName("canvas") val canvas: Canvas = Canvas(),
    @SerialName("coord_space") val coordSpace: String = "canvas",
    @SerialName("coord_space_note") val coordSpaceNote: String = "",
    @SerialName("boxes") val boxes: List<Box> = emptyList(),
    @SerialName("fields") val fields: List<Field> = emptyList(),
    @SerialName("ocr") val ocr: Map<String, String> = emptyMap(),
    /**
     * Heterogeneous by contract — `good`/`bad` for glare and blur, `REAL`/`FAKE` for spoofing, and a NUMBER
     * for DocConf. A [JsonElement] rather than a typed map, because no single Kotlin type expresses that and
     * normalising it would change the wire format.
     */
    @SerialName("quality") val quality: Map<String, JsonElement> = emptyMap(),
    @SerialName("timings") val timings: Map<String, Double> = emptyMap(),
    /** Null for every type except INTPASSPORTADDR, which this port does not implement. */
    @SerialName("address") val address: Address? = null,
    /**
     * Only present under `?include=debug` — the one key that IS omitted when absent, because the reference
     * omits it rather than sending null. The builder therefore emits two different payload shapes rather than
     * relying on a serialiser setting that would apply to every field.
     */
    @SerialName("debug") val debug: Debug? = null,
)

@Serializable
public data class Canvas(
    @SerialName("width") val width: Int? = null,
    @SerialName("height") val height: Int? = null,
    /**
     * True when recognition short-circuited and there is no corrected canvas.
     *
     * The SPA then shows the original upload and hides the overlay, because drawing canvas-space boxes on the
     * original would be wrong by the whole perspective transform.
     */
    @SerialName("is_fallback") val isFallback: Boolean = false,
)

@Serializable
public data class Box(
    @SerialName("id") val id: String = "",
    @SerialName("label") val label: String = "",
    @SerialName("display") val display: String = "",
    /** `"text"` or `"visual"` — Face and Signature are the visual ones. */
    @SerialName("kind") val kind: String = "text",
    @SerialName("x1") val x1: Int? = null,
    @SerialName("y1") val y1: Int? = null,
    @SerialName("x2") val x2: Int? = null,
    @SerialName("y2") val y2: Int? = null,
    @SerialName("conf") val conf: Double? = null,
    @SerialName("cls") val cls: Int? = null,
    @SerialName("text") val text: String? = null,
    /** True on a duplicate detection of an OCR'd field that does NOT own the text. */
    @SerialName("ambiguous") val ambiguous: Boolean = false,
)

@Serializable
public data class Field(
    @SerialName("name") val name: String = "",
    @SerialName("display") val display: String = "",
    @SerialName("value") val value: String? = null,
    @SerialName("script") val script: String = "ru",
    @SerialName("conf") val conf: Double? = null,
    /**
     * Every box carrying this label, in detection order.
     *
     * A list rather than one id: a field can be detected twice — the internal passport prints its series in
     * two places — and split fields like `Birth_place_ru` legitimately span several boxes.
     */
    @SerialName("box_ids") val boxIds: List<String> = emptyList(),
)

@Serializable
public data class Debug(
    @SerialName("doc_outline") val docOutline: DocOutline = DocOutline(),
)

@Serializable
public data class DocOutline(
    /**
     * `prewarp`, and the tag is load-bearing: these coordinates are in the space BEFORE the perspective
     * correction, so drawing them on the canvas would be wrong by the whole transform.
     */
    @SerialName("coord_space") val coordSpace: String = "prewarp",
    @SerialName("polygon") val polygon: List<List<List<Int>>>? = null,
)

@Serializable
public data class Address(
    @SerialName("aligned") val aligned: Boolean = false,
    @SerialName("lines") val lines: List<AddressLine> = emptyList(),
)

@Serializable
public data class AddressLine(
    @SerialName("id") val id: String = "",
    @SerialName("kind") val kind: String? = null,
    @SerialName("text") val text: String? = null,
    @SerialName("p_handwritten") val pHandwritten: Double? = null,
    @SerialName("obbox") val obbox: Obbox? = null,
)

@Serializable
public data class Obbox(
    @SerialName("cx") val cx: Double? = null,
    @SerialName("cy") val cy: Double? = null,
    @SerialName("w") val w: Double? = null,
    @SerialName("h") val h: Double? = null,
    @SerialName("angle_rad") val angleRad: Double? = null,
    @SerialName("conf") val conf: Double? = null,
    @SerialName("label") val label: String? = null,
)
