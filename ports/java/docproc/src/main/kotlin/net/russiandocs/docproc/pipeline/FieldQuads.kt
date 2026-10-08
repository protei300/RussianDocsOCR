package net.russiandocs.docproc.pipeline

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.docproc.geometry.Chain
import net.russiandocs.docproc.geometry.Offset
import net.russiandocs.docproc.geometry.PointMap
import net.russiandocs.docproc.geometry.QuarterTurns
import net.russiandocs.docproc.geometry.corners
import net.russiandocs.docproc.imaging.Pt
import kotlin.math.max
import kotlin.math.min

/**
 * Where the read fields and their word patches lie on the input image. Port of `Pipeline._field_quads` and
 * `PipelineResults.field_quads` / `word_quads`.
 *
 * Each quadrilateral is four points from the box's top-left corner, clockwise, in continuous pixels of the image
 * passed to `run`, NOT rounded. [fields] and [words] are null — together, once for the whole run — when the way
 * back to the input image is not known: a caller draws these over a photo, so a quadrilateral that silently landed
 * elsewhere would be worse than none.
 */
public class FieldQuads(
    /** label -> one quadrilateral per detection of the field, in the order the fields were read (top to bottom). */
    public val fields: Map<String, List<List<Pt>>>?,
    /** label -> the quadrilateral of each word patch, in the order of the field's words. */
    public val words: Map<String, List<List<Pt>>>?,
) {
    /** The `quads` conformance stage: `{"fields": {...}, "words": {...}}`, or both null. */
    public fun toJson(): JsonElement {
        fun quad(points: List<Pt>): JsonElement =
            JsonArray(points.map { JsonArray(listOf(JsonPrimitive(it.x), JsonPrimitive(it.y))) })

        fun group(map: Map<String, List<List<Pt>>>?): JsonElement =
            if (map == null) JsonNull else JsonObject(map.mapValues { (_, quads) -> JsonArray(quads.map(::quad)) })
        return JsonObject(mapOf("fields" to group(fields), "words" to group(words)))
    }

    public companion object {
        /**
         * Computes the quadrilaterals of the [kept] fields.
         *
         * A field patch is the canvas (or, on a spread, the page) cut at its box — its [Field.cutMap] says where —
         * and the series/number patch is also turned ([needsLicenceRotation]); a word patch is its box cut from the
         * field patch the way `WordsDetector` cuts it. [toInput] is `PipelineResults.to_input`: the map of the whole
         * canvas back to the input image.
         *
         * A field nobody placed (no cut map) falls back to the canvas box itself, as the reference does for a caller
         * that bypassed `_fields_detector`.
         */
        public fun compute(
            kept: List<KeptField>,
            needsLicenceRotation: Boolean,
            toInput: (List<Pt>) -> List<Pt>?,
        ): FieldQuads {
            val fields = LinkedHashMap<String, MutableList<List<Pt>>>()
            val words = LinkedHashMap<String, MutableList<List<Pt>>>()
            for (k in kept) {
                val box = k.field.box
                val label = box.label
                val patch = k.field.patch
                val rotated = needsLicenceRotation && label == "Licence_number"
                // The patch as CUT: a series/number patch was turned a quarter before the words were found on it.
                val cutW = if (rotated) patch.height else patch.width
                val cutH = if (rotated) patch.width else patch.height

                val toCanvas: PointMap = k.field.cutMap ?: Offset(-box.x1, -box.y1)
                val fw = if (k.field.cutMap != null) cutW.toDouble() else box.x2 - box.x1
                val fh = if (k.field.cutMap != null) cutH.toDouble() else box.y2 - box.y1
                val field = toCanvas.toInput(corners(0.0, 0.0, fw, fh))?.let(toInput)
                if (field == null) {
                    // The way back is not known for this run, and it is not known for any field of it: say so
                    // once, for the whole run, instead of per box.
                    return FieldQuads(null, null)
                }
                fields.getOrPut(label) { ArrayList() } += field
                val wordBoxes = k.wordBoxes
                if (wordBoxes == null) {
                    words.getOrPut(label) { ArrayList() } += field
                    continue
                }
                // the patch the words were found on: h x w of the turned one (w x h as cut)
                val h = patch.height
                val w = patch.width
                var chain = Chain(listOf(toCanvas))
                if (rotated) {
                    chain = chain.then(QuarterTurns(h, w, 1))
                }
                for (wb in wordBoxes) {
                    val wx0 = max(0, wb.x1.toInt())
                    val wy0 = max(0, wb.y1.toInt())
                    val wx1 = min(w, wb.x2.toInt())
                    val wy1 = min(h, wb.y2.toInt())
                    val quad = chain.toInput(corners(wx0.toDouble(), wy0.toDouble(), wx1.toDouble(), wy1.toDouble()))
                    words.getOrPut(label) { ArrayList() } += quad!!.let(toInput)!!
                }
            }
            return FieldQuads(fields, words)
        }
    }
}
