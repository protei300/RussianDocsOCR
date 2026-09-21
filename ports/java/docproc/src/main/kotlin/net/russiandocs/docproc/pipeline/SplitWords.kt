package net.russiandocs.docproc.pipeline

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.modules.Field
import net.russiandocs.docproc.modules.WordsDetector
import net.russiandocs.docproc.postprocess.Box
import net.russiandocs.docproc.tensors.Ops
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * One field's word crops.
 *
 * [wordBoxes] has one entry per DETECTION of the field, and a null entry means "this field needed no
 * splitting" — which is not the same as "the detector found one word".
 */
public class FieldWords(
    public val label: String,
    public val patches: MutableList<Image> = mutableListOf(),
    public val wordBoxes: MutableList<List<Box>?> = mutableListOf(),
) : AutoCloseable {
    override fun close() {
        patches.forEach { it.close() }
        patches.clear()
    }
}

/**
 * A line that was read WHOLE because the word split lost most of it — `PipelineResults.words_fallback`.
 *
 * Part of the contract, not debug output: a measurement that corrects for the missing spaces (a line
 * read whole comes back glued) must apply that correction ONLY to these lines, or it becomes a blanket
 * amnesty. [gap] is the widest empty stretch on the line in typical word widths; null means no words
 * were found at all — the line was one hole, so the ratio has no denominator.
 */
public data class WordsFallback(val field: String, val line: Int, val gap: Double?)

/**
 * A line the guard declined to re-read because it carries no strokes — `PipelineResults.words_no_ink`.
 *
 * Kept apart from [WordsFallback] on purpose: those are lines the guard acted on, these are lines it
 * deliberately left alone. "The check was never asked" and "the check said no" are different facts.
 */
public data class WordsNoInk(val field: String, val line: Int, val ink: Double)

/** What word splitting produced: the per-field crops plus the two guard reports. */
public class SplitResult(
    public val fields: List<FieldWords>,
    public val fallback: List<WordsFallback>,
    public val noInk: List<WordsNoInk>,
)

public object SplitWords {

    /**
     * An empty stretch on a line wider than this many typical words means the split dropped a word,
     * and the line is read whole instead. `Pipeline.WORDS_MAX_GAP` (pipeline.py:1310).
     *
     * Measured over 157 documents, 815 fields: the widest gap on a line that really lost a long word
     * has a median of 2.74 typical word widths, against 0.06 on intact lines — a factor of forty. The
     * SHARE of the line covered by boxes was rejected: it reads print density, and the lowest coverage
     * belongs to a correctly read passport Licence_number (three digit groups with wide gaps).
     */
    public const val WORDS_MAX_GAP: Double = 3.0

    /**
     * Below this much fine-grained detail a line carries no strokes and the fallback must NOT re-read
     * it. `Pipeline.LINE_MIN_INK` (pipeline.py:1350). Variance of the Laplacian: sharp text sits at a
     * median of 1590, blur wider than the stroke at 27; 100 is in that trough. Refusing wrongly loses a
     * repair; accepting wrongly MANUFACTURES a value — «ЛВН» read from an anonymised blur strip.
     */
    public const val LINE_MIN_INK: Double = 100.0

    /** Closes every word crop. Unconditional — see [run]. */
    public fun closeAll(fieldWords: Iterable<FieldWords>?) {
        fieldWords?.forEach { it.close() }
    }

    /**
     * Turns detected fields into per-field word crops. Port of `Pipeline._split_words`.
     *
     * Fields that are not OCR fields for this document type are dropped, duplicates of the must-be-unique
     * fields are dropped, and the rest are either split into words or passed through whole.
     *
     * **A field can be detected TWICE and legitimately so** — the internal passport prints its series and
     * number in two places — in which case the crops are concatenated under one label and the OCR results
     * join. That is why [FieldWords.wordBoxes] is a list of lists.
     *
     * [docType] is the BARE type, without the year suffix: the gap guard is switched off for SNILS by
     * name, and "SNILS_1996" would not match.
     */
    public fun run(
        fields: List<Field>,
        options: OcrOptions,
        words: WordsDetector,
        docType: String,
    ): SplitResult {
        val drop = duplicateFieldIndices(fields)

        var kept = fields.indices.filter { i ->
            i !in drop && options.isOcrField(fields[i].box.label)
        }
        // Top-to-bottom by the box CENTRE (pipeline.py:1546): a multi-line field is labelled one
        // detection per line and its words are concatenated in this order, so this is reading order.
        // A pure y-sort is safe because classes collect only their own boxes. `sortedBy` is stable,
        // which keeps two boxes sharing a centre in detection order — as Python's sort does.
        kept = kept.sortedBy { (fields[it].box.y1 + fields[it].box.y2) / 2 }
        val splitIndices = kept.filter { options.needsSplit(fields[it].box.label) }

        // Splitting runs CONCURRENTLY across fields, one task each. Results are collected POSITIONALLY —
        // see Parallel for why that is correctness rather than style.
        val byIndex = HashMap<Int, Pair<List<Box>, List<Image>>>()
        if (splitIndices.isNotEmpty()) {
            val results = try {
                Parallel.run(splitIndices.map { i ->
                    { words.predictTransform(fields[i].patch) }
                })
            } catch (e: Throwable) {
                // Nothing to release here: Parallel.run rethrows the first failure after every task has
                // finished, and a task that threw produced no crops. A task that SUCCEEDED alongside a
                // failing sibling is the leak the Go port had to fix by returning partial results — this
                // port cannot express that with invokeAll, so the note belongs here: if this ever becomes a
                // real leak, the fix is a Parallel variant that hands back what completed.
                throw e
            }
            for (k in splitIndices.indices) {
                byIndex[splitIndices[k]] = results[k]
            }
        }

        // Gap guard (pipeline.py:1565). A hole on the line wider than a few typical words means the
        // split dropped a word — measured twice: a 9 px crop where the detector returned NO words (the
        // field vanished without a trace), and a line whose longest word («Тракторозаводский», 17
        // characters) was the one missed. Reading the line whole recovers it; the price is that the
        // engine emits no spaces, so the line comes back glued, which is why this is a fallback on a
        // signal and not the default.
        //
        // SNILS is excluded BY CONSTRUCTION, not by hoping the threshold spares it: there the engine is
        // chosen by word-index parity (see Ocr.run), and a line read whole destroys the parity the
        // routing depends on.
        val fallback = mutableListOf<WordsFallback>()
        val noInk = mutableListOf<WordsNoInk>()
        val wholeLine = HashSet<Int>()
        if (docType != "SNILS") {
            // `kept` order is top-to-bottom, and a multi-line field collects its lines in that same
            // order below — so counting per label here gives the line's ordinal WITHIN its field, which
            // is what a reader of the flag can act on.
            val seen = HashMap<String, Int>()
            for (i in kept) {
                val label = fields[i].box.label
                val ordinal = seen[label] ?: 0
                seen[label] = ordinal + 1
                val split = byIndex[i] ?: continue
                val boxes = split.first
                val gap = widestGap(boxes, fields[i].patch.width.toDouble())
                if (gap > WORDS_MAX_GAP && boxes.isEmpty()) {
                    // No boxes at all has two causes, and only one of them is a lost split: the other
                    // is a line with nothing on it. Asked HERE only — where boxes were found the text is
                    // there by definition, and the question would be noise.
                    val ink = lineInk(fields[i].patch)
                    if (ink < LINE_MIN_INK) {
                        noInk += WordsNoInk(label, ordinal, Ops.roundHalfEven(ink, 2))
                        continue
                    }
                }
                if (gap > WORDS_MAX_GAP) {
                    wholeLine += i
                    fallback += WordsFallback(label, ordinal,
                        if (gap.isInfinite()) null else Ops.roundHalfEven(gap, 3))
                }
            }
        }

        val output = mutableListOf<FieldWords>()
        val position = HashMap<String, Int>()
        try {
            for (i in kept) {
                val label = fields[i].box.label

                val patches: MutableList<Image>
                val boxes: List<Box>?
                val split = byIndex[i]
                if (split != null && i in wholeLine) {
                    // The line is read WHOLE: the split's crops are replaced by the field patch, but
                    // the detector's boxes are still reported — that is what the `words.<Field>.bbox`
                    // stage records, and the reference keeps them too (word_bbox_by_idx is untouched).
                    split.second.forEach { it.close() }
                    patches = mutableListOf(fields[i].patch.clone())
                    boxes = split.first
                } else if (split != null) {
                    patches = split.second.toMutableList()
                    boxes = split.first
                    // An empty detection yields an EMPTY word list, exactly as the reference does — it does
                    // NOT fall back to the whole patch. The fallback belongs to the gap guard above.
                } else {
                    // CLONED, not borrowed. The reference aliases the field's own patch here and Python's GC
                    // makes that free; in a port, a borrowed Mat inside a list the caller closes is a double
                    // free that surfaces only in bulk. One copy per unsplit field buys uniform ownership and
                    // removes the special case from closeAll.
                    patches = mutableListOf(fields[i].patch.clone())
                    boxes = null
                }

                val at = position[label]
                if (at != null) {
                    output[at].patches.addAll(patches)
                    output[at].wordBoxes.add(boxes)
                    continue
                }
                position[label] = output.size
                output += FieldWords(label, patches, mutableListOf(boxes))
            }
            return SplitResult(output, fallback, noInk)
        } catch (e: Throwable) {
            closeAll(output)
            throw e
        }
    }

    /**
     * The widest empty stretch on the line, measured in typical word widths. `Pipeline._widest_gap`.
     *
     * A dropped word leaves a hole about as wide as a word; evenly spaced printing does not, however wide
     * the spacing. That is the whole reason this is a ratio to the line's OWN median word width instead of
     * a share of the line. Edges count as gaps too — a word lost from the start or the end of a line leaves
     * the hole at the border, not between boxes. Nothing found at all means the whole line is one hole, so
     * the answer is infinity.
     */
    public fun widestGap(wordBoxes: List<Box>?, lineWidth: Double): Double {
        if (wordBoxes == null || wordBoxes.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }
        if (lineWidth == 0.0) {
            return 0.0
        }
        // `sorted()` over (x1, x2) tuples: by x1, then x2.
        val spans = wordBoxes.map { it.x1 to it.x2 }.sortedWith(compareBy({ it.first }, { it.second }))
        val widths = spans.filter { it.second > it.first }.map { it.second - it.first }
        if (widths.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }
        val typical = widths.sorted()[widths.size / 2]
        if (typical <= 0) {
            return Double.POSITIVE_INFINITY
        }
        var widest = spans[0].first                      // empty stretch on the left
        var end = spans[0].second
        for ((a, b) in spans.drop(1)) {
            widest = max(widest, max(0.0, a - end))
            end = max(end, b)
        }
        widest = max(widest, max(0.0, lineWidth - end))  // and on the right
        return widest / typical
    }

    /**
     * How much fine detail the line crop carries — strokes, not darkness. `Pipeline._line_ink`.
     *
     * Variance of the Laplacian (3x3, `cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)`): a printed
     * stroke is a sharp local change, and a strip that has none has almost no such change left, whatever
     * its overall brightness. The variance is the two-pass population variance, as `np.var` is.
     */
    public fun lineInk(patch: Image): Double {
        if (patch.isEmpty) {
            return 0.0
        }
        Io.toGray(patch).use { gray ->
            val asFloat = Mat()
            val laplacian = Mat()
            try {
                gray.mat.convertTo(asFloat, CvType.CV_32F)
                Imgproc.Laplacian(asFloat, laplacian, CvType.CV_32F)
                val values = FloatArray(laplacian.rows() * laplacian.cols())
                laplacian.get(0, 0, values)
                return Ops.variance(DoubleArray(values.size) { values[it].toDouble() })
            } finally {
                asFloat.release()
                laplacian.release()
            }
        }
    }

    /**
     * Marks all but the highest-confidence detection of each must-be-unique field.
     *
     * The internal passport prints its series and number — and the FMS code — twice, so the detector
     * legitimately returns duplicates and OCR'ing both would read the same value twice.
     *
     * **Strict `>`, so a confidence tie keeps the EARLIER detection.** That matches Python's `max()`, which
     * returns the first maximum. Using `>=` would keep the later one and pick a different crop on any tie.
     */
    private fun duplicateFieldIndices(fields: List<Field>): Set<Int> {
        val uniqueFields = listOf("Licence_number", "Issue_organisation_code")
        val drop = HashSet<Int>()

        for (label in uniqueFields) {
            val indices = fields.indices.filter { fields[it].box.label == label }
            if (indices.size <= 1) {
                continue
            }
            var best = indices[0]
            for (i in indices.drop(1)) {
                if (fields[i].box.conf > fields[best].box.conf) {
                    best = i
                }
            }
            drop += indices.filter { it != best }
        }
        return drop
    }
}
