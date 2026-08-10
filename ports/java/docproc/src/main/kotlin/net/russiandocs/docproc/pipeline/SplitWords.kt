package net.russiandocs.docproc.pipeline

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.modules.Field
import net.russiandocs.docproc.modules.WordsDetector
import net.russiandocs.docproc.postprocess.Box

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

public object SplitWords {

    /** Closes every word crop. Unconditional — see [run]. */
    public fun closeAll(fieldWords: Iterable<FieldWords>?) {
        fieldWords?.forEach { it.close() }
    }

    /**
     * Turns detected fields into per-field word crops.
     *
     * Fields that are not OCR fields for this document type are dropped, duplicates of the must-be-unique
     * fields are dropped, and the rest are either split into words or passed through whole.
     *
     * **A field can be detected TWICE and legitimately so** — the internal passport prints its series and
     * number in two places — in which case the crops are concatenated under one label and the OCR results
     * join. That is why [FieldWords.wordBoxes] is a list of lists.
     */
    public fun run(
        fields: List<Field>,
        options: OcrOptions,
        words: WordsDetector,
    ): List<FieldWords> {
        val drop = duplicateFieldIndices(fields)

        val kept = fields.indices.filter { i ->
            i !in drop && options.isOcrField(fields[i].box.label)
        }
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

        val output = mutableListOf<FieldWords>()
        val position = HashMap<String, Int>()
        try {
            for (i in kept) {
                val label = fields[i].box.label

                val patches: MutableList<Image>
                val boxes: List<Box>?
                val split = byIndex[i]
                if (split != null) {
                    patches = split.second.toMutableList()
                    boxes = split.first
                    // An empty detection yields an EMPTY word list, exactly as the reference does — it does
                    // NOT fall back to the whole patch. The fallback belongs to fields never split at all.
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
            return output
        } catch (e: Throwable) {
            closeAll(output)
            throw e
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
