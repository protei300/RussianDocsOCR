package net.russiandocs.docproc.postprocess

import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.tensors.Ops

public data class TextResult(val text: String) : ModelResult

/**
 * Greedy CTC decoding with per-step alphabet masking.
 *
 * The model emits a softmax matrix `[1, T, C]`; this collapses it to a string.
 */
public class OcrProbs(
    alphabet: String,
    allowedChars: Set<String>?,
    private val blankIndex: Int,
) : Postprocessor {

    /**
     * The alphabet, as CODE POINTS.
     *
     * **Not chars, and not bytes.** The alphabet in `model.json` is a UTF-8 string of Cyrillic and Latin
     * letters; indexing it by byte gives mojibake, which is the trap Go had to be warned about. A Kotlin
     * `String` is UTF-16, so `alphabet[i]` would work for the shipped alphabets — every character is in the
     * BMP — but it would break silently the first time one is not. Splitting by code point costs nothing at
     * construction and cannot be wrong.
     */
    private val alphabet: List<String> = alphabet.let { text ->
        require(text.isNotEmpty()) { "postprocess: OCRProbs needs an Alphabet" }
        val out = ArrayList<String>(text.length)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            out += String(Character.toChars(cp))
            i += Character.charCount(cp)
        }
        out
    }

    private val allowed: MutableSet<Int>?
    private val disallowed: MutableSet<Int>?

    init {
        if (allowedChars == null) {
            allowed = null
            disallowed = null
        } else {
            // **Class index is alphabet index PLUS ONE**, because class 0 is the blank. Getting this off by
            // one shifts every decoded character by one position in the alphabet, which produces
            // readable-looking nonsense rather than an error.
            allowed = HashSet()
            disallowed = HashSet()
            for (i in this.alphabet.indices) {
                if (this.alphabet[i] in allowedChars) {
                    allowed.add(i + 1)
                } else {
                    disallowed.add(i + 1)
                }
            }
        }
    }

    override fun apply(output: NdArray, context: Context): ModelResult = TextResult(decode(output))

    /** Greedy decode: argmax per timestep, mask, then collapse repeats and blanks. */
    public fun decode(output: NdArray): String {
        val data = output.asFloat32()
        var shape = output.shape
        if (shape.size == 3) {
            shape = shape.copyOfRange(1, shape.size)
        }
        require(shape.size == 2) {
            "postprocess: OCRProbs expects [T,C] or [1,T,C], got ${NdArray.describe(output.shape)}"
        }

        val steps = shape[0]
        val classes = shape[1]
        require(classes <= alphabet.size + 1) {
            "postprocess: $classes classes exceeds alphabet of ${alphabet.size} plus blank"
        }

        val indices = IntArray(steps)
        val masking = !disallowed.isNullOrEmpty()

        for (t in 0 until steps) {
            val base = t * classes
            val best = Ops.argmaxRow(data, base, classes)

            if (!masking || best == blankIndex || allowed!!.contains(best)) {
                indices[t] = best
                continue
            }

            // **Masking SUBSTITUTES the best allowed non-blank; it does not zero the column.** Zeroing
            // disallowed classes lets the blank win and the character disappears entirely — the reference
            // instead swaps in the nearest permitted letter, which is how `Î` becomes `I` and `І` becomes `И`
            // rather than vanishing.
            var bestAllowed = -1
            var bestScore = Float.NEGATIVE_INFINITY
            for (c in 0 until classes) {
                if (c == blankIndex || disallowed!!.contains(c)) {
                    continue
                }
                if (data[base + c] > bestScore) {
                    bestAllowed = c
                    bestScore = data[base + c]
                }
            }
            indices[t] = if (bestAllowed >= 0) bestAllowed else best
        }

        // Standard CTC collapse: drop repeats of the same class and drop blanks.
        val text = StringBuilder()
        var previous = -1
        for (index in indices) {
            if (index != previous && index != blankIndex) {
                text.append(alphabet[index - 1])
            }
            previous = index
        }
        return text.toString()
    }
}
