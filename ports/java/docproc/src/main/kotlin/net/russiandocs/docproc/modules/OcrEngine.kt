package net.russiandocs.docproc.modules

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.inference.Session
import net.russiandocs.docproc.models.Loader
import net.russiandocs.docproc.models.ModelConfig
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.pipeline.OcrTier
import net.russiandocs.docproc.postprocess.OcrProbs
import net.russiandocs.docproc.preprocess.Preprocessor
import java.io.File

/**
 * One OCR engine. **ONE type for both scripts**, with a `script` field.
 *
 * D-11: the reference has `OCRCyrillic` and `OCRLatin` as separate classes, but they share no state and
 * override nothing — the only difference is which artifact they load and which corrections they apply.
 * Keeping two copies of the field lists in each of four languages is four extra places for them to drift.
 */
public class OcrEngine private constructor(
    private val script: String,
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
    configKey: String,
) : AutoCloseable {

    private val pre: Preprocessor
    private val decoder: OcrProbs
    private val session: Session

    init {
        val dir = File(ModelPaths.resolve(root, paths, configKey), "ONNX").path
        val config = ModelConfig.load(dir)

        pre = Loader.newPreprocessor(config.inputs[0])

        // From the switch, with `root` so it can resolve the ALLOWED charset from ocr_alphabets.json. The
        // model's FULL alphabet comes from model.json and the allowed subset from that table — two different
        // things, and passing the full alphabet as the allowed set would disable masking with no error at all.
        decoder = Loader.newPostprocessor(config.outputs[0], config.dir, root) as? OcrProbs
            ?: throw IllegalArgumentException(
                "modules: $configKey output 0 is not an OCR decoder (${config.outputs[0].type})")

        session = Session(config.modelPath, device, threads)
    }

    /** Decodes one word crop. */
    public fun predict(word: Image): String {
        val (tensor, _) = pre.apply(word)
        val raw = session.run(listOf(tensor))
        return decoder.decode(raw[0])
    }

    /**
     * The per-field text corrections.
     *
     * Dispatched on the SCRIPT first and the field name second, matching the reference. A Cyrillic engine
     * never applies the date normaliser even to a date field, because dates are routed to the Latin engine —
     * except in SNILS, where the month is a Russian word and the parity rule sends odd words to Cyrillic
     * anyway.
     */
    public fun fixErrors(fieldType: String, text: String): String {
        if (script == "cyrillic") {
            if (fieldType == "Sex_ru") {
                return OcrCorrections.checkRusSex(text)
            }
            return if (fieldType in RU_NAME_FIELDS) OcrCorrections.stripEdgeDots(text) else text
        }

        if (fieldType in DATE_FIELDS) {
            // The reference wraps this in `except ValueError: return text`; checkDdmmyyyy folds that in by
            // returning its input unchanged rather than raising.
            return OcrCorrections.checkDdmmyyyy(text)
        }
        return when (fieldType) {
            "Sex_en" -> OcrCorrections.checkEnSex(text)
            "Driver_class" -> OcrCorrections.checkDriverClass(text)
            else -> text
        }
    }

    override fun close(): Unit = session.close()

    public companion object {
        /** Cyrillic name fields, which get their leading dots stripped. */
        private val RU_NAME_FIELDS = listOf(
            "Last_name_ru", "First_name_ru", "Birth_place_ru", "Living_region_ru", "Middle_name_ru",
            "Issue_organization_ru",
        )

        private val DATE_FIELDS = listOf("Issue_date", "Expiration_date", "Birth_date")

        public fun cyrillic(
            root: String,
            paths: Map<String, String>,
            device: Device,
            threads: Int,
            tier: OcrTier,
        ): OcrEngine = OcrEngine("cyrillic", root, paths, device, threads,
            if (tier == OcrTier.ACCURATE) "OCRCyrillicAccurate" else "OCRCyrillicFast")

        public fun latin(
            root: String,
            paths: Map<String, String>,
            device: Device,
            threads: Int,
            tier: OcrTier,
        ): OcrEngine = OcrEngine("latin", root, paths, device, threads,
            if (tier == OcrTier.ACCURATE) "OCRLatinAccurate" else "OCRLatinFast")
    }
}
