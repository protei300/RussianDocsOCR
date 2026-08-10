package net.russiandocs.docproc.pipeline

/**
 * Which fields a document type has, which need splitting into words, and which script each uses.
 *
 * Port of the `OCROptions*` class family. A single type with lists rather than a class hierarchy: the
 * subclasses in the reference differ ONLY in their data, so inheritance buys nothing and costs one place per
 * language where a base method could be called by mistake.
 */
public data class OcrOptions(
    val neededSplit: List<String> = emptyList(),
    val enFields: List<String> = emptyList(),
    val ruFields: List<String> = emptyList(),
    val needsLicenceRotation: Boolean = false,
    val hasAddress: Boolean = false,
) {
    public fun isOcrField(label: String): Boolean = label in enFields || label in ruFields

    public fun needsSplit(label: String): Boolean = label in neededSplit

    public companion object {
        /**
         * Splits a document label into its bare type and issuance year.
         *
         * The reference uses `rsplit('_', maxsplit=1)` and would raise on a label without an underscore.
         * This returns an empty year instead — a label the model produced should not crash the pipeline, and
         * every shipped label has the suffix anyway.
         */
        public fun splitDocType(label: String): Pair<String, String> {
            val at = label.lastIndexOf('_')
            return if (at >= 0) label.substring(0, at) to label.substring(at + 1) else label to ""
        }

        /**
         * Builds the options for a document type.
         *
         * **`intpassportaddr` MUST be tested before `intpassport`.** The check is a substring match, so
         * reversing the two sends the registration page down the ordinary text-field path and produces a
         * document with no address and no error. The reference has the same ordering dependency and the same
         * comment.
         *
         * An unrecognised type returns EMPTY options rather than null. The reference returns None here and
         * the next attribute access throws `AttributeError` — a crash two lines later that says nothing
         * about the document type. Empty options mean "no OCR fields", which is what an unknown document
         * deserves.
         */
        public fun forDocType(docType: String): OcrOptions {
            val t = docType.lowercase()

            if (t.contains("intpassportaddr")) {
                return OcrOptions(hasAddress = true)
            }
            if (t.contains("intpassport")) {
                return OcrOptions(
                    neededSplit = listOf("Licence_number", "Birth_place_ru",
                        "Issue_organization_ru"),
                    enFields = listOf("Licence_number", "Issue_date", "Expiration_date",
                        "Birth_date", "Issue_organisation_code"),
                    ruFields = listOf("Last_name_ru", "First_name_ru", "Birth_place_ru",
                        "Issue_organization_ru", "Living_region_ru", "Middle_name_ru", "Sex_ru"),
                    // The internal passport prints its series and number sideways, so the crop is rotated
                    // before OCR. Only this type does.
                    needsLicenceRotation = true,
                )
            }
            if (t.contains("extpassport")) {
                return OcrOptions(
                    neededSplit = listOf("Licence_number", "Birth_place_ru", "Birth_place_en"),
                    enFields = listOf("Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                        "Expiration_date", "Birth_date", "Birth_place_en", "Issue_organization_en",
                        "Living_region_en", "Sex_en", "Issue_organisation_code", "Middle_name_en"),
                    ruFields = listOf("Last_name_ru", "First_name_ru", "Birth_place_ru",
                        "Issue_organization_ru", "Living_region_ru", "Middle_name_ru", "Sex_ru"),
                )
            }
            if (t.contains("dl")) {
                return OcrOptions(
                    neededSplit = listOf("Licence_number", "Driver_class", "Birth_place_ru",
                        "Birth_place_en", "Living_region_ru", "Living_region_en"),
                    enFields = listOf("Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                        "Expiration_date", "Driver_class", "Birth_date", "Birth_place_en",
                        "Issue_organization_en", "Living_region_en", "Issue_organisation_code",
                        "Middle_name_en"),
                    ruFields = listOf("Last_name_ru", "First_name_ru", "Birth_place_ru",
                        "Issue_organization_ru", "Living_region_ru", "Middle_name_ru"),
                )
            }
            if (t.contains("snils")) {
                return OcrOptions(
                    neededSplit = listOf("Last_name_ru", "First_name_ru", "Licence_number",
                        "Issue_date", "Birth_date", "Birth_place_ru", "Middle_name_ru", "Sex_ru"),
                    enFields = listOf("Licence_number", "Issue_date", "Birth_date"),
                    ruFields = listOf("Last_name_ru", "First_name_ru", "Birth_place_ru",
                        "Middle_name_ru", "Sex_ru"),
                )
            }
            return OcrOptions()
        }
    }
}
