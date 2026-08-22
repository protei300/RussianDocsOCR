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
                    // MRZ is read by the Latin engine and is NOT in neededSplit: the zone is
                    // detected one box per LINE, and each line must reach the engine whole —
                    // splitting it at its filler runs would destroy the fixed 44-character layout
                    // the check digits are computed over.
                    enFields = listOf("Issue_date", "Expiration_date",
                        "Birth_date", "Issue_organisation_code", "MRZ"),
                    // Licence_number is CYRILLIC-routed although it is digits only: the Latin engine
                    // reads the passport's red '3' as '8' at p=0.94..1.00, and the Cyrillic engine
                    // reads the same crops correctly (issue #12). Matches the reference,
                    // OCROptionsINTPassport in pipeline.py.
                    ruFields = listOf("Last_name_ru", "First_name_ru", "Birth_place_ru",
                        "Issue_organization_ru", "Living_region_ru", "Middle_name_ru", "Sex_ru",
                        "Licence_number"),
                    // The internal passport prints its series and number sideways, so the crop is rotated
                    // before OCR. Only this type does.
                    needsLicenceRotation = true,
                )
            }
            if (t.contains("extpassport")) {
                return OcrOptions(
                    neededSplit = listOf("Licence_number", "Birth_place_ru", "Birth_place_en"),
                    // MRZ: Latin engine, never split — see intpassport above.
                    enFields = listOf("Last_name_en", "First_name_en", "Issue_date",
                        "Expiration_date", "Birth_date", "Birth_place_en", "Issue_organization_en",
                        "Living_region_en", "Sex_en", "Issue_organisation_code", "Middle_name_en",
                        "MRZ"),
                    // Licence_number: Cyrillic-routed, same reason as intpassport above.
                    ruFields = listOf("Last_name_ru", "First_name_ru", "Birth_place_ru",
                        "Issue_organization_ru", "Living_region_ru", "Middle_name_ru", "Sex_ru",
                        "Licence_number"),
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
            if (t.contains("birthcert")) {
                // Birth certificates (OCROptionsBIRTHCERT, pipeline.py:156). ONE branch for both
                // blank generations — BIRTHCERT_1998 and BIRTHCERT_2018 (order 167/2018: worded
                // birth date, parents' birth dates, place of issue, 21-digit act number) —
                // because the dispatcher never sees the year suffix.
                //
                // enFields is EMPTY and that is the whole point: every date on these forms is
                // spelled out in Cyrillic («16 декабря 2001», «15 октября 2020 г.»), and the
                // Cyrillic engine reads the 1998 digit-only birth date just as well — the same
                // precedent as the passport Licence_number (issue #12). Licence_number mixes a
                // Roman-numeral series with Cyrillic and «№»; routed Cyrillic as the lesser evil,
                // same as the reference.
                return OcrOptions(
                    neededSplit = listOf("First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                        "Issue_date", "Licence_number",
                        "Father_first_middle_ru", "Mother_first_middle_ru",
                        "Birth_date", "Father_birth_date", "Mother_birth_date",
                        "Issue_place_ru"),
                    enFields = emptyList(),
                    ruFields = listOf("Last_name_ru", "First_name_ru", "Birth_place_ru",
                        "Issue_organization_ru", "Issue_date", "Licence_number",
                        "Father_last_name_ru", "Father_first_middle_ru",
                        "Mother_last_name_ru", "Mother_first_middle_ru",
                        "Birth_date", "Father_birth_date", "Mother_birth_date",
                        "Issue_place_ru", "Act_number"),
                )
            }
            return OcrOptions()
        }
    }
}
