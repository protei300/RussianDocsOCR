package net.russiandocs.docproc.viewmodel

/**
 * Display names, field ordering and script hints.
 *
 * D-01: the view model lives on the LIBRARY side, not in the service. The conformance CLI needs it and must
 * not depend on HTTP — which is why this is here and not in a service module, even though the reference puts
 * it in `service/ml/transform.py`.
 */
public object Labels {

    /**
     * Human-readable names.
     *
     * The English UI shows these, so both the `_ru` and `_en` variants of a field map to the SAME display
     * string — the script is carried separately.
     */
    private val FIELD_LABELS = mapOf(
        "Last_name_ru" to "Last name",
        "Last_name_en" to "Last name",
        "First_name_ru" to "First name",
        "First_name_en" to "First name",
        "Middle_name_ru" to "Middle name",
        "Middle_name_en" to "Middle name",
        "Birth_date" to "Date of birth",
        "Birth_place_ru" to "Place of birth",
        "Birth_place_en" to "Place of birth",
        "Sex_ru" to "Sex",
        "Sex_en" to "Sex",
        "Licence_number" to "Document number",
        "Issue_date" to "Date of issue",
        "Expiration_date" to "Valid until",
        "Issue_organization_ru" to "Issuing authority",
        "Issue_organization_en" to "Issuing authority",
        "Issue_organisation_code" to "Authority code",
        "Living_region_ru" to "Place of residence",
        "Living_region_en" to "Place of residence",
        "Driver_class" to "Categories",
        "Face" to "Photo",
        "Signature" to "Signature",
        "Address" to "Registration address",
        "Address_has_handwritten" to "Contains handwriting",
    )

    /**
     * Labels that are detected but never OCR'd.
     *
     * They become `kind: "visual"` so the overlay can draw them without the UI expecting text — free value
     * from the detector, since it finds them anyway.
     */
    private val NON_TEXT_LABELS = setOf("Face", "Signature")

    /**
     * Fields the UI renders monospaced.
     *
     * Numbers and dates only. Monospacing capital Cyrillic — Ш, Щ, Ж, Ы — looks wrong, which is why this is a
     * short allowlist rather than a default.
     */
    private val MONOSPACE_FIELDS = setOf(
        "Licence_number", "Issue_date", "Expiration_date", "Birth_date", "Issue_organisation_code",
    )

    private val PASSPORT_ORDER = listOf(
        "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
        "Birth_date", "Birth_place_ru",
        "Licence_number", "Issue_date", "Expiration_date",
        "Issue_organization_ru", "Issue_organisation_code",
        "Living_region_ru",
    )

    private val EXT_PASSPORT_ORDER = listOf(
        "Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en",
        "Middle_name_ru", "Middle_name_en", "Sex_ru", "Sex_en",
        "Birth_date", "Birth_place_ru", "Birth_place_en",
        "Licence_number", "Issue_date", "Expiration_date",
        "Issue_organization_ru", "Issue_organization_en", "Issue_organisation_code",
        "Living_region_ru", "Living_region_en",
    )

    private val DL_ORDER = listOf(
        "Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en",
        "Middle_name_ru", "Middle_name_en",
        "Birth_date", "Birth_place_ru", "Birth_place_en",
        "Licence_number", "Issue_date", "Expiration_date",
        "Issue_organization_ru", "Issue_organization_en", "Issue_organisation_code",
        "Living_region_ru", "Living_region_en",
        "Driver_class",
    )

    private val SNILS_ORDER = listOf(
        "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
        "Birth_date", "Birth_place_ru",
        "Licence_number", "Issue_date",
    )

    private val ADDR_ORDER = listOf("Address", "Address_has_handwritten")

    private val FIELD_ORDER = mapOf(
        "INTPASSPORT" to PASSPORT_ORDER,
        "INTPASSPORTADDR" to ADDR_ORDER,
        "EXTPASSPORT" to EXT_PASSPORT_ORDER,
        "EXTPASSPORTBIO" to EXT_PASSPORT_ORDER,
        "DL" to DL_ORDER,
        "SNILS" to SNILS_ORDER,
    )

    /** The type without its era suffix. Returns the input unchanged when there is none. */
    public fun baseDocType(docType: String): String {
        val at = docType.lastIndexOf('_')
        return if (at >= 0) docType.substring(0, at) else docType
    }

    /** The era suffix, or null when the label has none. */
    public fun docTypeEra(docType: String): String? {
        val at = docType.lastIndexOf('_')
        return if (at >= 0) docType.substring(at + 1) else null
    }

    public fun fieldDisplay(name: String): String = FIELD_LABELS[name] ?: name

    public fun isNonText(label: String): Boolean = label in NON_TEXT_LABELS

    /**
     * Which font family the UI should use: `num`, `ru` or `en`.
     *
     * The monospace check comes FIRST, because `Birth_date` has no script suffix and would otherwise fall
     * through to the `ru` default.
     */
    public fun fieldScript(name: String): String = when {
        name in MONOSPACE_FIELDS -> "num"
        name.endsWith("_ru") -> "ru"
        name.endsWith("_en") -> "en"
        else -> "ru"
    }

    /**
     * Orders field names for display: canonical order first, then anything unrecognised.
     *
     * **The sort must be stable and the unknown tail sorted by name**, because the reference's dictionary
     * iteration order is insertion order and a port cannot reproduce that — sorting is what makes the result
     * deterministic across languages. An unknown field appearing after the known ones rather than being
     * dropped is deliberate: a new field should show up somewhere rather than vanish.
     */
    public fun orderFields(docType: String, names: Iterable<String>): List<String> {
        val canonical = FIELD_ORDER[baseDocType(docType)] ?: emptyList()
        val rank = canonical.withIndex().associate { (i, name) -> name to i }

        val known = mutableListOf<String>()
        val unknown = mutableListOf<String>()
        for (name in names) {
            (if (rank.containsKey(name)) known else unknown).add(name)
        }

        // sortedBy is stable on the JVM; a primitive sort would not be.
        return known.sortedBy { rank.getValue(it) } + unknown.sorted()
    }
}
