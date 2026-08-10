package net.russiandocs.docproc.config

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

/**
 * The per-document charsets from `config/ocr_alphabets.json`.
 *
 * **This is NOT the model's alphabet.** The model declares its full alphabet in `model.json`; this table says
 * which of those characters a given script and country is allowed to produce, and the decoder substitutes
 * anything else. Conflating the two silently disables masking.
 */
public object Alphabets {

    private val json = Json { ignoreUnknownKeys = true }

    // Loaded once. The file is small, but it is read per OCR engine and there are four of them.
    private val gate = Any()
    private var cached: JsonObject? = null
    private var cachedRoot: String? = null

    private fun load(root: String): JsonObject = synchronized(gate) {
        cached?.let { if (cachedRoot == root) return it }

        val path = File(root, "document_processing/config/ocr_alphabets.json")
        val text = path.readText(Charsets.UTF_8).trimStart(ModelPaths.UTF8_BOM)
        val table = json.parseToJsonElement(text).jsonObject

        require(table["specials"] != null && table["letters_per_country"] != null) {
            "config: ${path.name} is missing specials or letters_per_country"
        }
        cached = table
        cachedRoot = root
        table
    }

    public fun defaultCountry(root: String, script: String): String {
        val table = load(root)
        return table["default_country"]?.jsonObject?.get(script)?.jsonPrimitive?.content
            ?: throw IllegalArgumentException("config: no default country for script \"$script\"")
    }

    /**
     * The characters a script and country may produce, INCLUDING the shared specials.
     *
     * Returned as a set of STRINGS rather than chars so the decoder can compare code points: the shipped
     * alphabets are all BMP, but a set of `Char` would break silently on the first that is not.
     */
    public fun allowedCharset(root: String, script: String, country: String?): Set<String> {
        val table = load(root)
        val resolved = if (country.isNullOrEmpty()) defaultCountry(root, script) else country

        val byCountry = table["letters_per_country"]?.jsonObject?.get(script)?.jsonObject
            ?: throw IllegalArgumentException("config: unknown script \"$script\"")
        val letters = byCountry[resolved]?.jsonPrimitive?.content
            ?: throw IllegalArgumentException(
                "config: script \"$script\" has no country \"$resolved\"")
        val specials = table["specials"]!!.jsonPrimitive.content

        val set = HashSet<String>()
        for (source in listOf(letters, specials)) {
            var i = 0
            while (i < source.length) {
                val cp = source.codePointAt(i)
                set.add(String(Character.toChars(cp)))
                i += Character.charCount(cp)
            }
        }
        return set
    }
}
