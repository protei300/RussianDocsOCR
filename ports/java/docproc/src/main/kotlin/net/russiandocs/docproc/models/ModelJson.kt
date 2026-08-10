package net.russiandocs.docproc.models

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import net.russiandocs.docproc.config.ModelPaths
import java.io.File
import kotlin.math.abs
import kotlin.math.floor

/**
 * The `model.json` that sits beside every artifact.
 *
 * **Read through the JSON tree rather than into `@Serializable` classes**, and that is a deliberate
 * departure from the .NET port's DTOs. Two reasons, both concrete: `Labels` is sometimes a list of
 * strings and sometimes a list of numbers (the angle head declares `[0, 90, 180, 270]`), which no single
 * typed field expresses; and kotlinx.serialization is strict about unknown keys by default, so a config
 * gaining a field would fail to load rather than being ignored — the opposite of what a loader wants.
 *
 * **Every optional numeric field is nullable**, and that is load-bearing rather than pedantic:
 * `BlankIndex` is legitimately 0 and `Threshold` legitimately defaults to 0.5, so a non-nullable
 * `Double` cannot tell "absent" from "zero" — and the two mean different things.
 */
public class ModelConfig private constructor(
    /** Directory the config was read from. Not part of the JSON. */
    public val dir: String,
    private val root: JsonObject,
) {
    public val name: String get() = string("Name") ?: ""
    public val file: String get() = string("File") ?: ""
    public val modelType: String get() = string("ModelType") ?: ""
    public val runtime: String? get() = string("Runtime")

    public val inputs: List<ModelInput> =
        (root["Inputs"] as? JsonArray)?.map { ModelInput(it.jsonObject) } ?: emptyList()

    public val outputs: List<ModelOutput> =
        (root["Outputs"] as? JsonArray)?.map { ModelOutput(it.jsonObject) } ?: emptyList()

    /** Absolute path to the weights, with separators normalised for this platform. */
    public val modelPath: String
        get() = File(dir, ModelPaths.normaliseSeparators(file)).path

    private fun string(key: String): String? =
        (root[key] as? JsonPrimitive)?.takeIf { it.isString }?.content

    public companion object {
        private val json = Json { ignoreUnknownKeys = true; isLenient = false }

        /**
         * Reads a `model.json`.
         *
         * **The BOM must be stripped before parsing** (D-10). PowerShell's `Set-Content -Encoding utf8`
         * writes one, and the parser then fails on the very first character with a message about invalid
         * JSON rather than about encoding — which sends the reader looking for a syntax error that is
         * not there.
         */
        public fun load(dir: String): ModelConfig {
            val path = File(dir, "model.json")
            val text = path.readText(Charsets.UTF_8).trimStart(ModelPaths.UTF8_BOM)
            val root = json.parseToJsonElement(text).jsonObject

            val config = ModelConfig(dir, root)
            require(config.inputs.isNotEmpty() && config.outputs.isNotEmpty()) {
                "models: $path declares no inputs or no outputs"
            }
            return config
        }
    }
}

public class ModelInput(private val root: JsonObject) {
    public val type: String get() = str("Type") ?: ""
    public val name: String? get() = str("Name")
    public val shape: List<Int>? get() = ints("Shape")
    public val normalization: List<Double>? get() = doubles("Normalization")
    public val paddingSize: List<Int>? get() = ints("PaddingSize")
    public val paddingColor: List<Int>? get() = ints("PaddingColor")
    public val height: Int? get() = (root["Height"] as? JsonPrimitive)?.intOrNull
    public val colorOrder: String? get() = str("ColorOrder")
    public val dtype: String? get() = str("Dtype")

    private fun str(key: String): String? =
        (root[key] as? JsonPrimitive)?.takeIf { it.isString }?.content

    private fun ints(key: String): List<Int>? =
        (root[key] as? JsonArray)?.mapNotNull { it.jsonPrimitive.intOrNull }

    private fun doubles(key: String): List<Double>? =
        (root[key] as? JsonArray)?.mapNotNull { it.jsonPrimitive.doubleOrNull }
}

public class ModelOutput(private val root: JsonObject) {
    public val type: String get() = str("Type") ?: ""
    public val name: String? get() = str("Name")
    public val shape: List<Int>? get() = (root["Shape"] as? JsonArray)
        ?.mapNotNull { it.jsonPrimitive.intOrNull }

    public val threshold: Double? get() = num("Threshold")
    public val iou: Double? get() = num("IOU")
    public val cls: Double? get() = num("CLS")
    public val maskFilter: Double? get() = num("MaskFilter")
    public val metric: String? get() = str("Metric")
    public val centers: String? get() = str("Centers")
    public val alphabet: String? get() = str("Alphabet")
    public val script: String? get() = str("Script")
    public val country: String? get() = str("Country")
    public val blankIndex: Int? get() = (root["BlankIndex"] as? JsonPrimitive)?.intOrNull

    /**
     * Labels as strings, whichever way they were written.
     *
     * Numbers are formatted the way Python's `str()` would — `90`, not `90.0` — because the angle lookup
     * compares label strings, and a formatting difference there turns a valid angle into "not one of the
     * declared labels".
     */
    public fun labelsAsStrings(): List<String> {
        val array = root["Labels"] as? JsonArray ?: return emptyList()
        return array.map { item ->
            val primitive = item as? JsonPrimitive
                ?: throw IllegalArgumentException(
                    "models: Labels contains ${item::class.simpleName}, expected string or number")
            when {
                primitive.isString -> primitive.content
                primitive.doubleOrNull != null -> formatNumber(primitive.double)
                else -> throw IllegalArgumentException(
                    "models: Labels contains $primitive, expected string or number")
            }
        }
    }

    public fun labelsAsInts(): List<Int> = labelsAsStrings().map { label ->
        label.toIntOrNull()
            ?: throw IllegalArgumentException("models: label \"$label\" is not an integer")
    }

    private fun str(key: String): String? =
        (root[key] as? JsonPrimitive)?.takeIf { it.isString }?.content

    private fun num(key: String): Double? = (root[key] as? JsonPrimitive)?.doubleOrNull

    private val JsonPrimitive.double: Double get() = content.toDouble()

    private companion object {
        /** Matches Go's `%g` and Python's `str()` for the integral values used here. */
        fun formatNumber(value: Double): String =
            if (value == floor(value) && abs(value) < 1e15) {
                value.toLong().toString()
            } else {
                value.toString()
            }
    }
}
