package net.russiandocs.service.settings

import java.util.Locale
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * The types a setting can have.
 *
 * String constants, not an enum, because they go to the UI which switches on them to pick a widget.
 */
public object SettingType {
    public const val BOOL: String = "bool"
    public const val INT: String = "int"
    public const val FLOAT: String = "float"
    public const val CHOICE: String = "choice"
    public const val STR: String = "str"
}

/**
 * One setting's description.
 *
 * Nullable numeric bounds, so "no minimum" is distinguishable from "minimum zero" — a distinction that
 * matters for `docconf`, whose valid range genuinely starts at 0.
 */
@Serializable
public data class SettingDef(
    @SerialName("key") val key: String,
    @SerialName("type") val type: String,
    @SerialName("default") val default: String,
    @SerialName("label") val label: String,
    @SerialName("description") val description: String = "",
    @SerialName("group") val group: String,
    @SerialName("min_value") val minValue: Double? = null,
    @SerialName("max_value") val maxValue: Double? = null,
    @SerialName("choices") val choices: List<String>? = null,
    /**
     * Marks a setting baked into the pipeline's construction.
     *
     * Changing `ocr_mode` in the UI cannot affect a pipeline that is already built, and silently
     * pretending otherwise is worse than saying so.
     */
    @SerialName("restart_required") val restartRequired: Boolean = false,
)

/** Thrown for a rejected value. The message reaches the UI, so it names the bound. */
public class SettingValidationException(message: String) : RuntimeException(message)

/**
 * The server-owned schema for runtime-tunable settings.
 *
 * The server describes its own knobs — type, bounds, choices, help text, group — and the UI renders itself
 * from that. The alternative, a hand-written form, means every new pipeline knob is a frontend change and
 * the defaults end up duplicated on both sides, where they drift.
 *
 * Values are STORED AS STRINGS (the store is JSON; SQL would use a key/value table). Coercion and
 * validation happen here, in one place, on the way IN.
 *
 * Port of `service/core/settings_schema.py`.
 */
public object SettingsSchema {

    /**
     * The ordered list. **ORDER IS THE UI ORDER** — the settings page renders it as given, so this is
     * grouping and sequencing, not just a registry.
     */
    public val ALL: List<SettingDef> = listOf(
        SettingDef(
            key = "compute_device", type = SettingType.CHOICE, default = "auto",
            label = "Compute device",
            description = "GPU is used only when onnxruntime reports a CUDA provider AND the " +
                "pipeline actually builds on it. Applied at startup.",
            group = "Recognition", choices = listOf("auto", "cpu", "gpu"), restartRequired = true,
        ),
        SettingDef(
            key = "ocr_mode", type = SettingType.CHOICE, default = "accurate",
            label = "OCR engine",
            description = "'accurate' is MobileNetV4 (best quality); 'fast' is EdgeNext. " +
                "Baked into the pipeline at construction.",
            group = "Recognition", choices = listOf("accurate", "fast"), restartRequired = true,
        ),
        SettingDef(
            key = "docconf", type = SettingType.FLOAT, default = "0.5",
            label = "Document confidence threshold",
            description = "Minimum confidence for accepting a detected document type.",
            group = "Recognition", minValue = 0.0, maxValue = 1.0,
        ),
        SettingDef(
            key = "img_size", type = SettingType.INT, default = "1500",
            label = "Processing image size",
            description = "Longest side the image is scaled to before inference. Only ever " +
                "downscales — a smaller upload is not enlarged.",
            group = "Recognition", minValue = 640.0, maxValue = 2560.0,
        ),
        SettingDef(
            key = "job_timeout_sec", type = SettingType.INT, default = "120",
            label = "Job timeout (seconds)",
            description = "Typical processing is well under one second; this is a wedge detector, " +
                "not a performance limit.",
            group = "Queue", minValue = 10.0, maxValue = 600.0,
        ),
        SettingDef(
            key = "max_retries", type = SettingType.INT, default = "2", label = "Max retries",
            description = "Applies to transient failures only. A corrupt image fails immediately " +
                "and is never retried.",
            group = "Queue", minValue = 0.0, maxValue = 5.0,
        ),
        SettingDef(
            key = "log_level", type = SettingType.CHOICE, default = "INFO", label = "Log level",
            group = "Service", choices = listOf("DEBUG", "INFO", "WARNING", "ERROR"),
        ),
    )

    /**
     * Indexes the schema. The write whitelist is DERIVED from it rather than duplicated, so a new setting
     * cannot be readable but not writable.
     */
    public val BY_KEY: Map<String, SettingDef> = ALL.associateBy { it.key }

    /** Whether a key may be written through the settings endpoint. */
    public fun isUiKey(key: String): Boolean = key in BY_KEY

    /**
     * Validates against the schema and normalises to the stored string form.
     *
     * **A KNOWN key with a bad value is an ERROR, not a silent drop**: a UI that reports "saved" while
     * discarding the value is worse than one that shows a message.
     */
    public fun coerce(key: String, value: Any?): String {
        val def = BY_KEY[key] ?: throw SettingValidationException("unknown setting \"$key\"")
        val raw = (value?.toString() ?: "").trim()

        return when (def.type) {
            SettingType.BOOL -> when (raw.lowercase(Locale.ROOT)) {
                "1", "true", "yes", "on" -> "1"
                else -> "0"
            }

            SettingType.INT, SettingType.FLOAT -> {
                val number = raw.toDoubleOrNull()
                    ?: throw SettingValidationException("$key must be a number, got \"$raw\"")
                def.minValue?.let {
                    if (number < it) {
                        throw SettingValidationException("$key must be >= ${trim(it)}")
                    }
                }
                def.maxValue?.let {
                    if (number > it) {
                        throw SettingValidationException("$key must be <= ${trim(it)}")
                    }
                }
                // The shortest round-trippable form, so 0.5 stores as "0.5" and not "0.500000" — the
                // stored form is compared against the previous value to decide whether a restart is
                // required, and a formatting change would look like an edit.
                if (def.type == SettingType.INT) number.toInt().toString() else trim(number)
            }

            SettingType.CHOICE -> {
                val choices = def.choices
                if (choices != null && choices.isNotEmpty()) {
                    if (raw in choices) {
                        raw
                    } else {
                        throw SettingValidationException(
                            "$key must be one of ${choices.joinToString(", ")}")
                    }
                } else {
                    raw
                }
            }

            else -> raw
        }
    }

    /**
     * Renders a double without a trailing `.0` on whole values.
     *
     * The invariant locale is not optional: a JVM started with a comma decimal separator would otherwise
     * store `0,5`, which no other implementation — and not even this one's own parser — reads back.
     */
    private fun trim(value: Double): String =
        if (value == Math.floor(value) && !value.isInfinite()) {
            value.toLong().toString()
        } else {
            String.format(Locale.ROOT, "%s", value)
        }

    /**
     * Converts a stored string to the value the worker wants.
     *
     * **A malformed STORED value must not take the worker down** — it falls back to the schema default.
     * That is the opposite policy from [coerce], deliberately: bad input is rejected at the boundary, but a
     * store that somehow holds a bad value must still yield a running service.
     */
    public fun typedInt(key: String, stored: String, fallback: Int): Int =
        typedNumber(key, stored)?.toInt() ?: fallback

    public fun typedFloat(key: String, stored: String, fallback: Double): Double =
        typedNumber(key, stored) ?: fallback

    public fun typedString(key: String, stored: String, fallback: String): String {
        val def = BY_KEY[key] ?: return stored.ifEmpty { fallback }
        val raw = stored.ifEmpty { def.default }
        return raw.ifEmpty { fallback }
    }

    public fun typedBool(key: String, stored: String): Boolean {
        var raw = stored
        if (raw.isEmpty()) {
            raw = BY_KEY[key]?.default ?: ""
        }
        return raw.lowercase(Locale.ROOT) in setOf("1", "true", "yes", "on")
    }

    private fun typedNumber(key: String, stored: String): Double? {
        val def = BY_KEY[key] ?: return null
        val raw = stored.ifEmpty { def.default }
        return raw.toDoubleOrNull() ?: def.default.toDoubleOrNull()
    }
}
