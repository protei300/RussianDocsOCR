package net.russiandocs.service.repositories

import java.util.Locale
import net.russiandocs.service.config.Settings
import net.russiandocs.service.settings.SettingDef
import net.russiandocs.service.settings.SettingValidationException
import net.russiandocs.service.settings.SettingsSchema
import net.russiandocs.service.store.DocumentStore

/**
 * The repository for runtime settings.
 *
 * Runtime settings are stored as strings and validated against the schema. The worker reads them fresh on
 * every loop iteration, so an operator change takes effect without a restart — except for the ones flagged
 * `restartRequired`, which are baked into the pipeline's construction.
 */
public class SettingsRepository(
    private val cfg: Settings,
    private val log: (String) -> Unit,
) {

    /**
     * The default for a key AFTER the environment has had its say.
     *
     * **Precedence is STORED VALUE → ENVIRONMENT → SCHEMA DEFAULT**, resolved here so no caller can get it
     * wrong. Every schema key that is also configurable by environment shares its name with the config
     * field, so the two tiers line up by construction rather than through a hand-maintained table.
     *
     * The reference had this layering missing in two different ways and both were real: the worker's value
     * ignored the environment entirely, so `COMPUTE_DEVICE=cpu` was logged and then disregarded; and the
     * settings page read the schema default, so it displayed "auto" for a service actually running on CPU.
     * Bypassing this method reintroduces both.
     */
    public fun effectiveDefault(key: String): String {
        val def: SettingDef = SettingsSchema.BY_KEY[key] ?: return ""
        val envValue = envValueFor(key)
        if (envValue.isEmpty()) {
            return def.default
        }
        return try {
            SettingsSchema.coerce(key, envValue)
        } catch (e: SettingValidationException) {
            // A bad environment value must not take the service down, but silence would hide a deployment
            // mistake behind a plausible default.
            log("[SETTINGS] ignoring invalid value from the environment: " +
                "${key.uppercase(Locale.ROOT)}=\"$envValue\" — using ${def.default} (${e.message})")
            def.default
        }
    }

    /**
     * Maps a schema key onto its config field.
     *
     * An explicit `when` rather than reflection: it is seven lines, it is the same shape in Go and C#, and
     * a reflective version would silently return nothing the moment somebody renames a property.
     */
    private fun envValueFor(key: String): String = when (key) {
        "compute_device" -> cfg.computeDevice
        "ocr_mode" -> cfg.ocrMode
        // Rendered the way coerce will store it, so a value that came from the environment and one that
        // came from the settings page compare equal.
        "docconf" -> if (cfg.docconf == Math.floor(cfg.docconf)) {
            cfg.docconf.toLong().toString()
        } else {
            String.format(Locale.ROOT, "%s", cfg.docconf)
        }
        "img_size" -> cfg.imgSize.toString()
        "job_timeout_sec" -> cfg.jobTimeoutSec.toString()
        "max_retries" -> cfg.maxRetries.toString()
        "log_level" -> cfg.logLevel
        else -> ""
    }

    /** Current values, with environment-or-schema defaults for anything unset. */
    public fun allSettings(db: DocumentStore): Map<String, String> {
        val stored = db.allSettings()
        val output = LinkedHashMap<String, String>()
        for (def in SettingsSchema.ALL) {
            output[def.key] = stored[def.key] ?: effectiveDefault(def.key)
        }
        return output
    }

    /**
     * The stored string for one key, resolved through the same precedence.
     *
     * The worker's accessor, paired with [SettingsSchema.typedInt] and friends. There is no `fallback`
     * parameter for known keys on purpose: the environment layer above is authoritative, precisely so a
     * caller passing the wrong fallback cannot desync the runtime from what the settings page displays.
     */
    public fun settingValue(db: DocumentStore, key: String): String {
        if (key !in SettingsSchema.BY_KEY) {
            return ""
        }
        return db.allSettings()[key] ?: effectiveDefault(key)
    }

    /**
     * Validates and stores. Returns all values and the keys needing a restart.
     *
     * UNKNOWN keys are dropped silently — that is the whitelist doing its job. KNOWN keys with bad values
     * throw, because a UI reporting "saved" while discarding the value is worse than an error message.
     */
    public fun bulkUpdate(
        db: DocumentStore,
        values: Map<String, Any?>,
    ): Pair<Map<String, String>, List<String>> {
        val accepted = LinkedHashMap<String, String>()
        val restartRequired = mutableListOf<String>()
        val current = db.allSettings()

        // Iterated in SCHEMA order rather than map order, so the restart_required list and any error
        // message are deterministic across runs. A nondeterministic error message is a bad thing to debug
        // from a screenshot.
        for (def in SettingsSchema.ALL) {
            if (!values.containsKey(def.key)) {
                continue
            }
            val normalised = SettingsSchema.coerce(def.key, values[def.key])
            accepted[def.key] = normalised

            val previous = current[def.key] ?: effectiveDefault(def.key)
            if (def.restartRequired && normalised != previous) {
                restartRequired += def.key
            }
        }

        if (accepted.isNotEmpty()) {
            db.setSettings(accepted)
        }
        return allSettings(db) to restartRequired
    }
}
