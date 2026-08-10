using System.Globalization;
using Microsoft.Extensions.Logging;
using RussianDocs.Service.Settings;
using RussianDocs.Service.Store;

namespace RussianDocs.Service.Repositories;

/// <summary>
/// The repository for runtime settings.
///
/// <para>
/// Runtime settings are stored as strings and validated against the schema. The worker reads them
/// fresh on every loop iteration, so an operator change takes effect without a restart — except for
/// the ones flagged <c>RestartRequired</c>, which are baked into the pipeline's construction.
/// </para>
/// </summary>
public sealed class SettingsRepository(Config.Settings cfg, ILogger log)
{
    /// <summary>
    /// The default for a key AFTER the environment has had its say.
    ///
    /// <para>
    /// **Precedence is STORED VALUE → ENVIRONMENT → SCHEMA DEFAULT**, resolved here so no caller can
    /// get it wrong. Every schema key that is also configurable by environment shares its name with
    /// the config field, so the two tiers line up by construction rather than through a
    /// hand-maintained table.
    /// </para>
    ///
    /// <para>
    /// The reference had this layering missing in two different ways and both were real: the worker's
    /// value ignored the environment entirely, so <c>COMPUTE_DEVICE=cpu</c> was logged and then
    /// disregarded; and the settings page read the schema default, so it displayed "auto" for a
    /// service actually running on CPU. Bypassing this method reintroduces both.
    /// </para>
    /// </summary>
    public string EffectiveDefault(string key)
    {
        if (!SettingsSchema.ByKey.TryGetValue(key, out SettingDef? def))
        {
            return "";
        }
        string envValue = EnvValueFor(key);
        if (envValue.Length == 0)
        {
            return def.Default;
        }
        try
        {
            return SettingsSchema.Coerce(key, envValue);
        }
        catch (SettingValidationException ex)
        {
            // A bad environment value must not take the service down, but silence would hide a
            // deployment mistake behind a plausible default.
            log.LogWarning(
                "[SETTINGS] ignoring invalid value from the environment: {Key}=\"{Value}\" " +
                "— using {Default} ({Error})",
                key.ToUpperInvariant(), envValue, def.Default, ex.Message);
            return def.Default;
        }
    }

    /// <summary>
    /// Maps a schema key onto its config field.
    ///
    /// <para>
    /// An explicit switch rather than reflection: it is seven lines, it is the same shape in Go and
    /// Kotlin, and a reflective version would silently return nothing the moment somebody renames a
    /// property.
    /// </para>
    /// </summary>
    private string EnvValueFor(string key) => key switch
    {
        "compute_device" => cfg.ComputeDevice,
        "ocr_mode" => cfg.OcrMode,
        // Rendered the way Coerce will store it, so a value that came from the environment and one
        // that came from the settings page compare equal.
        "docconf" => cfg.Docconf.ToString(CultureInfo.InvariantCulture),
        "img_size" => cfg.ImgSize.ToString(CultureInfo.InvariantCulture),
        "job_timeout_sec" => cfg.JobTimeoutSec.ToString(CultureInfo.InvariantCulture),
        "max_retries" => cfg.MaxRetries.ToString(CultureInfo.InvariantCulture),
        "log_level" => cfg.LogLevel,
        _ => "",
    };

    /// <summary>Current values, with environment-or-schema defaults for anything unset.</summary>
    public Dictionary<string, string> AllSettings(IDocumentStore db)
    {
        Dictionary<string, string> stored = db.AllSettings();
        var output = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (SettingDef def in SettingsSchema.All)
        {
            output[def.Key] = stored.TryGetValue(def.Key, out string? value)
                ? value
                : EffectiveDefault(def.Key);
        }
        return output;
    }

    /// <summary>
    /// The stored string for one key, resolved through the same precedence.
    ///
    /// <para>
    /// The worker's accessor, paired with <see cref="SettingsSchema.TypedInt"/> and friends. There is
    /// no <c>fallback</c> parameter for known keys on purpose: the environment layer above is
    /// authoritative, precisely so a caller passing the wrong fallback cannot desync the runtime from
    /// what the settings page displays.
    /// </para>
    /// </summary>
    public string SettingValue(IDocumentStore db, string key)
    {
        if (!SettingsSchema.ByKey.ContainsKey(key))
        {
            return "";
        }
        return db.AllSettings().TryGetValue(key, out string? value) ? value : EffectiveDefault(key);
    }

    /// <summary>
    /// Validates and stores. Returns all values and the keys needing a restart.
    ///
    /// <para>
    /// UNKNOWN keys are dropped silently — that is the whitelist doing its job. KNOWN keys with bad
    /// values throw, because a UI reporting "saved" while discarding the value is worse than an error
    /// message.
    /// </para>
    /// </summary>
    public (Dictionary<string, string> Values, List<string> RestartRequired) BulkUpdate(
        IDocumentStore db, IReadOnlyDictionary<string, object?> values)
    {
        var accepted = new Dictionary<string, string>(StringComparer.Ordinal);
        var restartRequired = new List<string>();
        Dictionary<string, string> current = db.AllSettings();

        // Iterated in SCHEMA order rather than dictionary order, so the restart_required list and any
        // error message are deterministic across runs. A nondeterministic error message is a bad
        // thing to debug from a screenshot.
        foreach (SettingDef def in SettingsSchema.All)
        {
            if (!values.TryGetValue(def.Key, out object? value))
            {
                continue;
            }
            string normalised = SettingsSchema.Coerce(def.Key, value);
            accepted[def.Key] = normalised;

            string previous = current.TryGetValue(def.Key, out string? stored)
                ? stored
                : EffectiveDefault(def.Key);
            if (def.RestartRequired && normalised != previous)
            {
                restartRequired.Add(def.Key);
            }
        }

        if (accepted.Count > 0)
        {
            db.SetSettings(accepted);
        }
        return (AllSettings(db), restartRequired);
    }
}
