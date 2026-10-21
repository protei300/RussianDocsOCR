using RussianDocs.Service.Auth;

namespace RussianDocs.Service.Config;

/// <summary>
/// Which processing mode is in force: documents stored, or recognised in memory only.
///
/// <para>
/// <c>stored</c> (the default) is the service as it always was. <c>memory</c> is for deployments that must
/// not keep personal data at all: the document store, the background worker and every document route are
/// switched off, and <c>POST /recognize</c> is the only way a document gets in — decoded, recognised and
/// answered inside the request, with nothing written.
/// </para>
///
/// <para>
/// **Resolution never fails, and a wrong value fails TOWARD PRIVACY** — the opposite of
/// <see cref="Auth.AuthMode"/>, deliberately. Somebody who typed <c>PROCESSING_MODE=memroy</c> asked for
/// documents not to be stored; quietly storing them would break the one promise the setting exists to make.
/// </para>
///
/// <code>
/// unset, empty   -> stored   (backward compatible: the default is not a mistake)
/// "stored"       -> stored
/// "memory"       -> memory
/// anything else  -> memory, with a reason that is logged and shown on /status
/// </code>
///
/// <para>Port of <c>service/core/processing.py</c>; ports/MEMORY.md §1. The wording is copied from it.</para>
/// </summary>
public static class ProcessingMode
{
    public const string Stored = "stored";
    public const string Memory = "memory";

    private static readonly string[] Modes = [Stored, Memory];

    /// <summary><c>(mode, reason)</c> — the reason is set only when the configured value was not used.</summary>
    public static (string Mode, string? Reason) Resolve(string? configured)
    {
        string raw = (configured ?? "").Trim().ToLowerInvariant();
        if (raw.Length == 0)
        {
            return (Stored, null);
        }
        if (Array.IndexOf(Modes, raw) >= 0)
        {
            return (raw, null);
        }
        return (Memory, $"PROCESSING_MODE={PyRepr.Quote(raw)} is not one of {string.Join(", ", Modes)} — " +
                        "using 'memory' (documents are not stored), the safe reading of a mistyped " +
                        "privacy setting");
    }

    public static (string Mode, string? Reason) Resolve(Settings cfg) => Resolve(cfg.ProcessingMode);

    /// <summary>Whether documents are stored: the document routes, the worker and seeding are on.</summary>
    public static bool StorageEnabled(Settings cfg) => Resolve(cfg).Mode == Stored;
}
