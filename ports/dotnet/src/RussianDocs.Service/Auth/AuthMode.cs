using System.Globalization;
using System.Text;

namespace RussianDocs.Service.Auth;

/// <summary>
/// Which authentication is in force: the shared PIN, or named accounts.
///
/// <para>
/// **Resolved in ONE function, and nothing else reads the raw setting.** The whole point is that
/// there is one place where an unusable configuration turns into a usable one — an endpoint that read
/// <c>AUTH_MODE</c> itself would half-honour a value the rest of the service had already downgraded.
/// </para>
///
/// <para>
/// **It never throws and never refuses to start.** An existing deployment that pulls this version
/// must keep working exactly as before, and "as before" is the PIN. So every way of getting it wrong
/// — unset, a typo, a mode the storage backend cannot support — resolves to <c>pin</c> with a reason
/// attached. The reason is RETURNED rather than only logged, so <c>/auth/config</c> can show it: a
/// silent downgrade is the failure to avoid, because somebody configured named accounts and nothing in
/// the interface would otherwise say they got a shared four-digit PIN instead.
/// </para>
///
/// <para>
/// The reference has a fourth row, for the Argon2 library being missing. Here Argon2 is a compile-time
/// dependency and cannot be missing, so the row does not exist. Port of
/// <c>service/core/auth.py::resolve_auth_mode</c>; the wording is copied from it.
/// </para>
/// </summary>
public static class AuthMode
{
    public const string Pin = "pin";
    public const string Users = "users";

    private static readonly string[] Modes = [Pin, Users];

    /// <summary>
    /// Returns <c>(mode, downgrade_reason)</c>. <paramref name="storeBackend"/> is
    /// <see cref="Store.IDocumentStore.Backend"/>: named accounts are implemented for the file store
    /// only, so the storage choice decides whether users mode is possible at all — and that has to be
    /// known here, not discovered at the first login attempt.
    /// </summary>
    public static (string Mode, string? DowngradeReason) Resolve(string? raw, string storeBackend)
    {
        string value = (raw ?? "").Trim().ToLowerInvariant();
        if (value.Length == 0)
        {
            return (Pin, null); // unset is not a mistake, it is the default
        }
        if (Array.IndexOf(Modes, value) < 0)
        {
            return (Pin, $"AUTH_MODE={PyRepr.Quote(value)} is not one of {string.Join(", ", Modes)} — " +
                         "falling back to PIN authentication");
        }
        if (value == Users && storeBackend != "files")
        {
            return (Pin, "AUTH_MODE=users is implemented for the temporary file store only; the " +
                         "database backend's user methods are stubs you are expected to implement " +
                         "(see docs/auth.md) — falling back to PIN authentication");
        }
        return (value, null);
    }
}

/// <summary>
/// Python's <c>repr()</c> of a string, for the handful of messages the reference formats with
/// <c>{x!r}</c> — <c>Unknown role 'root'</c>, <c>User 'ADMIN' already exists</c>,
/// <c>AUTH_MODE='bogus' …</c>.
///
/// <para>
/// Reproduced rather than approximated with <c>$"'{x}'"</c> because the messages are the contract,
/// and the two differ exactly where it matters: a value containing a quote or a control character.
/// Single quotes unless the text contains a single quote and no double one; backslash, the chosen
/// quote and control characters escaped; printable non-ASCII (Cyrillic) kept as is.
/// </para>
/// </summary>
public static class PyRepr
{
    public static string Quote(string text)
    {
        char quote = text.Contains('\'') && !text.Contains('"') ? '"' : '\'';
        var builder = new StringBuilder(text.Length + 2);
        builder.Append(quote);
        foreach (char c in text)
        {
            switch (c)
            {
                case '\\': builder.Append(@"\\"); break;
                case '\n': builder.Append(@"\n"); break;
                case '\r': builder.Append(@"\r"); break;
                case '\t': builder.Append(@"\t"); break;
                default:
                    if (c == quote)
                    {
                        builder.Append('\\').Append(c);
                    }
                    else if (c < 0x20 || c == 0x7f)
                    {
                        builder.Append(@"\x").Append(((int)c).ToString("x2", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        builder.Append(c);
                    }
                    break;
            }
        }
        return builder.Append(quote).ToString();
    }
}
