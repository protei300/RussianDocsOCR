using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace RussianDocs.Service.Auth;

/// <summary>
/// Two authentication paths, for two different callers.
///
/// <list type="bullet">
/// <item><b>The website</b> signs in with a PIN and gets a short-lived JWT. One shared operator
/// identity; there are no user accounts.</item>
/// <item><b>Machine callers</b> send an API key in <c>X-API-Key</c>. Keys are managed from the UI at
/// runtime, plus one bootstrap key from the environment.</item>
/// </list>
///
/// <para>
/// Why the split: a PIN is a human affordance and a terrible service credential — four digits,
/// shared, and it would have to be embedded in every integration. An API key is the opposite.
/// Endpoints both kinds of caller use accept either.
/// </para>
///
/// <para>
/// Security notes, honestly:
/// </para>
/// <list type="bullet">
/// <item>Comparison is constant-time. For the PIN that is mostly symbolic against a four-digit
/// space: there is no rate limiting or lockout here, and a PIN is not a defence against an attacker
/// who can reach the port. It keeps honest people out of the browser UI; the NETWORK BOUNDARY is the
/// real control.</item>
/// <item>Only key HASHES are stored. A leaked data directory must not yield working
/// credentials.</item>
/// </list>
///
/// <para>
/// Port of <c>service/core/auth.py</c>. **The JWT is hand-rolled rather than taken from a
/// dependency** — HS256 with two base64url segments and an HMAC is about forty lines, and it keeps
/// this port's dependency list at the two native libraries it genuinely needs. That matters for a
/// reference project somebody has to audit, and it is the same choice the Go port made, so the two
/// files read alike.
/// </para>
/// </summary>
public static class Tokens
{
    /// <summary>
    /// Makes keys greppable in logs and recognisable when pasted somewhere they should not be — the
    /// same reason GitHub uses <c>ghp_</c>.
    /// </summary>
    public const string KeyPrefix = "rdk_";

    /// <summary><c>rdk_</c> plus six characters: enough to tell keys apart.</summary>
    public const int KeyPrefixDisplayLen = 10;

    /// <summary>What auth needs from the environment tier.</summary>
    public sealed record Config
    {
        public string Pin { get; init; } = "";
        public string JwtSecret { get; init; } = "";
        public string JwtAlgorithm { get; init; } = "HS256";
        public int JwtExpireMinutes { get; init; } = 480;
        public string DefaultApiKey { get; init; } = "";
    }

    /// <summary>The JWT payload. Only what is actually used.</summary>
    public sealed class Claims
    {
        [JsonPropertyName("sub")] public string Sub { get; set; } = "";
        [JsonPropertyName("exp")] public long Exp { get; set; }
    }

    /// <summary>Signs a JWT valid for the configured window.</summary>
    public static string CreateAccessToken(Config cfg, string subject)
    {
        if (cfg.JwtAlgorithm.Length > 0 && cfg.JwtAlgorithm != "HS256")
        {
            // Refused rather than silently downgraded: a caller who configured RS256 and got HS256
            // would believe they had asymmetric signing.
            throw new InvalidOperationException(
                $"auth: unsupported JWT algorithm \"{cfg.JwtAlgorithm}\" (only HS256)");
        }
        byte[] header = JsonSerializer.SerializeToUtf8Bytes(
            new Dictionary<string, string> { ["alg"] = "HS256", ["typ"] = "JWT" });
        byte[] claims = JsonSerializer.SerializeToUtf8Bytes(new Claims
        {
            Sub = subject,
            Exp = DateTimeOffset.UtcNow.AddMinutes(cfg.JwtExpireMinutes).ToUnixTimeSeconds(),
        });
        string signing = B64(header) + "." + B64(claims);
        return signing + "." + B64(Sign(signing, cfg.JwtSecret));
    }

    /// <summary>
    /// Returns the claims, or <c>null</c> for anything invalid or expired.
    ///
    /// <para>
    /// **The signature is verified BEFORE the claims are parsed**, and with a constant-time compare.
    /// Parsing first would mean acting on attacker-controlled JSON; a plain equality test on the MAC
    /// leaks how much of it matched.
    /// </para>
    /// </summary>
    public static Claims? DecodeAccessToken(Config cfg, string token)
    {
        string[] parts = token.Split('.');
        if (parts.Length != 3)
        {
            return null;
        }
        string signing = parts[0] + "." + parts[1];
        byte[] want = Sign(signing, cfg.JwtSecret);
        if (!TryUnb64(parts[2], out byte[] got) ||
            !CryptographicOperations.FixedTimeEquals(want, got))
        {
            return null;
        }

        if (!TryUnb64(parts[1], out byte[] raw))
        {
            return null;
        }
        Claims? claims;
        try
        {
            claims = JsonSerializer.Deserialize<Claims>(raw);
        }
        catch (JsonException)
        {
            return null;
        }
        if (claims is null)
        {
            return null;
        }
        if (claims.Exp != 0 && DateTimeOffset.UtcNow.ToUnixTimeSeconds() >= claims.Exp)
        {
            return null;
        }
        return claims;
    }

    private static byte[] Sign(string signing, string secret) =>
        HMACSHA256.HashData(Encoding.UTF8.GetBytes(secret), Encoding.UTF8.GetBytes(signing));

    /// <summary>base64url without padding, which is what the JWT format requires.</summary>
    private static string B64(byte[] data) => Convert.ToBase64String(data)
        .TrimEnd('=').Replace('+', '-').Replace('/', '_');

    private static bool TryUnb64(string value, out byte[] data)
    {
        string padded = value.Replace('-', '+').Replace('_', '/');
        padded += (padded.Length % 4) switch { 2 => "==", 3 => "=", _ => "" };
        try
        {
            data = Convert.FromBase64String(padded);
            return true;
        }
        catch (FormatException)
        {
            data = [];
            return false;
        }
    }

    /// <summary>
    /// Compares in constant time. See the type note on what that is and is not worth for a
    /// four-digit secret.
    /// </summary>
    public static bool VerifyPin(Config cfg, string candidate) =>
        CryptographicOperations.FixedTimeEquals(
            Encoding.UTF8.GetBytes(candidate), Encoding.UTF8.GetBytes(cfg.Pin));

    /// <summary>Mints a fresh key, shown to the user exactly once.</summary>
    public static string GenerateApiKey() =>
        KeyPrefix + Convert.ToBase64String(RandomNumberGenerator.GetBytes(32))
            .TrimEnd('=').Replace('+', '-').Replace('/', '_');

    public static string HashApiKey(string key) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key))).ToLowerInvariant();

    public static string Prefix(string key) =>
        key.Length < KeyPrefixDisplayLen ? key : key[..KeyPrefixDisplayLen];

    // --- the bootstrap key --------------------------------------------------
    //
    // Resolved once per process. Two cases:
    //
    //   DEFAULT_API_KEY set    -> use it. Stable across restarts, so integrations keep working.
    //                             Treated as a secret the operator already holds, so the UI shows
    //                             it masked.
    //   DEFAULT_API_KEY unset  -> generate a random one and log it. Nobody could know it
    //                             otherwise, so the UI DOES reveal it in full. That is the
    //                             deliberate trade, and it only happens when no explicit key was
    //                             configured.
    //
    // The alternative — a constant fallback in the source — would give every unconfigured
    // deployment the same publicly-known key. That is worse than either branch here.

    private static readonly object DefaultGate = new();
    private static string? _defaultKey;
    private static bool _defaultGenerated;

    /// <summary>
    /// Returns the bootstrap key and whether it was generated. Idempotent; safe to call from
    /// anywhere.
    /// </summary>
    public static (string Key, bool WasGenerated) ResolveDefaultKey(Config cfg)
    {
        lock (DefaultGate)
        {
            if (_defaultKey is null)
            {
                string configured = cfg.DefaultApiKey.Trim();
                if (configured.Length > 0)
                {
                    (_defaultKey, _defaultGenerated) = (configured, false);
                }
                else
                {
                    (_defaultKey, _defaultGenerated) = (GenerateApiKey(), true);
                }
            }
            return (_defaultKey, _defaultGenerated);
        }
    }

    /// <summary>Test seam: forgets the resolved bootstrap key.</summary>
    internal static void ResetDefaultKeyForTests()
    {
        lock (DefaultGate)
        {
            _defaultKey = null;
            _defaultGenerated = false;
        }
    }
}
