using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace RussianDocs.Service.Auth;

/// <summary>
/// Two authentication paths, for two different callers.
///
/// <list type="bullet">
/// <item><b>The website</b> signs in — with the shared PIN, or with a named account when
/// <c>AUTH_MODE=users</c> — and gets a short-lived JWT.</item>
/// <item><b>Machine callers</b> send an API key in <c>X-API-Key</c>. Keys are managed from the UI at
/// runtime, plus one bootstrap key from the environment.</item>
/// </list>
///
/// <para>
/// Why the split: a PIN or a password is a human affordance and a terrible service credential —
/// short, human-chosen, and it would have to be embedded in every integration. An API key is the
/// opposite. Endpoints both kinds of caller use accept either.
/// </para>
///
/// <para>
/// Security notes, honestly:
/// </para>
/// <list type="bullet">
/// <item>Comparison is constant-time. For the PIN that is mostly symbolic against a four-digit
/// space; what actually limits guessing is the failed-login throttle (<see cref="LoginThrottle"/>),
/// and even that is not a defence against an attacker who can reach the port. The NETWORK BOUNDARY
/// is the real control.</item>
/// <item>Only key HASHES are stored. A leaked data directory must not yield working
/// credentials.</item>
/// <item>**The published default secret never signs anything** — see <see cref="SigningSecret"/>.</item>
/// <item>A named-account token carries a user id and a token version; the gate in
/// <c>Api/Identity.cs</c> re-checks both against the store on every request.</item>
/// </list>
///
/// <para>
/// Port of <c>service/core/auth.py</c>. **The JWT is hand-rolled rather than taken from a
/// dependency** — HS256 with two base64url segments and an HMAC is about forty lines, and it is the
/// same choice the Go port made, so the two files read alike. (Password hashing is NOT hand-rolled;
/// see <see cref="Passwords"/>.)
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

        public static Config From(RussianDocs.Service.Config.Settings cfg) => new()
        {
            Pin = cfg.AuthPin,
            JwtSecret = cfg.JwtSecret,
            JwtAlgorithm = cfg.JwtAlgorithm,
            JwtExpireMinutes = cfg.JwtExpireMinutes,
            DefaultApiKey = cfg.DefaultApiKey,
        };
    }

    /// <summary>
    /// The JWT payload, in both of its shapes.
    ///
    /// <code>
    ///          PIN token     account token
    /// sub      "operator"    username
    /// name     "Operator"    display name, or the username
    /// role     "admin"       the account's role
    /// uid      ABSENT        integer user id
    /// tv       ABSENT        integer token_version
    /// exp      seconds       seconds
    /// </code>
    ///
    /// <para>
    /// **<c>uid</c> and <c>tv</c> are nullable, and absent is not zero.** PIN mode must REFUSE a token
    /// that carries a uid (ports/AUTH.md §5), so "there was a uid and it was 0" and "there was no uid"
    /// have to be different values. <see cref="UidPresent"/> goes one step further and records the KEY,
    /// so a uid that is present but not an integer (<c>"uid": null</c>, <c>"uid": "1"</c>) still counts
    /// as a uid for that refusal — which is what the reference's <c>"uid" in claims</c> does.
    /// </para>
    /// </summary>
    public sealed class Claims
    {
        [JsonPropertyName("sub")] public string Sub { get; set; } = "";

        [JsonPropertyName("name")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? Name { get; set; }

        [JsonPropertyName("role")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? Role { get; set; }

        [JsonPropertyName("uid")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public int? Uid { get; set; }

        [JsonPropertyName("tv")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public int? Tv { get; set; }

        [JsonPropertyName("exp")] public long Exp { get; set; }

        /// <summary>Whether the token had a <c>uid</c> key at all, integer or not. Decode side only.</summary>
        [JsonIgnore] public bool UidPresent { get; set; }
    }

    // --- the signing secret -------------------------------------------------

    /// <summary>The value shipped as the default. Kept here so the check below names it once.</summary>
    public const string DefaultJwtSecret = "changeme-in-production";

    private static readonly Lazy<string> ProcessSecret = new(
        () => B64(RandomNumberGenerator.GetBytes(48)), LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    /// The secret actually used to sign and verify.
    ///
    /// <para>
    /// **A known secret is not a secret.** The default is published — it is in this repository — so
    /// with it anyone can mint a token. In users mode that is a full takeover: the administrator is
    /// uid 1 and token_version starts at 1, so a forged <c>{"uid": 1, "tv": 1}</c> is a guess, not an
    /// attack. So when <c>JWT_SECRET</c> (trimmed) is empty or still the default, a random 48-byte
    /// secret is made ONCE per process and used instead. The only cost is that sessions end at a
    /// restart — and on this service nothing survives one anyway: the store is wiped, so a session
    /// outliving it would point at an account that no longer exists.
    /// </para>
    /// </summary>
    public static string SigningSecret(Config cfg) =>
        SecretIsEphemeral(cfg) ? ProcessSecret.Value : cfg.JwtSecret.Trim();

    /// <summary>Whether <see cref="SigningSecret"/> falls back to the per-process secret.</summary>
    public static bool SecretIsEphemeral(Config cfg)
    {
        string configured = cfg.JwtSecret.Trim();
        return configured.Length == 0 || configured == DefaultJwtSecret;
    }

    // --- tokens -------------------------------------------------------------

    /// <summary>Signs a JWT valid for the configured window. <c>exp</c> is set here.</summary>
    public static string CreateAccessToken(Config cfg, Claims claims)
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
        claims.Exp = DateTimeOffset.UtcNow.AddMinutes(cfg.JwtExpireMinutes).ToUnixTimeSeconds();
        byte[] payload = JsonSerializer.SerializeToUtf8Bytes(claims);
        string signing = B64(header) + "." + B64(payload);
        return signing + "." + B64(Sign(signing, SigningSecret(cfg)));
    }

    /// <summary>
    /// Returns the claims, or <c>null</c> for anything invalid or expired.
    ///
    /// <para>Three checks, in this order, and the order is the security property:</para>
    /// <list type="number">
    /// <item>**The algorithm is PINNED.** A header that does not say exactly <c>HS256</c> is refused
    /// before anything else is read — so <c>alg: none</c>, or HS512 keyed with our own secret, is
    /// refused rather than negotiated. This code only ever computes HS256, so a mismatched header
    /// could not verify anyway; checking it explicitly is what makes that a rule rather than an
    /// accident of the implementation.</item>
    /// <item>**The signature is verified BEFORE the claims are parsed**, in constant time. Parsing first
    /// would mean acting on attacker-controlled JSON; a plain equality test on the MAC leaks how much
    /// of it matched.</item>
    /// <item>Only then the claims, and the expiry.</item>
    /// </list>
    /// </summary>
    public static Claims? DecodeAccessToken(Config cfg, string token)
    {
        string[] parts = token.Split('.');
        if (parts.Length != 3)
        {
            return null;
        }
        if (!TryUnb64(parts[0], out byte[] rawHeader) || !HeaderIsHs256(rawHeader))
        {
            return null;
        }

        string signing = parts[0] + "." + parts[1];
        byte[] want = Sign(signing, SigningSecret(cfg));
        if (!TryUnb64(parts[2], out byte[] got) ||
            !CryptographicOperations.FixedTimeEquals(want, got))
        {
            return null;
        }

        if (!TryUnb64(parts[1], out byte[] raw) || ParseClaims(raw) is not { } claims)
        {
            return null;
        }
        if (claims.Exp != 0 && DateTimeOffset.UtcNow.ToUnixTimeSeconds() >= claims.Exp)
        {
            return null;
        }
        return claims;
    }

    private static bool HeaderIsHs256(byte[] raw)
    {
        try
        {
            return JsonNode.Parse(raw) is JsonObject header &&
                   header["alg"] is JsonValue alg &&
                   alg.TryGetValue(out string? name) && name == "HS256";
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>
    /// Reads the claims by hand rather than with the serializer, so the PRESENCE of <c>uid</c>
    /// survives — see <see cref="Claims"/>. A claim of the wrong type does not poison the others: a
    /// non-integer uid is recorded as present-but-null and the gate refuses it; a non-numeric
    /// <c>exp</c> refuses the whole token, as the reference's jose does.
    /// </summary>
    private static Claims? ParseClaims(byte[] raw)
    {
        JsonObject body;
        try
        {
            if (JsonNode.Parse(raw) is not JsonObject parsed)
            {
                return null;
            }
            body = parsed;
        }
        catch (JsonException)
        {
            return null;
        }

        var claims = new Claims
        {
            Sub = Text(body, "sub") ?? "",
            Name = Text(body, "name"),
            Role = Text(body, "role"),
            UidPresent = body.ContainsKey("uid"),
            Uid = Integer(body, "uid"),
            Tv = Integer(body, "tv"),
        };
        if (body.ContainsKey("exp"))
        {
            if (body["exp"] is not JsonValue exp || !exp.TryGetValue(out double seconds))
            {
                return null;
            }
            claims.Exp = (long)seconds;
        }
        return claims;
    }

    private static string? Text(JsonObject body, string key) =>
        body[key] is JsonValue v && v.TryGetValue(out string? s) ? s : null;

    /// <summary>
    /// An integer claim, or null. A JSON <c>1.0</c> or <c>"1"</c> is NOT an integer here, matching the
    /// reference's <c>isinstance(uid, int)</c> — the raw token text is inspected because the reader
    /// would otherwise happily convert <c>1.0</c> to 1.
    /// </summary>
    private static int? Integer(JsonObject body, string key)
    {
        if (body[key] is not JsonValue v)
        {
            return null;
        }
        JsonElement element = v.GetValue<JsonElement>();
        string text = element.GetRawText();
        return element.ValueKind == JsonValueKind.Number &&
               text.All(c => c is '-' or (>= '0' and <= '9')) &&
               element.TryGetInt32(out int n)
            ? n
            : null;
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
