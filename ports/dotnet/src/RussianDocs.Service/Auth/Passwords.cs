using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Konscious.Security.Cryptography;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace RussianDocs.Service.Auth;

/// <summary>
/// Password hashing — deliberately slow, unlike the API-key hashing next door.
///
/// <para>
/// **Why this is not SHA-256, which <see cref="Tokens.HashApiKey"/> uses.** An API key is 32 random
/// bytes: there is nothing to guess, so a fast hash is the right tool. A password is a short string a
/// human chose, and a fast hash over a low-entropy input is exactly what an attacker wants — a consumer
/// GPU tries billions of SHA-256 candidates a second against a stolen file. The defence is an algorithm
/// that is INTENTIONALLY expensive in time and memory, with a per-password salt so one cracking run
/// cannot amortise across accounts. Argon2id is memory-hard and OWASP's first recommendation.
/// </para>
///
/// <para>
/// **Why the stored value is a PHC string and not (salt, hash) columns.** The data directory is
/// implementation-neutral: the Python, Go and Kotlin services read the same <c>users.json</c> as this
/// one. <c>$argon2id$v=19$m=65536,t=3,p=4$&lt;salt&gt;$&lt;digest&gt;</c> carries its own algorithm
/// and parameters, so any implementation verifies it without agreeing on a private layout. Konscious
/// computes the digest but does not speak PHC, so the encoder and parser are here — and they are the
/// part that has to match argon2-cffi byte for byte, which is what the interop vectors in the tests
/// pin down.
/// </para>
///
/// <para>
/// Port of <c>service/core/passwords.py</c>; the contract is <c>ports/AUTH.md</c> §3.
/// </para>
/// </summary>
public static class Passwords
{
    // OWASP baseline. Changing these is safe: the parameters travel inside each hash, and
    // NeedsRehash reports which stored hashes predate the change.
    public const int MemoryKiB = 65536;   // 64 MiB
    public const int Iterations = 3;
    public const int Parallelism = 4;
    public const int DigestBytes = 32;
    public const int SaltBytes = 16;
    private const int Version = 19;       // 0x13, the only version Konscious implements

    /// <summary>
    /// The fixed prefix of every hash this service writes — asserted by the contract test on
    /// <c>users.json</c>, and by this port's own round-trip test.
    /// </summary>
    public const string CurrentPrefix = "$argon2id$v=19$m=65536,t=3,p=4$";

    /// <summary>
    /// **At most this many Argon2 computations at once, service-wide**, hashing and verifying alike.
    ///
    /// <para>
    /// Memory-hard hashing is a denial-of-service lever pointed at yourself: every computation costs
    /// 64 MiB, and the sign-in endpoint is reachable without a token. A flood of logins with a
    /// different username each time walks straight past the per-account lockout, and Kestrel will
    /// happily run dozens of them in parallel. Four slots cap the peak at 256 MiB; requests beyond
    /// that wait their turn instead of all allocating at once.
    /// </para>
    /// </summary>
    public const int HashConcurrency = 4;

    private static readonly SemaphoreSlim Slots = new(HashConcurrency, HashConcurrency);

    /// <summary>
    /// Where the "unreadable hash" warning goes. Static because verification is a static primitive
    /// called from deep inside the repository; Program points it at the service logger at startup.
    /// </summary>
    public static ILogger Log { get; set; } = NullLogger.Instance;

    // -- hashing -------------------------------------------------------------

    /// <summary>Hashes a password for storage. Returns a self-describing PHC string.</summary>
    public static string Hash(string plain)
    {
        byte[] salt = RandomNumberGenerator.GetBytes(SaltBytes);
        byte[] digest = Compute(plain, salt, MemoryKiB, Iterations, Parallelism, DigestBytes);
        return string.Create(CultureInfo.InvariantCulture,
            $"$argon2id$v={Version}$m={MemoryKiB},t={Iterations},p={Parallelism}" +
            $"${B64(salt)}${B64(digest)}");
    }

    /// <summary>
    /// Checks a password against a stored hash.
    ///
    /// <para>
    /// **Fails closed on EVERYTHING.** <c>false</c> for a wrong password and for a malformed, truncated,
    /// wrong-variant, empty or absurd stored hash, and no exception ever leaves this method. A catch-all
    /// is usually a smell; in an authentication primitive it is the correct default, because both
    /// alternatives are worse: an uncaught exception turns a corrupt record into a 500 that tells an
    /// attacker the account exists, and a narrow list lets the next unforeseen type through as a crash.
    /// The Python version's narrow list was wrong twice in five minutes. The exception TYPE is logged so a
    /// real bug stays visible instead of hiding behind a silent <c>false</c>.
    /// </para>
    ///
    /// <para>
    /// **The parameters are READ FROM THE STRING, and bounded before anything is allocated.** A record
    /// saying <c>m=4194304</c> would otherwise make one login allocate 4 GiB; see
    /// <see cref="Parse"/>.
    /// </para>
    /// </summary>
    public static bool Verify(string? storedHash, string candidate)
    {
        try
        {
            Phc phc = Parse(storedHash);
            byte[] digest = Compute(candidate, phc.Salt, phc.MemoryKiB, phc.Iterations,
                phc.Parallelism, phc.Digest.Length);
            // Constant time: an early-exit compare leaks how many leading bytes matched.
            return CryptographicOperations.FixedTimeEquals(digest, phc.Digest);
        }
        catch (Exception ex)
        {
            Log.LogWarning(
                "[AUTH] unreadable password hash ({Type}) — treating as a failed login",
                ex.GetType().Name);
            return false;
        }
    }

    /// <summary>
    /// True when the hash was made with different parameters than the current ones.
    ///
    /// <para>
    /// Called after a SUCCESSFUL verification — the only moment the plaintext is available to re-hash
    /// with the current cost. An unparsable string answers <c>true</c>, as argon2-cffi's does: whatever
    /// it is, it is not a current hash.
    /// </para>
    /// </summary>
    public static bool NeedsRehash(string? storedHash)
    {
        try
        {
            Phc phc = Parse(storedHash);
            return phc.MemoryKiB != MemoryKiB || phc.Iterations != Iterations ||
                   phc.Parallelism != Parallelism || phc.Digest.Length != DigestBytes;
        }
        catch (Exception)
        {
            return true;
        }
    }

    /// <summary>
    /// The one place Argon2 runs, inside the service-wide semaphore.
    ///
    /// <para>
    /// Konscious takes the memory in KiB, like the PHC <c>m</c>, and the password as bytes — UTF-8,
    /// which is what argon2-cffi hashes, so the Cyrillic interop vector verifies.
    /// </para>
    /// </summary>
    private static byte[] Compute(string password, byte[] salt, int memoryKiB, int iterations,
        int parallelism, int digestBytes)
    {
        Slots.Wait();
        try
        {
            using var argon = new Argon2id(Encoding.UTF8.GetBytes(password))
            {
                Salt = salt,
                MemorySize = memoryKiB,
                Iterations = iterations,
                DegreeOfParallelism = parallelism,
            };
            return argon.GetBytes(digestBytes);
        }
        finally
        {
            Slots.Release();
        }
    }

    // -- the PHC string ------------------------------------------------------

    /// <summary>A parsed <c>$argon2id$…</c> string, already checked against the bounds.</summary>
    internal sealed record Phc(int MemoryKiB, int Iterations, int Parallelism, byte[] Salt,
        byte[] Digest);

    /// <summary>
    /// Parses and BOUNDS a stored hash, throwing <see cref="FormatException"/> for anything else.
    ///
    /// <para>
    /// **The bounds are the point.** A stored hash is data an attacker may have written (a restored
    /// backup, a hand-edited file), and without them it dictates how much memory a login allocates:
    /// <c>1 ≤ t ≤ 10</c>, <c>8·p ≤ m ≤ 1 048 576</c> (1 GiB), <c>1 ≤ p ≤ 16</c>, digest 16–64 bytes,
    /// salt ≥ 8 bytes. All checked before a single byte of Argon2 memory exists.
    /// </para>
    ///
    /// <para>
    /// Only <c>argon2id</c> and only <c>v=19</c>: <c>argon2i</c>/<c>argon2d</c> are different functions,
    /// and verifying one as the other would be a guaranteed mismatch dressed up as a success path.
    /// </para>
    /// </summary>
    internal static Phc Parse(string? stored)
    {
        if (string.IsNullOrEmpty(stored))
        {
            throw new FormatException("empty hash");
        }
        // "", "argon2id", "v=19", "m=…,t=…,p=…", salt, digest
        string[] parts = stored.Split('$');
        if (parts.Length != 6 || parts[0].Length != 0)
        {
            throw new FormatException("not a PHC string");
        }
        if (parts[1] != "argon2id")
        {
            throw new FormatException("not argon2id");
        }
        if (parts[2] != "v=" + Version.ToString(CultureInfo.InvariantCulture))
        {
            throw new FormatException("unsupported version");
        }

        int? m = null, t = null, p = null;
        foreach (string pair in parts[3].Split(','))
        {
            int eq = pair.IndexOf('=');
            if (eq <= 0)
            {
                throw new FormatException("malformed parameter");
            }
            int value = Decimal(pair[(eq + 1)..]);
            switch (pair[..eq])
            {
                case "m" when m is null: m = value; break;
                case "t" when t is null: t = value; break;
                case "p" when p is null: p = value; break;
                default: throw new FormatException("unknown or repeated parameter");
            }
        }
        if (m is null || t is null || p is null)
        {
            throw new FormatException("missing parameter");
        }

        byte[] salt = Unb64(parts[4]);
        byte[] digest = Unb64(parts[5]);

        if (t < 1 || t > 10)
        {
            throw new FormatException("t out of bounds");
        }
        if (p < 1 || p > 16)
        {
            throw new FormatException("p out of bounds");
        }
        if (m < 8 * p || m > 1_048_576)
        {
            throw new FormatException("m out of bounds");
        }
        if (digest.Length < 16 || digest.Length > 64)
        {
            throw new FormatException("digest length out of bounds");
        }
        if (salt.Length < 8)
        {
            throw new FormatException("salt too short");
        }
        return new Phc(m.Value, t.Value, p.Value, salt, digest);
    }

    /// <summary>
    /// A plain decimal, ASCII digits only — no sign, no whitespace, nothing <c>int.Parse</c> would
    /// otherwise forgive. Nine digits at most, so the value cannot overflow before the bounds check.
    /// </summary>
    private static int Decimal(string text)
    {
        if (text.Length is 0 or > 9 || text.Any(c => c is < '0' or > '9'))
        {
            throw new FormatException("not a decimal");
        }
        return int.Parse(text, NumberStyles.None, CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// **Standard** base64 WITHOUT padding (<c>+/</c>, not URL-safe) — the PHC format, and what
    /// argon2-cffi writes. <see cref="Convert"/> insists on padding, so it is stripped here…
    /// </summary>
    private static string B64(byte[] data) => Convert.ToBase64String(data).TrimEnd('=');

    /// <summary>
    /// …and re-added here. The alphabet is checked first because <see cref="Convert"/> silently
    /// skips whitespace, and a hash that verifies with a stray space in it is not the hash on disk.
    /// </summary>
    private static byte[] Unb64(string text)
    {
        if (text.Length == 0 || text.Length % 4 == 1 ||
            text.Any(c => !(char.IsAsciiLetterOrDigit(c) || c == '+' || c == '/')))
        {
            throw new FormatException("not unpadded standard base64");
        }
        return Convert.FromBase64String(text + new string('=', (4 - text.Length % 4) % 4));
    }

    // -- composition rules ---------------------------------------------------

    /// <summary>The shortest password a person may choose.</summary>
    public const int MinLength = 8;

    /// <summary>One composition rule, as the UI receives it plus how the server checks it.</summary>
    public sealed record Rule(string Code, string Label, string Pattern, Func<string, bool> Met);

    /// <summary>
    /// The composition rules, as DATA and in this order.
    ///
    /// <para>
    /// The patterns are served to the UI verbatim, so the password page can tick them off as you type
    /// without a second copy in TypeScript — two copies of a validation rule drift, and the drift shows
    /// up as a form that accepts a password the server then rejects. They are written to mean the same
    /// thing in Python's <c>re</c>, .NET's <c>Regex</c> and the browser's <c>RegExp</c>.
    /// </para>
    ///
    /// <para>
    /// **With one exception the server checks by the pattern itself, and the exception is length.**
    /// <c>.{8,}</c> counts UTF-16 units in .NET but code points in Python, so a password of four emoji
    /// would pass here and fail in the reference. <see cref="LongEnough"/> counts code points, in runs
    /// between line feeds — which is exactly what <c>.{8,}</c> means where <c>.</c> is a code point
    /// that is not <c>\n</c>. Cyrillic is in the letter classes on purpose: this is a Russian deployment,
    /// and rejecting "Пароль1" as having no letters would be a bug, not a policy.
    /// </para>
    /// </summary>
    public static readonly IReadOnlyList<Rule> Rules =
    [
        new("length", $"at least {MinLength} characters", $".{{{MinLength},}}", LongEnough),
        Pattern("digit", "at least one digit", "[0-9]"),
        Pattern("letter", "at least one letter", "[a-zA-Zа-яёА-ЯЁ]"),
        Pattern("upper", "at least one capital letter", "[A-ZА-ЯЁ]"),
    ];

    private static Rule Pattern(string code, string label, string pattern)
    {
        var regex = new Regex(pattern, RegexOptions.CultureInvariant);
        return new Rule(code, label, pattern, regex.IsMatch);
    }

    private static bool LongEnough(string text)
    {
        foreach (string line in text.Split('\n'))
        {
            // Code points, not chars: a surrogate pair is one character to a person and to Python.
            int points = 0;
            for (int i = 0; i < line.Length; i++)
            {
                if (!char.IsLowSurrogate(line[i]) || i == 0 || !char.IsHighSurrogate(line[i - 1]))
                {
                    points++;
                }
            }
            if (points >= MinLength)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Codes of the rules this password fails, in declaration order.</summary>
    public static List<string> UnmetRules(string? candidate)
    {
        string text = candidate ?? "";
        return Rules.Where(r => !r.Met(text)).Select(r => r.Code).ToList();
    }

    /// <summary>
    /// A human-readable complaint, or <c>null</c> when the password is acceptable.
    ///
    /// <para>
    /// The SEEDED administrator password bypasses this — it is <c>1234</c> by design, printed on the
    /// login page, and made safe only by <c>must_change_password</c>. Every password a person chooses
    /// goes through it.
    /// </para>
    /// </summary>
    public static string? Validate(string? candidate)
    {
        List<string> failed = UnmetRules(candidate);
        if (failed.Count == 0)
        {
            return null;
        }
        return "Password needs: " + string.Join(", ",
            failed.Select(code => Rules.First(r => r.Code == code).Label));
    }

    /// <summary>The rule list as the password page consumes it — one source of truth.</summary>
    public static List<Dictionary<string, string>> RulesForUi() => Rules
        .Select(r => new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["code"] = r.Code,
            ["label"] = r.Label,
            ["pattern"] = r.Pattern,
        })
        .ToList();
}
