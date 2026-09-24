using System.Text.Json.Serialization;

namespace RussianDocs.Service.Model;

/// <summary>The three roles, in ascending order of authority.</summary>
public static class Roles
{
    public const string Viewer = "viewer";
    public const string Operator = "operator";
    public const string Admin = "admin";

    /// <summary>The order IS the policy: a role satisfies every requirement at or below its index.</summary>
    public static readonly IReadOnlyList<string> All = [Viewer, Operator, Admin];

    /// <summary>
    /// True when <paramref name="role"/> is <paramref name="required"/> or higher.
    ///
    /// <para>
    /// **An unknown role satisfies NOTHING** — not even viewer. A typo in a stored record must deny,
    /// never default to allow; and "service", the API-key pseudo-role, is unknown here on purpose.
    /// </para>
    /// </summary>
    public static bool AtLeast(string? role, string required)
    {
        int have = role is null ? -1 : IndexOf(role);
        int need = IndexOf(required);
        return have >= 0 && need >= 0 && have >= need;
    }

    public static bool Known(string? role) => role is not null && IndexOf(role) >= 0;

    private static int IndexOf(string role)
    {
        for (int i = 0; i < All.Count; i++)
        {
            if (All[i] == role)
            {
                return i;
            }
        }
        return -1;
    }
}

/// <summary>
/// A named account for the website, used when <c>AUTH_MODE=users</c>.
///
/// <para>
/// Only the hash is stored, like <see cref="ApiKey"/> — but the algorithm is different and the reason
/// is in <c>Auth/Passwords.cs</c>: an API key is random, a password is not.
/// </para>
///
/// <para>
/// **<see cref="TokenVersion"/> is the field worth reading twice.** It is embedded in every issued JWT
/// and compared on each request. Bumping it — on a password change, a role change, deactivation —
/// invalidates every token already handed out for the account, on every device. Without it a disabled
/// administrator keeps working access until the token expires, which here is eight hours. One integer
/// closes a real hole.
/// </para>
///
/// <para>
/// **<see cref="MustChangePassword"/>** exists because the first account ships with a known,
/// deliberately weak password. A default credential usable indefinitely is the single most common way
/// a demo turns into an incident.
/// </para>
///
/// <para>
/// The JSON names are the <c>users.json</c> format shared by all four services (ports/AUTH.md §9) —
/// exactly these ten, snake_case. Port of <c>service/core/models.py::User</c>.
/// </para>
/// </summary>
public sealed class User
{
    [JsonPropertyName("id")] public int Id { get; set; }
    [JsonPropertyName("username")] public string Username { get; set; } = "";
    [JsonPropertyName("role")] public string Role { get; set; } = Roles.Viewer;

    /// <summary>An Argon2id PHC string. **Never** leaves the service — see <see cref="Public"/>.</summary>
    [JsonPropertyName("password_hash")] public string PasswordHash { get; set; } = "";

    [JsonPropertyName("display_name")] public string DisplayName { get; set; } = "";
    [JsonPropertyName("is_active")] public bool IsActive { get; set; } = true;
    [JsonPropertyName("must_change_password")] public bool MustChangePassword { get; set; }

    /// <summary>Starts at 1, so a token can never be minted against a version that "does not exist".</summary>
    [JsonPropertyName("token_version")] public int TokenVersion { get; set; } = 1;

    [JsonPropertyName("created_at")]
    [JsonConverter(typeof(NullableUtcConverter))]
    public DateTime? CreatedAt { get; set; }

    [JsonPropertyName("last_login_at")]
    [JsonConverter(typeof(NullableUtcConverter))]
    public DateTime? LastLoginAt { get; set; }

    /// <summary>
    /// "Now", truncated to MICROSECONDS.
    ///
    /// <para>
    /// A <see cref="DateTime"/> has 100 ns ticks, so the shared converter would write seven fractional
    /// digits; Python's <c>isoformat</c> writes six, and <c>users.json</c> is read by the reference too.
    /// Truncating here keeps what this port writes inside what every other service writes, instead of
    /// relying on each reader tolerating a seventh digit.
    /// </para>
    /// </summary>
    public static DateTime UtcNowMicros()
    {
        DateTime now = DateTime.UtcNow;
        return new DateTime(now.Ticks - now.Ticks % 10, DateTimeKind.Utc);
    }

    /// <summary>
    /// An independent copy. The store hands these out, never its indexed instance: a caller mutating a
    /// user it read must not change what the next reader sees — the first Python version shared them,
    /// which made the last-admin rule a check-then-act race against whoever held the same object.
    /// </summary>
    public User Clone() => new()
    {
        Id = Id, Username = Username, Role = Role, PasswordHash = PasswordHash,
        DisplayName = DisplayName, IsActive = IsActive, MustChangePassword = MustChangePassword,
        TokenVersion = TokenVersion, CreatedAt = CreatedAt, LastLoginAt = LastLoginAt,
    };

    /// <summary>
    /// The ONLY shape of a user that leaves the service: never the hash, never the token version.
    /// <c>display_name</c> falls back to the username so the UI never renders an empty name.
    /// </summary>
    public Dictionary<string, object?> Public() => new(StringComparer.Ordinal)
    {
        ["id"] = Id,
        ["username"] = Username,
        ["display_name"] = DisplayName.Length > 0 ? DisplayName : Username,
        ["role"] = Role,
        ["is_active"] = IsActive,
        ["must_change_password"] = MustChangePassword,
        // Formatted here: a [JsonConverter] on a property does not reach a value in a dictionary.
        ["created_at"] = NullableUtcConverter.Format(CreatedAt),
        ["last_login_at"] = NullableUtcConverter.Format(LastLoginAt),
    };
}

/// <summary>
/// One recorded action: who did what, to which object, when.
///
/// <para>
/// **No personal data goes in here, and that is a hard rule rather than a preference.** Documents are
/// erased at every restart in temporary mode; the audit log is meant to outlive them. A filename such
/// as <c>Ivanov_passport.jpg</c>, a recognised field, or a client address would quietly carry personal
/// data across the very erasure the ephemeral store promises. So the target is an ID, and the detail
/// is a role name or a username — never document content, never an address. The throttle uses the
/// address, in memory only.
/// </para>
///
/// <para>
/// Not tamper-proof, and the documentation says so: anyone with write access to the data directory
/// can edit the file. Port of <c>service/core/models.py::AuditEntry</c>; fields as in ports/AUTH.md §9.
/// </para>
/// </summary>
public sealed class AuditEntry
{
    [JsonPropertyName("id")] public int Id { get; set; }

    /// <summary><c>login</c>, <c>login_failed</c>, <c>password.change</c>, <c>user.create</c>, …</summary>
    [JsonPropertyName("action")] public string Action { get; set; } = "";

    /// <summary>A username, or <c>pin</c> for the shared PIN session.</summary>
    [JsonPropertyName("actor")] public string Actor { get; set; } = "";

    [JsonPropertyName("target_type")] public string TargetType { get; set; } = "";

    /// <summary>A STRING even for a numeric id, so every target type shares one column.</summary>
    [JsonPropertyName("target_id")] public string TargetId { get; set; } = "";

    [JsonPropertyName("detail")] public string Detail { get; set; } = "";

    [JsonPropertyName("at")]
    [JsonConverter(typeof(NullableUtcConverter))]
    public DateTime? At { get; set; }

    public AuditEntry Clone() => new()
    {
        Id = Id, Action = Action, Actor = Actor, TargetType = TargetType, TargetId = TargetId,
        Detail = Detail, At = At,
    };

    public Dictionary<string, object?> Public() => new(StringComparer.Ordinal)
    {
        ["id"] = Id,
        ["action"] = Action,
        ["actor"] = Actor,
        ["target_type"] = TargetType,
        ["target_id"] = TargetId,
        ["detail"] = Detail,
        ["at"] = NullableUtcConverter.Format(At),
    };
}
