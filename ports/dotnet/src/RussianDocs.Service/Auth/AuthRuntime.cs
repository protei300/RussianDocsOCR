using RussianDocs.Service.Repositories;
using RussianDocs.Service.Store;

namespace RussianDocs.Service.Auth;

/// <summary>
/// Everything authentication needs at run time, resolved ONCE at startup and passed to the HTTP
/// surface as one value.
///
/// <para>
/// The reference re-resolves the mode on every call because it reads a cached settings object; the
/// environment tier here is immutable for the life of the process, so resolving it once is the same
/// answer computed fewer times. What matters is that <see cref="Mode"/> is the ONLY mode anything
/// consults — see <see cref="AuthMode"/>.
/// </para>
/// </summary>
public sealed class AuthRuntime
{
    public required string Mode { get; init; }
    public required string? DowngradeReason { get; init; }
    public required Tokens.Config Tokens { get; init; }
    public required Users Users { get; init; }
    public required LoginThrottle Throttle { get; init; }
    public required string AdminUsername { get; init; }
    public required string AdminPassword { get; init; }

    public bool UsersEnabled => Mode == AuthMode.Users;

    public static AuthRuntime Create(Config.Settings cfg, IDocumentStore db, Func<double>? clock = null)
    {
        (string mode, string? reason) = AuthMode.Resolve(cfg.AuthMode, db.Backend);
        return new AuthRuntime
        {
            Mode = mode,
            DowngradeReason = reason,
            Tokens = Auth.Tokens.Config.From(cfg),
            Users = new Users(db),
            Throttle = new LoginThrottle(cfg.LoginMaxAttempts, cfg.LoginLockoutSeconds, clock),
            AdminUsername = cfg.AdminUsername,
            AdminPassword = cfg.AdminPassword,
        };
    }
}
