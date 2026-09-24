using System.Text.RegularExpressions;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Model;
using RussianDocs.Service.Store;

namespace RussianDocs.Service.Repositories;

/// <summary>A rule was violated. The message is written to be shown to the caller as is.</summary>
public sealed class UserError(string message) : Exception(message);

/// <summary>
/// User accounts: the whole lifecycle, and the rules that keep it usable.
///
/// <para>
/// Only reachable when <c>AUTH_MODE=users</c>. In PIN mode nothing here is called and the store
/// stays empty — the <c>/users</c> routes answer 404 rather than an empty list.
/// </para>
///
/// <para>
/// **The rules worth stating before the code, because each is the kind that only bites in
/// production:**
/// </para>
/// <list type="bullet">
/// <item><b>You cannot lock everyone out.</b> Deleting, deactivating or demoting the last active
/// administrator is refused. It sounds like an edge case until someone demotes themselves to viewer to
/// test the role and discovers there is no account left that can undo it.</item>
/// <item><b>That check is only a check if it is atomic with the change.</b> Two administrators demoting
/// each other at the same moment each see the other still active, both checks pass, and the service
/// ends with none. So every mutation runs under ONE write lock and RE-READS the account inside it:
/// check and change are a single step. The lock is a <c>Monitor</c>, which is re-entrant, as the
/// reference's <c>RLock</c> is.</item>
/// <item><b>Hash before taking the lock.</b> Argon2 is the slow part (~100 ms) and depends on nothing
/// the lock protects; holding the lock across it would serialise all administration behind every
/// sign-in.</item>
/// <item><b>Anything that changes authority bumps the token version</b> — password, role, active flag.
/// Skipping it on a role change is the subtle case: a demoted administrator would otherwise keep
/// administrator rights for the rest of an eight-hour token.</item>
/// <item><b>Usernames are ASCII.</b> "admin" and "аdmin" (Cyrillic а) pass a case-insensitive uniqueness
/// check and are indistinguishable on screen. Display names can be anything — they are for reading,
/// not for trusting.</item>
/// </list>
///
/// <para>
/// An instance rather than a static class like <see cref="ApiKeys"/>, because it owns state that
/// must exist exactly once per store: the write lock, and the timing decoy. Program builds one.
/// Port of <c>service/repositories/users.py</c>; the contract is ports/AUTH.md §8.
/// </para>
/// </summary>
public sealed class Users(IDocumentStore db)
{
    /// <summary>Letters, digits, dot, underscore, hyphen; starting with a letter or digit; ≤ 64.</summary>
    private static readonly Regex UsernamePattern =
        new("^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", RegexOptions.CultureInvariant);

    /// <summary>
    /// One lock for every change to an account. Sign-ins do not hold it while hashing — see
    /// <see cref="Authenticate"/> — so a slow verification never blocks administration.
    /// </summary>
    private readonly object _write = new();

    /// <summary>
    /// A real Argon2 hash of a value nobody knows, used only to spend the same time verifying a password
    /// for a username that does not exist. Made ONCE — hashing on every failed login would itself be a
    /// timing signal — and forced at startup by <see cref="WarmDecoy"/> in users mode, so the first
    /// unknown-user login is not the slow one.
    /// </summary>
    private readonly Lazy<string> _decoy = new(() => Passwords.Hash("no-such-user-timing-decoy"),
        LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    /// Test seam: runs INSIDE the last-admin check, after the count. The concurrency test parks a thread
    /// here to force the interleaving the lock must prevent, instead of hoping two threads collide.
    /// </summary>
    internal Action? AfterAdminCount { get; set; }

    /// <summary>
    /// Test seam: runs in <see cref="Authenticate"/> between the (lock-free) verification and the
    /// write-back, which is exactly the window a concurrent demotion lands in.
    /// </summary>
    internal Action? AfterVerify { get; set; }

    public void WarmDecoy() => _ = _decoy.Value;

    public IReadOnlyList<User> All() => db.AllUsers();
    public User? Get(int id) => db.GetUser(id);
    public User? Find(string username) => db.FindUser(username);

    public int CountActiveAdmins(int? excluding = null) =>
        db.AllUsers().Count(u => u.Role == Roles.Admin && u.IsActive && u.Id != excluding);

    /// <summary>"Last active admin" = this account is an active admin and no OTHER account is.</summary>
    private void RequireNotLastAdmin(User user, string what)
    {
        if (user.Role != Roles.Admin || !user.IsActive)
        {
            return;
        }
        int others = CountActiveAdmins(excluding: user.Id);
        AfterAdminCount?.Invoke();
        if (others == 0)
        {
            throw new UserError($"Cannot {what} the last active administrator");
        }
    }

    private static string NormaliseUsername(string? username)
    {
        string name = (username ?? "").Trim();
        if (name.Length == 0)
        {
            throw new UserError("Username is required");
        }
        if (!UsernamePattern.IsMatch(name))
        {
            throw new UserError("Username may contain only Latin letters, digits, '.', '_' and '-', " +
                                "must start with a letter or digit, and be at most 64 characters");
        }
        return name;
    }

    private static void RequireKnownRole(string role)
    {
        if (!Roles.Known(role))
        {
            throw new UserError($"Unknown role {PyRepr.Quote(role)}");
        }
    }

    private static void RequireAcceptable(string password)
    {
        if (Passwords.Validate(password) is { } complaint)
        {
            throw new UserError(complaint);
        }
    }

    /// <summary>Re-read inside the lock. The copy the caller holds may already be stale.</summary>
    private User Fresh(int id) => db.GetUser(id) ?? throw new UserError("No such user");

    // -- lifecycle -----------------------------------------------------------

    public User Create(string username, string password, string role, string displayName = "",
        bool mustChangePassword = true)
    {
        string name = NormaliseUsername(username);
        RequireKnownRole(role);
        RequireAcceptable(password);
        // Hashed before the lock: the slow part, and independent of what the lock protects.
        string hash = Passwords.Hash(password);

        lock (_write)
        {
            // Uniqueness and id allocation are checked and used in one step, or two concurrent
            // creations of "petrov" would both succeed.
            if (db.FindUser(name) is not null)
            {
                throw new UserError($"User {PyRepr.Quote(name)} already exists");
            }
            return db.PutUser(new User
            {
                Id = db.NextUserId(),
                Username = name,
                Role = role,
                DisplayName = (displayName ?? "").Trim(),
                PasswordHash = hash,
                MustChangePassword = mustChangePassword,
                CreatedAt = User.UtcNowMicros(),
            });
        }
    }

    /// <summary>
    /// Creates the bootstrap administrator when there are NO users at all; <c>null</c> otherwise.
    ///
    /// <para>
    /// The one place that bypasses the password rules, and it has to: the default password is printed
    /// on the login page precisely so the demo can be used, and it fails every rule.
    /// <c>must_change_password</c> is what makes that defensible — the account can do nothing else
    /// until it is changed.
    /// </para>
    /// </summary>
    public User? SeedAdmin(string username, string password)
    {
        string name = NormaliseUsername(username);
        string hash = Passwords.Hash(password);
        lock (_write)
        {
            if (db.AllUsers().Count > 0)
            {
                return null;
            }
            return db.PutUser(new User
            {
                Id = db.NextUserId(),
                Username = name,
                Role = Roles.Admin,
                DisplayName = "Administrator",
                PasswordHash = hash,
                MustChangePassword = true,
                CreatedAt = User.UtcNowMicros(),
            });
        }
    }

    /// <summary>
    /// Verifies credentials. <c>null</c> for an unknown user, a wrong password and a disabled account
    /// alike.
    ///
    /// <para>
    /// One answer for all three on purpose: distinguishing "no such user" from "wrong password" lets
    /// anyone enumerate accounts. The password is still verified for a missing user — against the
    /// decoy — so the response TIME does not tell them apart either.
    /// </para>
    ///
    /// <para>
    /// **The verification runs OUTSIDE the lock, and the write-back RE-READS.** Writing back the copy
    /// loaded before hashing would silently undo anything an administrator changed during those
    /// ~100 ms — a demotion, a deactivation, a password reset. So the fresh copy is checked (gone,
    /// inactive, or hash changed meanwhile → no sign-in) and it, not the stale one, gets
    /// <c>last_login_at</c>.
    /// </para>
    /// </summary>
    public User? Authenticate(string username, string password)
    {
        User? user = db.FindUser(username);
        if (user is null)
        {
            Passwords.Verify(_decoy.Value, password);
            return null;
        }
        if (!Passwords.Verify(user.PasswordHash, password))
        {
            return null;
        }
        string? rehash = Passwords.NeedsRehash(user.PasswordHash) ? Passwords.Hash(password) : null;
        AfterVerify?.Invoke();

        lock (_write)
        {
            User? fresh = db.GetUser(user.Id);
            if (fresh is null || !fresh.IsActive || fresh.PasswordHash != user.PasswordHash)
            {
                return null;
            }
            if (rehash is not null)
            {
                fresh.PasswordHash = rehash;
            }
            fresh.LastLoginAt = User.UtcNowMicros();
            return db.PutUser(fresh);
        }
    }

    /// <summary>
    /// Changes one's own password. Bumps the token version, ending every session — including the one
    /// that made the request.
    /// </summary>
    public User ChangePassword(User user, string newPassword, string? currentPassword)
    {
        RequireAcceptable(newPassword);
        string hash = Passwords.Hash(newPassword);
        lock (_write)
        {
            User fresh = Fresh(user.Id);
            if (currentPassword is not null && !Passwords.Verify(fresh.PasswordHash, currentPassword))
            {
                throw new UserError("Current password is incorrect");
            }
            if (Passwords.Verify(fresh.PasswordHash, newPassword))
            {
                throw new UserError("The new password must differ from the current one");
            }
            fresh.PasswordHash = hash;
            fresh.MustChangePassword = false;
            fresh.TokenVersion++;
            return db.PutUser(fresh);
        }
    }

    /// <summary>An administrator sets someone else's password; they must change it at next sign-in.</summary>
    public User ResetPassword(User user, string newPassword)
    {
        RequireAcceptable(newPassword);
        string hash = Passwords.Hash(newPassword);
        lock (_write)
        {
            User fresh = Fresh(user.Id);
            fresh.PasswordHash = hash;
            fresh.MustChangePassword = true;
            fresh.TokenVersion++;
            return db.PutUser(fresh);
        }
    }

    /// <summary>
    /// Changes role, display name or active flag; <c>null</c> means unchanged. A role or active-flag
    /// change invalidates the account's tokens; a display-name change does not.
    /// </summary>
    public User Update(User user, string? role = null, string? displayName = null, bool? isActive = null)
    {
        if (role is not null)
        {
            RequireKnownRole(role);
        }
        lock (_write)
        {
            User fresh = Fresh(user.Id);
            bool authorityChanged = false;

            if (role is not null && role != fresh.Role)
            {
                if (role != Roles.Admin)
                {
                    RequireNotLastAdmin(fresh, "demote");
                }
                fresh.Role = role;
                authorityChanged = true;
            }
            if (isActive is { } active && active != fresh.IsActive)
            {
                if (!active)
                {
                    RequireNotLastAdmin(fresh, "deactivate");
                }
                fresh.IsActive = active;
                authorityChanged = true;
            }
            if (displayName is not null)
            {
                fresh.DisplayName = displayName.Trim();
            }
            if (authorityChanged)
            {
                fresh.TokenVersion++;
            }
            return db.PutUser(fresh);
        }
    }

    public void Delete(User user, int? actingUserId)
    {
        if (actingUserId is not null && user.Id == actingUserId)
        {
            // Less a safety rule than a usability one: there is no undo, and deleting the account you
            // are signed in as is never what was meant.
            throw new UserError("You cannot delete your own account");
        }
        lock (_write)
        {
            User fresh = Fresh(user.Id);
            RequireNotLastAdmin(fresh, "delete");
            db.DropUser(fresh.Id);
        }
    }
}

/// <summary>
/// The action log: who did what, to which object, when.
///
/// <para>
/// Thin on purpose — the rules live in <see cref="AuditEntry"/> (no personal data) and in
/// <see cref="IDocumentStore.AppendAudit"/> (append a line; never let a logging failure break the
/// action being logged). The actor is a STRING rather than a user so the PIN path uses it too: there
/// the actor is the literal <c>pin</c>, and one log covers both modes. Port of
/// <c>service/repositories/audit.py</c>.
/// </para>
/// </summary>
public static class Audit
{
    public static AuditEntry Record(IDocumentStore db, string action, string actor = "",
        string targetType = "", string targetId = "", string detail = "") =>
        db.AppendAudit(new AuditEntry
        {
            Action = action,
            Actor = actor.Length > 0 ? actor : "anonymous",
            TargetType = targetType,
            TargetId = targetId,
            Detail = detail,
            At = User.UtcNowMicros(),
        });

    public static IReadOnlyList<AuditEntry> Recent(IDocumentStore db, int limit, string? action,
        string? actor) => db.RecentAudit(limit, action, actor);
}
