using Microsoft.AspNetCore.Http;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Model;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Store;

namespace RussianDocs.Service.Api;

/// <summary>
/// Who is calling.
///
/// <para>
/// Three shapes: the PIN operator (<c>session</c>, role admin, no account fields), a named account
/// (<c>session</c>, with <see cref="UserId"/>, <see cref="Username"/> and
/// <see cref="MustChangePassword"/> — all read FROM THE STORE, never from the token), and an API key
/// (<c>api_key</c>, role <c>service</c>, which no role requirement accepts).
/// </para>
/// </summary>
public sealed record Identity(string Kind, string Name, string Role, int KeyId = 0)
{
    public int? UserId { get; init; }
    public string? Username { get; init; }
    public bool? MustChangePassword { get; init; }

    public bool IsSession => Kind == "session";

    /// <summary>The single PIN operator identity. The PIN authenticates "whoever is at the console".</summary>
    public static readonly Identity Session = new("session", "Operator", "admin");

    /// <summary>The name that goes into the audit log as the actor.</summary>
    public string Actor => Username ?? "pin";
}

/// <summary>
/// **The single gate.** One class decides who a request is and whether it may proceed, because the
/// alternative — each endpoint checking for itself — grows a hole the day someone adds a route and
/// forgets a line, and that hole is silent.
///
/// <para>
/// **The session is re-checked against the store on EVERY request** (ports/AUTH.md §5). A JWT is valid
/// until it expires — eight hours here — so a disabled account, a changed password or a demoted role
/// would otherwise keep its old authority for the rest of that window. Each request loads the user
/// and compares <c>token_version</c>; anything that changes authority bumps it, and every issued token
/// dies at once. The cost is one dictionary lookup in an in-memory index.
/// </para>
///
/// <para>
/// **Fail-safe by shape.** An account that owes a password change gets a real token, but every guard
/// refuses it except <see cref="RequireSessionAllowPasswordChange"/>, which exactly two routes use. A
/// route added later with an ordinary guard refuses the restricted session rather than serving it.
/// Port of <c>service/api/deps.py</c>.
/// </para>
/// </summary>
public sealed class Authenticator(IDocumentStore db, AuthRuntime auth)
{
    /// <summary>Machine-readable: the UI routes on this string, so it is never reworded.</summary>
    public const string PasswordChangeRequired = "password_change_required";

    /// <summary>
    /// Extracts the token from an Authorization header.
    ///
    /// <para>
    /// Case-insensitive on the scheme, because clients disagree about "Bearer" versus "bearer" and
    /// rejecting one of them is a support ticket, not a security measure.
    /// </para>
    /// </summary>
    private static string BearerToken(HttpRequest request)
    {
        string header = request.Headers.Authorization.ToString();
        return header.Length >= 7 &&
               header.AsSpan(0, 7).Equals("bearer ", StringComparison.OrdinalIgnoreCase)
            ? header[7..].Trim()
            : "";
    }

    /// <summary>
    /// Decodes the bearer token into an identity, or <c>null</c> when it is not usable.
    ///
    /// <list type="bullet">
    /// <item><b>PIN mode accepts only PIN tokens.</b> A token minted while the service ran with named
    /// accounts carries a uid, and it must be REFUSED, not have its uid ignored: every PIN session is
    /// the administrator, so ignoring the uid would promote a viewer's still-valid token to full
    /// control the moment the service is switched back to PIN. The first Python version did exactly
    /// that while its comment claimed the opposite.</item>
    /// <item><b>Users mode accepts only account tokens</b>, and only while the account exists, is
    /// active and has the token's version.</item>
    /// </list>
    /// </summary>
    public Identity? SessionIdentity(HttpRequest request)
    {
        string token = BearerToken(request);
        if (token.Length == 0 || Tokens.DecodeAccessToken(auth.Tokens, token) is not { } claims)
        {
            return null;
        }

        if (!auth.UsersEnabled)
        {
            return claims.UidPresent ? null : Identity.Session;
        }

        if (claims.Uid is not { } uid)
        {
            return null; // a PIN-era token, worthless in users mode
        }
        User? user = db.GetUser(uid);
        if (user is null || !user.IsActive || claims.Tv != user.TokenVersion)
        {
            // Password changed, role changed or the account was disabled since this token was
            // issued. This is the whole reason token_version exists.
            return null;
        }
        return new Identity("session", user.DisplayName.Length > 0 ? user.DisplayName : user.Username,
            user.Role)
        {
            UserId = user.Id,
            Username = user.Username,
            MustChangePassword = user.MustChangePassword,
        };
    }

    /// <summary>
    /// Best effort, never rejects. The session is checked FIRST because it is cheap — an HMAC and a
    /// dictionary lookup — while the API key path hashes and then scans every stored key.
    /// </summary>
    public Identity? Optional(HttpRequest request)
    {
        if (SessionIdentity(request) is { } session)
        {
            return session;
        }
        string presented = request.Headers["X-API-Key"].ToString();
        if (presented.Length > 0 && ApiKeys.Verify(db, auth.Tokens, presented) is { } key)
        {
            ApiKeys.Touch(db, key);
            return new Identity("api_key", key.Label, "service", key.Id);
        }
        return null;
    }

    /// <summary>
    /// Any session, INCLUDING one that still owes a password change. Used by exactly two routes:
    /// who-am-I and change-my-password. Named after what it permits so nobody uses it by accident.
    /// </summary>
    public Identity RequireSessionAllowPasswordChange(HttpRequest request) =>
        SessionIdentity(request) ?? throw ServiceException.Unauthorized("Sign in to use this endpoint");

    private static void RejectIfPasswordChangePending(Identity identity)
    {
        if (identity.MustChangePassword == true)
        {
            throw ServiceException.Forbidden(PasswordChangeRequired);
        }
    }

    private static void RequireAtLeast(Identity identity, string minimum)
    {
        if (!Roles.AtLeast(identity.Role, minimum))
        {
            throw ServiceException.Forbidden($"This action requires the {minimum} role");
        }
    }

    /// <summary>
    /// A session, no pending password change, role ≥ <paramref name="minimum"/>. API keys are refused
    /// outright: users, keys, settings, logs and status are not an integration's business. In PIN mode
    /// the operator is admin, so every one of these is satisfied and PIN deployments behave as before.
    /// </summary>
    public Identity RequireRole(HttpRequest request, string minimum)
    {
        Identity identity = RequireSessionAllowPasswordChange(request);
        RejectIfPasswordChangePending(identity);
        RequireAtLeast(identity, minimum);
        return identity;
    }

    /// <summary>
    /// An API key at any level, or a session as <see cref="RequireRole"/>. For the document routes,
    /// which serve both the bundled UI and integrations.
    ///
    /// <para>
    /// **Why a role at all here.** The first Python version guarded these with "API key or any
    /// session", which checks that a caller is SOMEONE and never what they may do: a viewer could
    /// upload, reprocess and delete — and the route test was green, because it asked whether a guard
    /// was present rather than which. Every document route now names the role it needs, and the route
    /// test checks that name against the table in ports/AUTH.md §5.
    /// </para>
    /// </summary>
    public Identity RequireApiOrRole(HttpRequest request, string minimum)
    {
        Identity identity = Optional(request)
                            ?? throw ServiceException.Unauthorized(
                                "Provide an API key in X-API-Key, or sign in");
        if (identity.Kind == "api_key")
        {
            return identity;
        }
        RejectIfPasswordChangePending(identity);
        RequireAtLeast(identity, minimum);
        return identity;
    }
}

/// <summary>
/// A named authorisation requirement: what a route declares, and what actually runs.
///
/// <para>
/// **One object is both**, and that is the point. The route table attaches the guard to the endpoint
/// as metadata AND calls it before the handler, so the route test (which enumerates the endpoint data
/// source and reads the guard's <see cref="Name"/>) is reading the thing that executes — not a label
/// that could disagree with it. The names are the reference's dependency names, so the test compares
/// against ports/AUTH.md §5 verbatim.
/// </para>
/// </summary>
public sealed class Guard
{
    private readonly Func<Authenticator, HttpRequest, Identity?> _check;

    private Guard(string name, Func<Authenticator, HttpRequest, Identity?> check)
    {
        Name = name;
        _check = check;
    }

    public string Name { get; }

    /// <summary>Runs the check. Throws a 401/403 <see cref="ServiceException"/>, or returns the caller.</summary>
    public Identity? Admit(Authenticator auth, HttpRequest request) => _check(auth, request);

    /// <summary>No credential needed. Returns the best-effort identity, which handlers may ignore.</summary>
    public static readonly Guard Public = new("public", (_, _) => null);

    public static readonly Guard SessionAllowPasswordChange = new("require_session_allow_password_change",
        (a, r) => a.RequireSessionAllowPasswordChange(r));

    public static readonly Guard Viewer = Role(Roles.Viewer);
    public static readonly Guard Operator = Role(Roles.Operator);
    public static readonly Guard Admin = Role(Roles.Admin);
    public static readonly Guard ApiOrViewer = ApiOrRole(Roles.Viewer);
    public static readonly Guard ApiOrOperator = ApiOrRole(Roles.Operator);

    private static Guard Role(string minimum) =>
        new($"require_{minimum}", (a, r) => a.RequireRole(r, minimum));

    private static Guard ApiOrRole(string minimum) =>
        new($"require_api_or_{minimum}", (a, r) => a.RequireApiOrRole(r, minimum));

    public override string ToString() => Name;
}
