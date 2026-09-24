using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Model;
using RussianDocs.Service.Repositories;
using Results = Microsoft.AspNetCore.Http.Results;

namespace RussianDocs.Service.Api;

/// <summary>
/// Sign-in (a shared PIN, or a named account), who-am-I, change-my-password, user management and the
/// audit log.
///
/// <para>
/// Which sign-in is live is decided by <c>AUTH_MODE</c>, resolved once in <see cref="AuthMode"/>; this
/// file only ever asks <see cref="AuthRuntime.UsersEnabled"/>. <c>GET /auth/config</c> is what makes
/// one frontend serve both modes: the login page asks what to render instead of guessing.
/// </para>
///
/// <para>Security notes on this file specifically:</para>
/// <list type="bullet">
/// <item>Failed sign-ins are throttled (<see cref="LoginThrottle"/>) and recorded in the audit log.
/// **The submitted PIN or password is never logged** — writing rejected credentials to disk is its own
/// small leak, and a rejected one is often a typo away from the real one.</item>
/// <item>The reply to a bad username, a bad password and a disabled account is identical, including
/// its timing (the repository verifies against a decoy for a missing account).</item>
/// <item>An account that owes a password change gets a RESTRICTED token: real, but refused by every
/// guard except the two routes below that allow it. The login response says so with
/// <c>must_change_password</c>, so the UI routes straight to the change form.</item>
/// </list>
///
/// <para>
/// Port of <c>service/api/auth.py</c> and <c>service/api/users.py</c>; bodies, codes and messages are
/// ports/AUTH.md §7–§8.
/// </para>
/// </summary>
public sealed partial class ApiServer
{
    // --- /auth ----------------------------------------------------------------

    /// <summary>
    /// What the login page needs before anyone has authenticated. Says nothing that is not already
    /// visible: the mode, the rules, and — only in the demo default — the seeded credentials.
    /// </summary>
    private IResult AuthConfigInfo()
    {
        var payload = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["mode"] = auth.Mode,
            ["pin_required"] = auth.Mode == AuthMode.Pin,
            // Named separately so the UI can hide user management without inferring it from the mode
            // string; the two are the same today and need not stay that way.
            ["users_enabled"] = auth.UsersEnabled,
            ["downgrade_reason"] = auth.DowngradeReason,
        };
        if (auth.UsersEnabled)
        {
            payload["password_rules"] = Passwords.RulesForUi();
            if (DemoCredentials() is { } demo)
            {
                payload["demo_credentials"] = demo;
            }
        }
        return Results.Json(payload);
    }

    /// <summary>
    /// The seeded credentials, **only while publishing them is harmless** — and both conditions are
    /// necessary.
    ///
    /// <para>
    /// <i>The configured password is the documented demo one.</i> The first Python version published
    /// <c>ADMIN_PASSWORD</c> unconditionally, so an operator who set a real secret had it printed on the
    /// login page for every anonymous visitor. A value from the environment is a secret by default.
    /// </para>
    /// <para>
    /// <i>The seeded account still owes its change.</i> After that "admin/1234" is false, and advertising
    /// a credential that does not work is noise at best and a hint about the account name at worst.
    /// </para>
    /// </summary>
    private Dictionary<string, string>? DemoCredentials()
    {
        if (auth.AdminPassword != Config.Settings.DefaultAdminPassword)
        {
            return null;
        }
        User? seeded = auth.Users.Find(auth.AdminUsername);
        if (seeded is null || !seeded.MustChangePassword)
        {
            return null;
        }
        return new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["username"] = seeded.Username,
            ["password"] = Config.Settings.DefaultAdminPassword,
        };
    }

    /// <summary>The throttle key's address: the socket peer, NEVER a header the client writes.</summary>
    private static string ClientAddress(HttpRequest request) =>
        request.HttpContext.Connection.RemoteIpAddress?.ToString() ?? "-";

    /// <summary>429 with <c>Retry-After</c>, when either counter is at its limit.</summary>
    private void RefuseIfThrottled(HttpRequest request, string identity)
    {
        int blocked = auth.Throttle.BlockedFor(identity, ClientAddress(request));
        if (blocked > 0)
        {
            request.HttpContext.Response.Headers.RetryAfter =
                blocked.ToString(CultureInfo.InvariantCulture);
            throw new ServiceException(ErrorKind.TooManyAttempts,
                $"Too many attempts. Try again in {blocked} s");
        }
    }

    private static IResult Detail(int status, string text) =>
        Results.Json(new ApiErrors.ErrorBody(text), statusCode: status);

    /// <summary>
    /// Exchanges the PIN for a session JWT. The token carries no <c>uid</c>: every PIN session is the
    /// single operator, and a uid in it would be refused by the gate.
    /// </summary>
    private IResult PinLogin(HttpRequest request)
    {
        if (auth.UsersEnabled)
        {
            // Not 401: the credential is not wrong, the endpoint is not in service.
            throw ServiceException.Conflict(
                "This service is configured for named accounts; sign in with a username and password");
        }
        JsonObject body = Body(request);
        string pin = RequiredText(body, "pin", 1, 32);
        string client = ClientAddress(request);
        RefuseIfThrottled(request, "pin");

        if (!Tokens.VerifyPin(AuthConfig, pin))
        {
            auth.Throttle.NoteFailure("pin", client);
            Audit.Record(db, "login_failed", actor: "pin");
            // Logged without the attempted value — see the type note.
            log.LogWarning("[API] rejected PIN sign-in attempt");
            return Detail(StatusCodes.Status401Unauthorized, "Wrong PIN");
        }

        auth.Throttle.Clear("pin", client);
        Audit.Record(db, "login", actor: "pin");
        string token = Tokens.CreateAccessToken(AuthConfig, new Tokens.Claims
        {
            Sub = "operator",
            Name = Identity.Session.Name,
            Role = Identity.Session.Role,
        });
        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["access_token"] = token,
            ["token_type"] = "bearer",
            ["user"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["name"] = Identity.Session.Name,
                ["role"] = Identity.Session.Role,
            },
        });
    }

    /// <summary>
    /// Named sign-in. One reply — <c>Wrong username or password</c> — for an unknown user, a wrong
    /// password and a disabled account, so the endpoint cannot be used to list accounts.
    /// </summary>
    private IResult Login(HttpRequest request)
    {
        if (!auth.UsersEnabled)
        {
            throw ServiceException.Conflict("This service is configured for PIN sign-in");
        }
        JsonObject body = Body(request);
        string username = RequiredText(body, "username", 1, 64);
        string password = RequiredText(body, "password", 1, 256);
        string client = ClientAddress(request);
        RefuseIfThrottled(request, username);

        User? user = auth.Users.Authenticate(username, password);
        if (user is null)
        {
            auth.Throttle.NoteFailure(username, client);
            // The username is recorded because it is what makes the log useful when someone walks a
            // list of accounts. The password never is, and neither is the client address: it is
            // personal data, and nothing personal goes into the audit log.
            Audit.Record(db, "login_failed", actor: username.Trim());
            log.LogWarning("[API] rejected sign-in for {Username}", PyRepr.Quote(username));
            return Detail(StatusCodes.Status401Unauthorized, "Wrong username or password");
        }

        auth.Throttle.Clear(username, client);
        Audit.Record(db, "login", actor: user.Username);
        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["access_token"] = TokenFor(user),
            ["token_type"] = "bearer",
            ["user"] = user.Public(),
            ["must_change_password"] = user.MustChangePassword,
        });
    }

    private string TokenFor(User user) => Tokens.CreateAccessToken(AuthConfig, new Tokens.Claims
    {
        Sub = user.Username,
        Uid = user.Id,
        // This is what kills stale sessions: see the gate in Identity.cs.
        Tv = user.TokenVersion,
        Role = user.Role,
        Name = user.DisplayName.Length > 0 ? user.DisplayName : user.Username,
    });

    /// <summary>
    /// Who the token belongs to. Uses the permissive guard, so the change-password screen can show who
    /// is signed in while the session is still restricted. Absent keys are <c>null</c>.
    /// </summary>
    private IResult Me(Identity identity) => Results.Json(
        new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["mode"] = auth.Mode,
            ["user"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["username"] = identity.Username,
                ["name"] = identity.Name,
                ["role"] = identity.Role,
                ["user_id"] = identity.UserId,
                ["must_change_password"] = identity.MustChangePassword,
            },
        });

    /// <summary>
    /// Changes your own password, ending every session you have — including this one.
    ///
    /// <para>
    /// The current password is required even though the caller is authenticated: a token left open on
    /// an unattended machine should not be enough to take an account over permanently. **No fresh token
    /// is returned**: the version bump has just invalidated this one, and handing back a new one would
    /// quietly defeat the point of asking the user to sign in with the password they just chose.
    /// </para>
    /// </summary>
    private IResult ChangePassword(HttpRequest request, Identity identity)
    {
        if (!auth.UsersEnabled)
        {
            throw ServiceException.Conflict("There are no accounts in PIN mode");
        }
        JsonObject body = Body(request);
        string current = RequiredText(body, "current_password", 1, 256);
        string next = RequiredText(body, "new_password", 1, 256);

        User? user = identity.UserId is { } id ? auth.Users.Get(id) : null;
        if (user is null)
        {
            throw ServiceException.Unauthorized("Session no longer valid");
        }
        try
        {
            auth.Users.ChangePassword(user, next, current);
        }
        catch (UserError error)
        {
            throw ServiceException.BadRequest(error.Message);
        }

        Audit.Record(db, "password.change", actor: user.Username, targetType: "user",
            targetId: user.Id.ToString(CultureInfo.InvariantCulture));
        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["status"] = "ok",
            ["reauthenticate"] = true,
        });
    }

    // --- /users ---------------------------------------------------------------
    //
    // **In PIN mode every route here answers 404, not an empty list.** "There are no users" and "users
    // are not a concept in this configuration" are different answers, and a UI that cannot tell them
    // apart shows a management page nobody can make work. The frontend asks /auth/config and hides the
    // section; the 404 is the backstop for anyone calling the API directly. Checked AFTER the guard, so
    // a viewer still gets its 403 rather than learning the mode.
    //
    // Every mutation is audited, with the acting administrator as the actor and the account as the
    // target — an id and a username, never a password.

    private void RequireUsersMode()
    {
        if (!auth.UsersEnabled)
        {
            throw new UsersDisabled();
        }
    }

    /// <summary>Carries the users-mode 404 past <see cref="ApiErrors"/>, which spells every 404 "Not found".</summary>
    private sealed class UsersDisabled() : Exception("User accounts are disabled (AUTH_MODE=pin)");

    private IResult UsersRoute(Func<IResult> body)
    {
        try
        {
            RequireUsersMode();
            return body();
        }
        catch (UsersDisabled disabled)
        {
            return Detail(StatusCodes.Status404NotFound, disabled.Message);
        }
        catch (NoSuchUser)
        {
            return Detail(StatusCodes.Status404NotFound, "No such user");
        }
        catch (UserError error)
        {
            // Written to be shown: "Cannot demote the last active administrator" is the whole
            // explanation, and a generic 400 would leave the operator guessing which rule they hit.
            return Detail(StatusCodes.Status400BadRequest, error.Message);
        }
    }

    private sealed class NoSuchUser() : Exception("No such user");

    /// <summary>
    /// The path id. A non-integer is a 422 in the reference's pydantic shape (FastAPI declares
    /// <c>user_id: int</c>); an integer that names nobody is the 404 below.
    /// </summary>
    private User Load(string rawId)
    {
        if (!int.TryParse(rawId, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int id))
        {
            throw new ParamException(new ParamErrorItem
            {
                Type = "int_parsing",
                Loc = ["path", "user_id"],
                Msg = "Input should be a valid integer, unable to parse string as an integer",
                Input = rawId,
            });
        }
        return auth.Users.Get(id) ?? throw new NoSuchUser();
    }

    private static string Actor(Identity admin) => admin.Username ?? "?";

    private static string Id(User user) => user.Id.ToString(CultureInfo.InvariantCulture);

    private IResult ListUsers() => UsersRoute(() => Results.Json(
        new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["items"] = auth.Users.All().Select(u => u.Public()).ToList(),
            ["roles"] = Roles.All,
            ["password_rules"] = Passwords.RulesForUi(),
        }));

    private IResult CreateUser(HttpRequest request, Identity admin) => UsersRoute(() =>
    {
        JsonObject body = Body(request);
        string username = RequiredText(body, "username", 1, 64);
        string password = RequiredText(body, "password", 1, 256);
        string role = OptionalText(body, "role", 0, int.MaxValue) ?? Roles.Viewer;
        string displayName = OptionalText(body, "display_name", 0, 128) ?? "";

        User user = auth.Users.Create(username, password, role, displayName, mustChangePassword: true);
        Audit.Record(db, "user.create", actor: Actor(admin), targetType: "user", targetId: Id(user),
            detail: $"{user.Username} as {user.Role}");
        return Results.Json(user.Public(), statusCode: StatusCodes.Status201Created);
    });

    private IResult UpdateUser(HttpRequest request, Identity admin, string rawId) => UsersRoute(() =>
    {
        User user = Load(rawId);
        JsonObject body = Body(request);
        string? role = OptionalText(body, "role", 0, int.MaxValue);
        string? displayName = OptionalText(body, "display_name", 0, 128);
        bool? isActive = OptionalBool(body, "is_active");

        (string wasRole, bool wasActive) = (user.Role, user.IsActive);
        User updated = auth.Users.Update(user, role, displayName, isActive);

        var changes = new List<string>();
        if (wasRole != updated.Role)
        {
            changes.Add($"role {wasRole}->{updated.Role}");
        }
        if (wasActive != updated.IsActive)
        {
            changes.Add(updated.IsActive ? "activated" : "deactivated");
        }
        Audit.Record(db, "user.update", actor: Actor(admin), targetType: "user", targetId: Id(updated),
            detail: $"{updated.Username}: {(changes.Count > 0 ? string.Join(", ", changes) : "profile")}");
        return Results.Json(updated.Public());
    });

    /// <summary>Sets someone else's password. They must change it at their next sign-in.</summary>
    private IResult ResetPassword(HttpRequest request, Identity admin, string rawId) => UsersRoute(() =>
    {
        User user = Load(rawId);
        JsonObject body = Body(request);
        string password = RequiredText(body, "new_password", 1, 256);

        User updated = auth.Users.ResetPassword(user, password);
        Audit.Record(db, "user.password_reset", actor: Actor(admin), targetType: "user",
            targetId: Id(updated), detail: updated.Username);
        return Results.Json(updated.Public());
    });

    private IResult DeleteUser(Identity admin, string rawId) => UsersRoute(() =>
    {
        User user = Load(rawId);
        auth.Users.Delete(user, admin.UserId);
        Audit.Record(db, "user.delete", actor: Actor(admin), targetType: "user", targetId: Id(user),
            detail: user.Username);
        return Results.NoContent();
    });

    /// <summary>
    /// The action log. Administrator only — it names who did what. Available in BOTH modes: in PIN mode
    /// the actor is the literal <c>pin</c>, and knowing when something happened is still worth more than
    /// nothing. <c>limit</c> is clamped to 1…1000 rather than rejected, as in the reference.
    /// </summary>
    private IResult ListAudit(HttpRequest request)
    {
        int limit = Math.Clamp(QueryParams.Int(request.Query, "limit", 200, int.MinValue, 0), 1, 1000);
        IReadOnlyList<AuditEntry> entries = Audit.Recent(db, limit,
            QueryParams.Str(request.Query, "action"), QueryParams.Str(request.Query, "actor"));
        return Results.Json(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["items"] = entries.Select(e => e.Public()).ToList(),
            ["count"] = entries.Count,
        });
    }

    // --- request bodies -------------------------------------------------------
    //
    // Hand-checked rather than bound to a type, so the length limits of ports/AUTH.md §7 (pin 1–32,
    // username 1–64, password 1–256, display name ≤ 128) sit next to the field they limit, as the
    // reference's pydantic Field(...) does. A violation is a 400 with a sentence naming the field: the
    // reference answers 422 with pydantic's list, and AUTH.md accepts either — see DEVIATIONS N-11.

    private static JsonObject Body(HttpRequest request)
    {
        try
        {
            if (JsonNode.Parse(ReadBody(request)) is JsonObject body)
            {
                return body;
            }
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException)
        {
            // falls through to the same answer as a non-object body
        }
        throw ServiceException.BadRequest("expected a JSON object body");
    }

    /// <summary>Length in CODE POINTS, as pydantic counts it — a Cyrillic letter is one, an emoji is one.</summary>
    private static int Length(string text) => text.EnumerateRunes().Count();

    private static string RequiredText(JsonObject body, string name, int min, int max) =>
        OptionalText(body, name, min, max)
        ?? throw ServiceException.BadRequest($"'{name}' is required");

    private static string? OptionalText(JsonObject body, string name, int min, int max)
    {
        if (!body.TryGetPropertyValue(name, out JsonNode? node) || node is null)
        {
            return null;
        }
        if (node is not JsonValue value || !value.TryGetValue(out string? text))
        {
            throw ServiceException.BadRequest($"'{name}' must be a string");
        }
        int length = Length(text);
        if (length < min || length > max)
        {
            throw ServiceException.BadRequest(max == int.MaxValue
                ? $"'{name}' must be at least {min} characters"
                : $"'{name}' must be {min}–{max} characters");
        }
        return text;
    }

    private static bool? OptionalBool(JsonObject body, string name)
    {
        if (!body.TryGetPropertyValue(name, out JsonNode? node) || node is null)
        {
            return null;
        }
        return node is JsonValue value && value.TryGetValue(out bool flag)
            ? flag
            : throw ServiceException.BadRequest($"'{name}' must be true or false");
    }
}
