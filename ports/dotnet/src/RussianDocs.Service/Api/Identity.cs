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
/// There are no user accounts: the PIN authenticates "whoever is at the console", nothing finer.
/// </para>
/// </summary>
public sealed record Identity(string Kind, string Name, string Role, int KeyId = 0)
{
    /// <summary>The single operator identity.</summary>
    public static readonly Identity Session = new("session", "Operator", "admin");
}

/// <summary>
/// Authentication comes in three levels, because two kinds of caller share one API.
///
/// <code>
/// RequireSession        Browser only, backed by the PIN-issued JWT. Guards anything that manages
///                       the SERVICE — API keys, settings, logs — because those are operator
///                       concerns and an integration has no business touching them.
/// RequireApiOrSession   Either a valid X-API-Key or a valid session JWT. Guards the WORKING
///                       endpoints, so the same routes serve the bundled UI and third-party
///                       integrations without duplicating them.
/// Optional              Never rejects. For endpoints that vary by caller but must stay reachable.
/// </code>
///
/// <para>
/// Why not one scheme for both: a four-digit PIN is a human affordance and a poor service credential
/// — shared, guessable, and it would have to be embedded in every integration. An API key is the
/// opposite. Conflating them forces one of the two into the wrong shape.
/// </para>
/// </summary>
public sealed class Authenticator(IDocumentStore db, Tokens.Config authConfig)
{
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
    /// Identifies a caller on a best-effort basis, returning <c>null</c> for anonymous.
    ///
    /// <para>
    /// The session is checked FIRST because it is cheap — an HMAC over the token — while the API key
    /// path hashes and then scans every stored key.
    /// </para>
    /// </summary>
    public Identity? Optional(HttpRequest request)
    {
        string token = BearerToken(request);
        if (token.Length > 0 && Tokens.DecodeAccessToken(authConfig, token) is not null)
        {
            return Identity.Session;
        }

        string presented = request.Headers["X-API-Key"].ToString();
        if (presented.Length > 0 && ApiKeys.Verify(db, authConfig, presented) is { } key)
        {
            ApiKeys.Touch(db, key);
            return new Identity("api_key", key.Label, "service", key.Id);
        }
        return null;
    }

    /// <summary>Admits browser sessions only.</summary>
    public Identity RequireSession(HttpRequest request)
    {
        string token = BearerToken(request);
        if (token.Length == 0 || Tokens.DecodeAccessToken(authConfig, token) is null)
        {
            throw ServiceException.Unauthorized("Sign in with the PIN to use this endpoint");
        }
        return Identity.Session;
    }

    /// <summary>Admits either kind of caller.</summary>
    public Identity RequireApiOrSession(HttpRequest request) =>
        Optional(request)
        ?? throw ServiceException.Unauthorized(
            "Provide an API key in X-API-Key, or sign in with the PIN");
}
