using System.Diagnostics;
using System.Globalization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Logging;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Ml;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Store;
using Results = Microsoft.AspNetCore.Http.Results;

namespace RussianDocs.Service.Api;

/// <summary>
/// The HTTP surface.
///
/// <para>
/// Explicit constructor parameters rather than a service locator or DI-by-attribute: a handler's
/// dependencies are then visible in one place, and the Go and Kotlin ports get the same shape without
/// a framework. The class is <c>partial</c> so the file split matches the Go port's
/// (<c>router</c> / <c>documents</c> / <c>misc</c>), which is what lets the two be read side by side.
/// </para>
/// </summary>
public sealed partial class ApiServer(
    IDocumentStore db,
    PipelineRuntime runtime,
    Worker.RecognitionWorker worker,
    Config.Settings cfg,
    SettingsRepository settings,
    AuthRuntime auth,
    string? webRoot,
    ILogger log)
{
    /// <summary>
    /// The API root. Versioned, because a published REST contract that cannot change shape is a
    /// published REST contract that gets replaced by a second service.
    /// </summary>
    public const string Prefix = "/api/v1";

    private readonly long _startedTicks = Stopwatch.GetTimestamp();

    private Tokens.Config AuthConfig => auth.Tokens;

    private readonly Authenticator _auth = new(db, auth);

    /// <summary>
    /// Builds the routing table — which is ports/AUTH.md §5's route table, line for line.
    ///
    /// <para>
    /// Minimal APIs, no controllers and no MVC: the routes below are the whole surface, and reading
    /// them as a PERMISSION LIST is the point. Every route goes through <see cref="Map"/>, which attaches
    /// its <see cref="Guard"/> as endpoint metadata AND runs it before the handler — so the route test,
    /// which enumerates the endpoints and reads that metadata, checks the guard that actually executes.
    /// A route mapped any other way has no guard metadata, and that test fails on it.
    /// </para>
    /// </summary>
    public void MapRoutes(WebApplication app)
    {
        // --- public ----------------------------------------------------------
        Map(app, "GET", "/health", Guard.Public, (_, _) => Health());
        Map(app, "GET", $"{Prefix}/auth/config", Guard.Public, (_, _) => AuthConfigInfo());
        Map(app, "POST", $"{Prefix}/auth/pin-login", Guard.Public, (r, _) => PinLogin(r));
        Map(app, "POST", $"{Prefix}/auth/login", Guard.Public, (r, _) => Login(r));

        // --- the two routes a restricted session may reach ----------------------
        Map(app, "GET", $"{Prefix}/auth/me", Guard.SessionAllowPasswordChange, (_, who) => Me(who!));
        Map(app, "POST", $"{Prefix}/auth/change-password", Guard.SessionAllowPasswordChange,
            (r, who) => ChangePassword(r, who!));

        // --- documents: API key OR a session with the role ----------------------
        // The same routes serve the bundled SPA and third-party integrations, which is why they accept
        // either credential rather than being duplicated per audience. Reads need viewer, writes need
        // operator; an API key passes both, because its scope is the document API and nothing else.
        Map(app, "GET", $"{Prefix}/documents", Guard.ApiOrViewer, (r, _) => List(r));
        Map(app, "GET", $"{Prefix}/documents/{{id}}", Guard.ApiOrViewer,
            (r, _) => GetDocument(ParseId(Route(r, "id"))));
        Map(app, "GET", $"{Prefix}/documents/{{id}}/progress", Guard.ApiOrViewer,
            (r, _) => DocumentProgress(ParseId(Route(r, "id"))));
        Map(app, "GET", $"{Prefix}/documents/{{id}}/image/{{kind}}", Guard.ApiOrViewer,
            (r, _) => ImageArtifact(ParseId(Route(r, "id")), Route(r, "kind")));
        Map(app, "POST", $"{Prefix}/documents", Guard.ApiOrOperator, (r, _) => Upload(r));
        Map(app, "POST", $"{Prefix}/documents/{{id}}/reprocess", Guard.ApiOrOperator,
            (r, _) => Reprocess(ParseId(Route(r, "id"))));
        Map(app, "DELETE", $"{Prefix}/documents/{{id}}", Guard.ApiOrOperator,
            (r, _) => DeleteDocument(ParseId(Route(r, "id"))));

        // Purge is administration, not document work: one call removes every document. Admin, and a
        // session — an integration has no business emptying the store.
        Map(app, "POST", $"{Prefix}/documents/purge", Guard.Admin, (_, _) => Purge());

        // --- operator surface: sessions only ------------------------------------
        // An API key is refused here outright: keys, settings, logs and the machine's status are not
        // an integration's concern.
        Map(app, "GET", $"{Prefix}/status", Guard.Viewer, (_, _) => Status());
        Map(app, "GET", $"{Prefix}/api-keys", Guard.Admin, (_, _) => ListKeys());
        Map(app, "POST", $"{Prefix}/api-keys", Guard.Admin, (r, _) => CreateKey(r));
        Map(app, "DELETE", $"{Prefix}/api-keys/{{id}}", Guard.Admin,
            (r, _) => DeleteKey(ParseId(Route(r, "id"))));
        Map(app, "GET", $"{Prefix}/settings", Guard.Admin, (_, _) => GetSettings());
        Map(app, "PUT", $"{Prefix}/settings", Guard.Admin, (r, _) => PutSettings(r));
        Map(app, "GET", $"{Prefix}/logs", Guard.Admin, (r, _) => Logs(r));

        // --- user management: administrators, and only in users mode -------------
        Map(app, "GET", $"{Prefix}/users", Guard.Admin, (_, _) => ListUsers());
        Map(app, "POST", $"{Prefix}/users", Guard.Admin, (r, who) => CreateUser(r, who!));
        Map(app, "PATCH", $"{Prefix}/users/{{id}}", Guard.Admin,
            (r, who) => UpdateUser(r, who!, Route(r, "id")));
        Map(app, "POST", $"{Prefix}/users/{{id}}/password", Guard.Admin,
            (r, who) => ResetPassword(r, who!, Route(r, "id")));
        Map(app, "DELETE", $"{Prefix}/users/{{id}}", Guard.Admin,
            (r, who) => DeleteUser(who!, Route(r, "id")));
        Map(app, "GET", $"{Prefix}/users/audit/entries", Guard.Admin, (r, _) => ListAudit(r));

        // --- the SPA, as a catch-all ------------------------------------------
        //
        // **The route pattern is explicit, and the parameterless MapFallback() is WRONG here.**
        // Its default pattern is `{*path:nonfile}`, and the `nonfile` constraint excludes any path
        // whose last segment contains a dot — which is every asset the SPA loads. The symptom is
        // as misleading as it gets: `/` returns index.html with a 200, so the server looks fine,
        // while `/assets/index-<hash>.js` 404s and the page renders BLANK with no server-side
        // error anywhere. Found by fetching the root and getting HTML that referenced files the
        // same server would not serve.
        app.MapFallback("/{*path}", Spa);
    }

    /// <summary>
    /// Maps one route with its guard.
    ///
    /// <para>
    /// **The guard runs BEFORE the handler touches the request body**, so a viewer's upload is a 403,
    /// not a 413 or a 400 about the file — the handler that would read it never starts. And every error
    /// leaves through <see cref="ApiErrors.Write"/>, the single place a status code is chosen.
    /// </para>
    ///
    /// <para>
    /// <c>WWW-Authenticate: Bearer</c> accompanies every 401 a guard produces, because that is what
    /// makes the status mean "you may retry with credentials" rather than "go away". A 403 does not
    /// carry it: the caller is known, and different credentials are not what is missing.
    /// </para>
    /// </summary>
    private void Map(WebApplication app, string method, string pattern, Guard guard,
        Func<HttpRequest, Identity?, IResult> handler)
    {
        app.MapMethods(pattern, [method], (HttpRequest request) =>
        {
            Identity? identity;
            try
            {
                identity = guard.Admit(_auth, request);
            }
            catch (Exception ex)
            {
                if (ex is ServiceException { Kind: ErrorKind.Unauthorized })
                {
                    request.HttpContext.Response.Headers.WWWAuthenticate = "Bearer";
                }
                return ApiErrors.Write(ex, log);
            }

            try
            {
                return handler(request, identity);
            }
            catch (Exception ex)
            {
                return ApiErrors.Write(ex, log);
            }
        }).WithMetadata(guard);
    }

    private static string Route(HttpRequest request, string name) =>
        request.RouteValues[name]?.ToString() ?? "";

    /// <summary>
    /// Parses the <c>{id}</c> path value.
    ///
    /// <para>
    /// A non-numeric id is a 404, because the route does not exist for that path — not a 400, which
    /// would suggest the request could be fixed.
    /// </para>
    /// </summary>
    private static int ParseId(string raw) =>
        int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) &&
        value >= 0
            ? value
            : throw ServiceException.NotFound("not a document id");

    private static double Round1(double v) => (int)(v * 10 + 0.5) / 10.0;

    /// <summary>
    /// Serves the built frontend, falling back to <c>index.html</c> for client-side routes.
    ///
    /// <para>
    /// Two things here are security-relevant rather than cosmetic:
    /// </para>
    /// <list type="bullet">
    /// <item>the resolved path is checked to be INSIDE the web root after link resolution, so a crafted
    /// path cannot escape it. Normalising the path alone is not enough on a tree that may contain
    /// links;</item>
    /// <item>anything under the API prefix that reached here is a 404 in JSON, not the SPA. Serving
    /// HTML for an unknown API route makes a client's JSON parse fail with a message about '&lt;',
    /// which is a genuinely confusing way to learn a route was misspelled.</item>
    /// </list>
    /// </summary>
    private IResult Spa(HttpContext context)
    {
        string path = context.Request.Path.Value ?? "/";
        if (path.StartsWith(Prefix, StringComparison.Ordinal))
        {
            return Results.Json(new ApiErrors.ErrorBody("Not found"),
                statusCode: StatusCodes.Status404NotFound);
        }
        if (webRoot is null)
        {
            return Results.Json(
                new ApiErrors.ErrorBody("No frontend build found; run `npm run build` in web/"),
                statusCode: StatusCodes.Status404NotFound);
        }

        string relative = path.TrimStart('/');
        if (relative.Length == 0)
        {
            relative = "index.html";
        }

        string root = Path.GetFullPath(webRoot);
        string candidate = Path.GetFullPath(Path.Combine(root, relative));
        // Outside the web root: treated as not found rather than forbidden, so a prober learns nothing
        // about the filesystem layout.
        if (candidate.StartsWith(root, StringComparison.Ordinal) && File.Exists(candidate))
        {
            return Results.File(candidate, ContentTypeFor(candidate));
        }

        // A client-side route: hand back index.html and let the SPA router resolve it.
        string index = Path.Combine(root, "index.html");
        if (File.Exists(index))
        {
            // no-cache on the shell only: the hashed asset files under /assets are immutable and get
            // the server's default caching, but a cached index.html pins the client to an old bundle
            // after a deploy.
            context.Response.Headers.CacheControl = "no-cache";
            return Results.File(index, "text/html; charset=utf-8");
        }
        return Results.Json(new ApiErrors.ErrorBody("Not found"),
            statusCode: StatusCodes.Status404NotFound);
    }

    /// <summary>
    /// The handful of content types the SPA build actually contains.
    ///
    /// <para>
    /// Explicit rather than a provider lookup: an unknown type served as <c>application/octet-stream</c>
    /// is a downloaded file instead of a rendered page, and the set of extensions Vite emits is small
    /// and known.
    /// </para>
    /// </summary>
    private static string ContentTypeFor(string path) =>
        Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".html" => "text/html; charset=utf-8",
            ".js" or ".mjs" => "text/javascript; charset=utf-8",
            ".css" => "text/css; charset=utf-8",
            ".json" => "application/json; charset=utf-8",
            ".svg" => "image/svg+xml",
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            ".ico" => "image/x-icon",
            ".woff2" => "font/woff2",
            ".woff" => "font/woff",
            ".ttf" => "font/ttf",
            ".map" => "application/json; charset=utf-8",
            _ => "application/octet-stream",
        };

    /// <summary>
    /// Locates a built frontend, or <c>null</c> if there is none.
    ///
    /// <para>
    /// Tries <c>web/dist</c> first and then <c>web/</c>, matching the reference: dist is the production
    /// build, while the bare directory is what a developer has before running the bundler. Returning
    /// <c>null</c> rather than failing is deliberate — the API is fully usable without a UI, and an
    /// integration does not care that npm was never run.
    /// </para>
    /// </summary>
    public static string? FindWebRoot(string? repoRoot)
    {
        if (repoRoot is null)
        {
            return null;
        }
        foreach (string relative in new[] { Path.Combine("web", "dist"), "web" })
        {
            string candidate = Path.Combine(repoRoot, relative);
            if (File.Exists(Path.Combine(candidate, "index.html")))
            {
                return candidate;
            }
        }
        return null;
    }
}
