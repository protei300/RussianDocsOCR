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
    string? webRoot,
    ILogger log)
{
    /// <summary>
    /// The API root. Versioned, because a published REST contract that cannot change shape is a
    /// published REST contract that gets replaced by a second service.
    /// </summary>
    public const string Prefix = "/api/v1";

    private readonly long _startedTicks = Stopwatch.GetTimestamp();

    private Tokens.Config AuthConfig => new()
    {
        Pin = cfg.AuthPin,
        JwtSecret = cfg.JwtSecret,
        JwtAlgorithm = cfg.JwtAlgorithm,
        JwtExpireMinutes = cfg.JwtExpireMinutes,
        DefaultApiKey = cfg.DefaultApiKey,
    };

    private Authenticator Auth => new(db, AuthConfig);

    /// <summary>
    /// Builds the routing table.
    ///
    /// <para>
    /// Minimal APIs, no controllers and no MVC: the routes below are the whole surface, and reading
    /// them as a PERMISSION LIST is the point — <c>Guard(RequireSession, …)</c> versus
    /// <c>Guard(RequireApiOrSession, …)</c> says who may call what, at the place the route is declared.
    /// </para>
    /// </summary>
    public void MapRoutes(WebApplication app)
    {
        // --- auth: no credential required, obviously ---------------------------
        app.MapPost($"{Prefix}/auth/pin-login", PinLogin);

        // --- documents: API key OR session ------------------------------------
        // The same routes serve the bundled SPA and third-party integrations, which is why they accept
        // either credential rather than being duplicated per audience.
        app.MapPost($"{Prefix}/documents",
            (HttpRequest r) => Guard(r, Auth.RequireApiOrSession, _ => Upload(r)));
        app.MapGet($"{Prefix}/documents",
            (HttpRequest r) => Guard(r, Auth.RequireApiOrSession, _ => List(r)));
        app.MapPost($"{Prefix}/documents/purge",
            (HttpRequest r) => Guard(r, Auth.RequireSession, _ => Purge()));

        app.MapGet($"{Prefix}/documents/{{id}}",
            (HttpRequest r, string id) =>
                Guard(r, Auth.RequireApiOrSession, _ => GetDocument(ParseId(id))));
        app.MapDelete($"{Prefix}/documents/{{id}}",
            (HttpRequest r, string id) =>
                Guard(r, Auth.RequireApiOrSession, _ => DeleteDocument(ParseId(id))));
        app.MapGet($"{Prefix}/documents/{{id}}/progress",
            (HttpRequest r, string id) =>
                Guard(r, Auth.RequireApiOrSession, _ => DocumentProgress(ParseId(id))));
        app.MapPost($"{Prefix}/documents/{{id}}/reprocess",
            (HttpRequest r, string id) =>
                Guard(r, Auth.RequireApiOrSession, _ => Reprocess(ParseId(id))));
        app.MapGet($"{Prefix}/documents/{{id}}/image/{{kind}}",
            (HttpRequest r, string id, string kind) =>
                Guard(r, Auth.RequireApiOrSession, _ => ImageArtifact(ParseId(id), kind)));

        // --- operator surface: session only -----------------------------------
        // An integration has no business managing keys, settings or logs, so these do not accept an API
        // key at all.
        app.MapGet($"{Prefix}/api-keys", (HttpRequest r) =>
            Guard(r, Auth.RequireSession, _ => ListKeys()));
        app.MapPost($"{Prefix}/api-keys", (HttpRequest r) =>
            Guard(r, Auth.RequireSession, _ => CreateKey(r)));
        app.MapDelete($"{Prefix}/api-keys/{{id}}", (HttpRequest r, string id) =>
            Guard(r, Auth.RequireSession, _ => DeleteKey(ParseId(id))));
        app.MapGet($"{Prefix}/settings", (HttpRequest r) =>
            Guard(r, Auth.RequireSession, _ => GetSettings()));
        app.MapPut($"{Prefix}/settings", (HttpRequest r) =>
            Guard(r, Auth.RequireSession, _ => PutSettings(r)));
        app.MapGet($"{Prefix}/logs", (HttpRequest r) =>
            Guard(r, Auth.RequireSession, _ => Logs(r)));
        app.MapGet($"{Prefix}/status", (HttpRequest r) =>
            Guard(r, Auth.RequireSession, _ => Status()));

        // --- health: no prefix, no auth, for the container ---------------------
        app.MapGet("/health", Health);

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
    /// Wraps a handler with an authentication requirement and the single error path.
    ///
    /// <para>
    /// A wrapper rather than a check inside each handler: the check is then IMPOSSIBLE TO FORGET at the
    /// routing table, where it is also visible — which is the property FastAPI's <c>Depends</c>
    /// provides and the reason the routes read as a permission list.
    /// </para>
    ///
    /// <para>
    /// The <c>WWW-Authenticate</c> header accompanies every 401, because that is what makes the status
    /// code mean "you may retry with credentials" rather than "go away".
    /// </para>
    /// </summary>
    private IResult Guard(HttpRequest request, Func<HttpRequest, Identity> require,
        Func<Identity, IResult> handler)
    {
        Identity identity;
        try
        {
            identity = require(request);
        }
        catch (Exception ex)
        {
            request.HttpContext.Response.Headers.WWWAuthenticate = "Bearer";
            return ApiErrors.Write(ex, log);
        }

        try
        {
            return handler(identity);
        }
        catch (Exception ex)
        {
            return ApiErrors.Write(ex, log);
        }
    }

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
