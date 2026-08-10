using System.Globalization;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Settings;

namespace RussianDocs.Service.Api;

/// <summary>
/// The HTTP error contract.
///
/// <para>
/// **This file comes first, and the order is not arbitrary.** Every constraint it fixes is inherited
/// by every other handler, and each one is a real client dependency rather than a style choice:
/// </para>
///
/// <list type="bullet">
/// <item>the error body is <c>{"detail": "&lt;string&gt;"}</c> — what FastAPI produces and what the
/// SPA's fetch wrapper reads;</item>
/// <item>a missing credential is <b>401</b>, not 403. 403 means "authenticated but not allowed", and
/// the SPA redirects to the PIN screen on 401 only;</item>
/// <item>DELETE returns <b>204 with an empty body</b>. A JSON body on a 204 is a protocol error that
/// some clients reject outright;</item>
/// <item>POST /documents returns <b>202 with the full list row</b>, so the SPA can insert the row
/// without a second request;</item>
/// <item>/progress returns <b>200 with a JSON <c>null</c></b>, never 404 — a finished document is not
/// a missing one, and a 404 there makes the SPA drop the row;</item>
/// <item>images carry <c>Cache-Control: private, no-cache</c> and are fetched with an Authorization
/// header, never a token in the query string, because a query token lands in logs and browser
/// history;</item>
/// <item>the list filter parameter is named <c>status</c>.</item>
/// </list>
///
/// <para>Port of the FastAPI conventions in <c>service/api/*</c>.</para>
/// </summary>
public static class ApiErrors
{
    /// <summary>
    /// The one error shape. <c>detail</c> is a STRING, not an object: FastAPI's HTTPException
    /// produces exactly this, and the SPA reads <c>detail</c> directly.
    /// </summary>
    public sealed record ErrorBody([property: JsonPropertyName("detail")] string Detail);

    /// <summary>
    /// Maps an exception to a status and the <c>detail</c> body.
    ///
    /// <para>
    /// **The mapping lives HERE and nowhere else**, so a handler never picks a status code: that is
    /// what keeps 401-versus-403 and 409-versus-400 consistent across a dozen endpoints.
    /// </para>
    /// </summary>
    public static IResult Write(Exception error, ILogger log)
    {
        // A query-parameter rejection is the ONE case whose body is not `{"detail": "<string>"}`:
        // FastAPI generates it from pydantic and `detail` is a list. See QueryParams for the captured
        // reference responses and why the inconsistency is reproduced rather than smoothed over.
        if (error is ParamException param)
        {
            return Microsoft.AspNetCore.Http.Results.Json(
                new Dictionary<string, object> { ["detail"] = new[] { param.Item } },
                statusCode: StatusCodes.Status422UnprocessableEntity);
        }

        (int status, string detail) = Classify(error, log);
        return Microsoft.AspNetCore.Http.Results.Json(new ErrorBody(detail), statusCode: status);
    }

    private static (int Status, string Detail) Classify(Exception error, ILogger log)
    {
        switch (error)
        {
            case ServiceException { Kind: ErrorKind.NotFound }:
                return (StatusCodes.Status404NotFound, "Not found");

            case ServiceException { Kind: ErrorKind.Unauthorized }:
                // 401, NOT 403 — see the type note.
                return (StatusCodes.Status401Unauthorized, "Not authenticated");

            case ServiceException { Kind: ErrorKind.Conflict } conflict:
                return (StatusCodes.Status409Conflict, conflict.Message);

            case SettingValidationException validation:
                // **400, not 422.** FastAPI's own validation errors are 422, which is why 422 looks
                // right here — but the reference raises HTTPException(400) for a rejected SETTING,
                // and the reference is the contract. The message passes through because it names the
                // bound that was violated, which is the only useful thing a settings form can show.
                return (StatusCodes.Status400BadRequest, validation.Message);

            case ServiceException { Kind: ErrorKind.BadRequest } bad:
                return (StatusCodes.Status400BadRequest, bad.Message);

            case ServiceException { Kind: ErrorKind.ImageUnreadable } unreadable:
                return (StatusCodes.Status422UnprocessableEntity, unreadable.Message);

            case ServiceException { Kind: ErrorKind.RuntimeNotReady or ErrorKind.PipelineBusy } busy:
                return (StatusCodes.Status503ServiceUnavailable, busy.Message);

            default:
                // The message is NOT echoed for an unclassified error: it may carry a path or an
                // internal detail, and the log already has it in full.
                log.LogError(error, "[API] unhandled error");
                return (StatusCodes.Status500InternalServerError, "Internal server error");
        }
    }
}

/// <summary>One pydantic validation entry.</summary>
public sealed record ParamErrorItem
{
    [JsonPropertyName("type")] public required string Type { get; init; }
    [JsonPropertyName("loc")] public required object[] Loc { get; init; }
    [JsonPropertyName("msg")] public required string Msg { get; init; }

    /// <summary>
    /// The RAW query string, not the parsed value — pydantic echoes what it was given, which is why
    /// an out-of-range 500 comes back as the string "500".
    /// </summary>
    [JsonPropertyName("input")] public required string Input { get; init; }

    /// <summary>Omitted entirely for a parse failure, which is exactly the reference's shape.</summary>
    [JsonPropertyName("ctx")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, int>? Ctx { get; init; }
}

/// <summary>
/// A query-parameter rejection. Thrown so handlers keep the one error path they use everywhere else.
/// </summary>
public sealed class ParamException(ParamErrorItem item) : Exception(item.Msg)
{
    public ParamErrorItem Item { get; } = item;
}

/// <summary>
/// Query-parameter validation, matching the reference byte for byte.
///
/// <para>
/// **This is the one place where the reference does NOT use <c>{"detail": "&lt;string&gt;"}</c>.**
/// Every hand-written error in <c>service/api/*</c> raises HTTPException with a string, but a query
/// parameter declared as <c>Query(1, ge=1, le=100)</c> is validated by FastAPI itself, and FastAPI
/// answers with pydantic's own shape: <c>detail</c> is a LIST of objects. Reproducing that means
/// reproducing an inconsistency in the reference — done deliberately, because a client parses what the
/// server actually sends, and the reference is the contract.
/// </para>
///
/// <para>
/// Captured from the running reference rather than written from memory:
/// <code>
/// GET /documents?page_size=500
/// 422 {"detail":[{"type":"less_than_equal","loc":["query","page_size"],
///                 "msg":"Input should be less than or equal to 100",
///                 "input":"500","ctx":{"le":100}}]}
/// GET /documents?page_size=abc
/// 422 {"detail":[{"type":"int_parsing","loc":["query","page_size"],
///                 "msg":"Input should be a valid integer, unable to parse string as an integer",
///                 "input":"abc"}]}
/// </code>
/// Note <c>ctx</c> is absent for a parse failure and present for a bound, and that the ORDER matters:
/// parsing is checked before bounds.
/// </para>
///
/// <para>
/// This replaced silent clamping in the Go port. The old behaviour answered 200 with 100 rows for
/// <c>page_size=500</c>, so a client got a successful response to a request the reference rejects —
/// invisible from the server side and impossible to notice without diffing the two.
/// </para>
/// </summary>
public static class QueryParams
{
    /// <summary>
    /// Reads a bounded integer query parameter the way the reference declares it.
    ///
    /// <para>
    /// An ABSENT parameter yields the default; an EMPTY one (<c>?page_size=</c>) is a parse failure,
    /// which is what the reference does — verified, not assumed. Pass <paramref name="hi"/> &lt;= 0 for
    /// no upper bound, matching <c>page</c>, which declares ge=1 and no le.
    /// </para>
    /// </summary>
    public static int Int(IQueryCollection query, string name, int def, int lo, int hi)
    {
        if (!query.TryGetValue(name, out Microsoft.Extensions.Primitives.StringValues values))
        {
            return def;
        }
        string raw = values.ToString();
        if (!int.TryParse(raw.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture,
                out int value))
        {
            throw new ParamException(new ParamErrorItem
            {
                Type = "int_parsing",
                Loc = ["query", name],
                Msg = "Input should be a valid integer, unable to parse string as an integer",
                Input = raw,
            });
        }
        // Bounds AFTER parsing, and ge before le, so the reported error is the same one the reference
        // reports when a value violates both.
        if (value < lo)
        {
            throw new ParamException(new ParamErrorItem
            {
                Type = "greater_than_equal",
                Loc = ["query", name],
                Msg = $"Input should be greater than or equal to {lo}",
                Input = raw,
                Ctx = new Dictionary<string, int> { ["ge"] = lo },
            });
        }
        if (hi > 0 && value > hi)
        {
            throw new ParamException(new ParamErrorItem
            {
                Type = "less_than_equal",
                Loc = ["query", name],
                Msg = $"Input should be less than or equal to {hi}",
                Input = raw,
                Ctx = new Dictionary<string, int> { ["le"] = hi },
            });
        }
        return value;
    }

    public static string Str(IQueryCollection query, string name) =>
        query.TryGetValue(name, out Microsoft.Extensions.Primitives.StringValues values)
            ? values.ToString()
            : "";
}
