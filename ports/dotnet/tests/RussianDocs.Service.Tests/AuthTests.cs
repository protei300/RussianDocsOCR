using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using NUnit.Framework;
using RussianDocs.Service.Api;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Errors;
using RussianDocs.Service.Ml;
using RussianDocs.Service.Model;
using RussianDocs.Service.Repositories;
using RussianDocs.Service.Store;
using RussianDocs.Service.Worker;

namespace RussianDocs.Service.Tests;

/// <summary>
/// What the black-box contract test (<c>conformance/auth_contract.py</c>) cannot see: the password
/// primitive against the interop vectors, the route table against the router, forced interleavings,
/// copy-on-read, and the throttle's arithmetic. Each test mirrors one in
/// <c>tests/service/test_auth_security.py</c>, per ports/AUTH.md §11.
///
/// <para>
/// **Every test in the concurrency and gate sections was proven by breaking the code it guards** —
/// the lock, the re-read, the copy, the PIN-mode uid refusal, the default-secret check — watching it
/// fail, and restoring. A test that passes both ways checks nothing; the first Python race test did
/// exactly that.
/// </para>
/// </summary>
[TestFixture]
public sealed class AuthTests
{
    private string _dir = null!;

    [SetUp]
    public void MakeDir()
    {
        _dir = Path.Combine(Path.GetTempPath(), "rdocs-auth-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    [TearDown]
    public void RemoveDir()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* temp dir */ }
    }

    private FileStore Store() => new(_dir, NullLogger.Instance);

    // -- passwords ---------------------------------------------------------------

    private const string Vector1 =
        "$argon2id$v=19$m=65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo";

    private const string Vector2 =
        "$argon2id$v=19$m=19456,t=2,p=1$rppcAWOP4qJuFb6Dc52G3g$NoDmjBcYZrj9DJvzNb421/YYxfGj1D+TxplD/tAEero";

    /// <summary>
    /// Made by argon2-cffi. The second has DIFFERENT parameters and a Cyrillic password, which proves
    /// the parameters are read from the string and the password is hashed as UTF-8.
    /// </summary>
    [Test]
    public void InteropVectorsVerifyAndRejectAWrongPassword()
    {
        Assert.That(Passwords.Verify(Vector1, "Vector-Pass1"), Is.True);
        Assert.That(Passwords.Verify(Vector2, "Пароль-42"), Is.True);
        Assert.That(Passwords.Verify(Vector1, "Vector-Pass2"), Is.False);
        Assert.That(Passwords.Verify(Vector2, "пароль-42"), Is.False);
    }

    [TestCase("")]
    [TestCase(null)]
    [TestCase("not a hash")]
    // A different Argon2 variant must not be verified as argon2id.
    [TestCase("$argon2i$v=19$m=65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    [TestCase("$argon2d$v=19$m=65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    // Truncated in the middle of the parameters, and in the digest.
    [TestCase("$argon2id$v=19$m=65536,t=3")]
    [TestCase("$argon2id$v=19$m=65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF")]
    // Not base64 / URL-safe alphabet / stray whitespace.
    [TestCase("$argon2id$v=19$m=65536,t=3,p=4$!!!!notbase64!!!$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    [TestCase("$argon2id$v=19$m=65536,t=3,p=4$PWP-BS8J-heQ62HqF9F7Yg$gKntrbpo0K_DP8I7BSoF-jkllxaGgjqr9555lul9PXo")]
    [TestCase("$argon2id$v=19$m=65536,t=3,p=4$PWP+BS8J +heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    // Wrong version, repeated and unknown parameters, a signed number.
    [TestCase("$argon2id$v=16$m=65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    [TestCase("$argon2id$v=19$m=65536,m=65536,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    [TestCase("$argon2id$v=19$m=65536,t=3,p=4,x=1$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    [TestCase("$argon2id$v=19$m=+65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    // Out of bounds: t, p, salt too short, digest too short.
    [TestCase("$argon2id$v=19$m=65536,t=11,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    [TestCase("$argon2id$v=19$m=65536,t=3,p=17$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    [TestCase("$argon2id$v=19$m=65536,t=3,p=4$PWP+BQ$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo")]
    [TestCase("$argon2id$v=19$m=65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7")]
    public void MalformedHashesFailClosedWithoutThrowing(string? stored)
    {
        Assert.That(() => Passwords.Verify(stored, "Vector-Pass1"), Throws.Nothing);
        Assert.That(Passwords.Verify(stored, "Vector-Pass1"), Is.False);
    }

    /// <summary>
    /// A hostile record must not dictate the allocation: <c>m=4194304</c> is 4 GiB. The bound is
    /// checked before Argon2 runs, so the call is fast and allocates almost nothing.
    /// </summary>
    [Test]
    public void AnAbsurdMemoryCostIsRefusedBeforeAllocating()
    {
        const string absurd = "$argon2id$v=19$m=4194304,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg" +
                              "$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo";
        Passwords.Verify(absurd, "warm the JIT");
        long before = GC.GetTotalAllocatedBytes(precise: true);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        bool ok = Passwords.Verify(absurd, "Vector-Pass1");
        clock.Stop();
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;

        Assert.That(ok, Is.False);
        Assert.That(allocated, Is.LessThan(1_000_000), "bytes allocated for a refused hash");
        Assert.That(clock.ElapsedMilliseconds, Is.LessThan(200));
    }

    [Test]
    public void AFreshHashRoundTripsWithTheCurrentParameters()
    {
        string hash = Passwords.Hash("Str0ng-Pass");
        Assert.That(hash, Does.StartWith(Passwords.CurrentPrefix));
        Assert.That(hash, Does.StartWith("$argon2id$v=19$m=65536,t=3,p=4$"));
        string[] parts = hash.Split('$');
        Assert.That(parts[4] + parts[5], Does.Not.Contain("="), "PHC base64 carries no padding");
        Assert.That(Convert.FromBase64String(parts[4] + "=="), Has.Length.EqualTo(16), "16-byte salt");
        Assert.That(Convert.FromBase64String(parts[5] + "="), Has.Length.EqualTo(32), "32-byte digest");
        Assert.That(Passwords.Verify(hash, "Str0ng-Pass"), Is.True);
        Assert.That(Passwords.Verify(hash, "Str0ng-Pas"), Is.False);
        // Two hashes of one password differ: the salt is random.
        Assert.That(Passwords.Hash("Str0ng-Pass"), Is.Not.EqualTo(hash));

        Assert.That(Passwords.NeedsRehash(hash), Is.False);
        Assert.That(Passwords.NeedsRehash(Vector1), Is.False);
        Assert.That(Passwords.NeedsRehash(Vector2), Is.True, "m=19456,t=2,p=1 is not current");
        Assert.That(Passwords.NeedsRehash("garbage"), Is.True);
    }

    // -- rules -------------------------------------------------------------------

    [TestCase("Str0ng-Pass", new string[0])]
    [TestCase("weakpass", new[] { "digit", "upper" })]
    [TestCase("пароль12", new[] { "upper" })]                  // Cyrillic letters are letters
    [TestCase("Пароль1", new[] { "length" })]                  // 7 code points (14 UTF-8 bytes)
    [TestCase("Парольк1", new string[0])]                      // 8 code points
    [TestCase("ПАРОЛЬЁЁ1", new string[0])]                     // Ё is in the classes
    [TestCase("😀😀😀😀a1A", new[] { "length" })]              // 7 code points, 10 UTF-16 units
    [TestCase("Abc1\nxyz", new[] { "length" })]                // `.` does not cross a line feed
    [TestCase("", new[] { "length", "digit", "letter", "upper" })]
    [TestCase("12345678", new[] { "letter", "upper" })]
    public void CompositionRules(string password, string[] unmet)
    {
        Assert.That(Passwords.UnmetRules(password), Is.EqualTo(unmet));
    }

    [Test]
    public void RulesAreServedInOrderWithTheExactPatterns()
    {
        List<Dictionary<string, string>> rules = Passwords.RulesForUi();
        Assert.That(rules.Select(r => r["code"]), Is.EqualTo(new[] { "length", "digit", "letter", "upper" }));
        Assert.That(rules.Select(r => r["pattern"]),
            Is.EqualTo(new[] { ".{8,}", "[0-9]", "[a-zA-Zа-яёА-ЯЁ]", "[A-ZА-ЯЁ]" }));
        Assert.That(Passwords.Validate("weakpass"),
            Is.EqualTo("Password needs: at least one digit, at least one capital letter"));
        Assert.That(Passwords.Validate("Str0ng-Pass"), Is.Null);
    }

    // -- the route table -----------------------------------------------------------

    /// <summary>ports/AUTH.md §5, transcribed. Every route listed, nothing unlisted.</summary>
    private static readonly (string Method, string Path, string Guard)[] RouteTable =
    [
        ("GET", "/health", "public"),
        ("GET", "/api/v1/auth/config", "public"),
        ("POST", "/api/v1/auth/pin-login", "public"),
        ("POST", "/api/v1/auth/login", "public"),
        ("GET", "/api/v1/auth/me", "require_session_allow_password_change"),
        ("POST", "/api/v1/auth/change-password", "require_session_allow_password_change"),
        ("GET", "/api/v1/documents", "require_api_or_viewer"),
        ("GET", "/api/v1/documents/{id}", "require_api_or_viewer"),
        ("GET", "/api/v1/documents/{id}/progress", "require_api_or_viewer"),
        ("GET", "/api/v1/documents/{id}/image/{kind}", "require_api_or_viewer"),
        ("POST", "/api/v1/documents", "require_api_or_operator"),
        ("POST", "/api/v1/documents/{id}/reprocess", "require_api_or_operator"),
        ("DELETE", "/api/v1/documents/{id}", "require_api_or_operator"),
        ("POST", "/api/v1/documents/purge", "require_admin"),
        ("GET", "/api/v1/status", "require_viewer"),
        ("GET", "/api/v1/api-keys", "require_admin"),
        ("POST", "/api/v1/api-keys", "require_admin"),
        ("DELETE", "/api/v1/api-keys/{id}", "require_admin"),
        ("GET", "/api/v1/settings", "require_admin"),
        ("PUT", "/api/v1/settings", "require_admin"),
        ("GET", "/api/v1/logs", "require_admin"),
        ("GET", "/api/v1/users", "require_admin"),
        ("POST", "/api/v1/users", "require_admin"),
        ("PATCH", "/api/v1/users/{id}", "require_admin"),
        ("POST", "/api/v1/users/{id}/password", "require_admin"),
        ("DELETE", "/api/v1/users/{id}", "require_admin"),
        ("GET", "/api/v1/users/audit/entries", "require_admin"),
    ];

    /// <summary>
    /// The router, read back from the endpoint data source, against the table — including WHICH guard.
    ///
    /// <para>
    /// A test that only asked "is there a guard" passed on the first Python version while a viewer
    /// could delete everything. The guard name comes from the same <see cref="Guard"/> object that
    /// runs, so this cannot pass on a label that disagrees with the code.
    /// </para>
    /// </summary>
    [Test]
    public void TheRouterIsExactlyTheRouteTable()
    {
        var cfg = new Config.Settings { DataDir = _dir, DataWipeOnStart = false, SeedSamples = -1 };
        ILogger log = NullLogger.Instance;
        FileStore db = Store();
        var runtime = new PipelineRuntime(log);
        var settings = new SettingsRepository(cfg, log);
        var worker = new RecognitionWorker(db, runtime, cfg, settings, log);
        WebApplication app = WebApplication.CreateBuilder().Build();
        new ApiServer(db, runtime, worker, cfg, settings, AuthRuntime.Create(cfg, db), null, log)
            .MapRoutes(app);

        var actual = new List<(string, string, string)>();
        foreach (RouteEndpoint endpoint in ((IEndpointRouteBuilder)app).DataSources
                     .SelectMany(source => source.Endpoints).OfType<RouteEndpoint>())
        {
            string pattern = endpoint.RoutePattern.RawText ?? "";
            if (pattern == "/{*path}")
            {
                continue; // the SPA catch-all: serves files, answers 404 JSON under /api/v1
            }
            IReadOnlyList<string> methods =
                endpoint.Metadata.GetMetadata<IHttpMethodMetadata>()?.HttpMethods ?? ["*"];
            string guard = endpoint.Metadata.GetMetadata<Guard>()?.Name ?? "<NO GUARD>";
            foreach (string method in methods)
            {
                actual.Add((method, pattern, guard));
            }
        }

        Assert.That(actual, Is.EquivalentTo(RouteTable.Select(r => (r.Method, r.Path, r.Guard))));
    }

    // -- the gate --------------------------------------------------------------------

    private const string Secret = "unit-test-secret-0123456789abcdef";

    private static string B64Url(byte[] data) =>
        Convert.ToBase64String(data).TrimEnd('=').Replace('+', '-').Replace('/', '_');

    /// <summary>A token built by hand, the way an attacker would, independent of Tokens.</summary>
    private static string Forge(string claimsJson, string secret, string alg = "HS256")
    {
        string signing = B64Url(Encoding.UTF8.GetBytes($"{{\"alg\":\"{alg}\",\"typ\":\"JWT\"}}")) + "." +
                         B64Url(Encoding.UTF8.GetBytes(claimsJson));
        byte[] mac = HMACSHA256.HashData(Encoding.UTF8.GetBytes(secret), Encoding.UTF8.GetBytes(signing));
        return signing + "." + B64Url(mac);
    }

    private static HttpRequest Bearer(string token)
    {
        var context = new DefaultHttpContext();
        context.Request.Headers.Authorization = "Bearer " + token;
        return context.Request;
    }

    private (Authenticator Gate, AuthRuntime Auth, FileStore Db) Gate(string mode, string secret = Secret)
    {
        FileStore db = Store();
        AuthRuntime auth = AuthRuntime.Create(
            new Config.Settings { AuthMode = mode, JwtSecret = secret }, db);
        return (new Authenticator(db, auth), auth, db);
    }

    /// <summary>
    /// Every PIN session is the administrator, so IGNORING a uid would promote a viewer's live token
    /// the moment the service is switched back to PIN. It must be refused.
    /// </summary>
    [Test]
    public void PinModeRefusesATokenCarryingAUid()
    {
        (Authenticator gate, _, _) = Gate("pin");
        long exp = DateTimeOffset.UtcNow.AddHours(1).ToUnixTimeSeconds();

        Assert.That(gate.SessionIdentity(Bearer(Forge(
            $"{{\"sub\":\"operator\",\"role\":\"admin\",\"exp\":{exp}}}", Secret))),
            Is.EqualTo(Identity.Session), "a PIN token is the operator");
        Assert.That(gate.SessionIdentity(Bearer(Forge(
            $"{{\"sub\":\"viewer1\",\"uid\":2,\"tv\":1,\"role\":\"viewer\",\"exp\":{exp}}}", Secret))),
            Is.Null, "an account token in PIN mode");
        Assert.That(gate.SessionIdentity(Bearer(Forge(
            $"{{\"sub\":\"x\",\"uid\":null,\"exp\":{exp}}}", Secret))),
            Is.Null, "a uid key is refused even when it is not an integer");
    }

    /// <summary>The published default never signs: a token made with it is worthless.</summary>
    [TestCase("changeme-in-production")]
    [TestCase("  changeme-in-production  ")]
    [TestCase("")]
    [TestCase("   ")]
    public void ThePublishedDefaultSecretSignsNothing(string configured)
    {
        (Authenticator gate, AuthRuntime auth, _) = Gate("pin", configured);
        long exp = DateTimeOffset.UtcNow.AddHours(1).ToUnixTimeSeconds();

        Assert.That(Tokens.SecretIsEphemeral(auth.Tokens), Is.True);
        Assert.That(Tokens.SigningSecret(auth.Tokens), Is.Not.EqualTo(Tokens.DefaultJwtSecret));
        Assert.That(Tokens.SigningSecret(auth.Tokens).Length, Is.GreaterThanOrEqualTo(64),
            "48 random bytes, base64url");
        Assert.That(gate.SessionIdentity(Bearer(Forge(
            $"{{\"sub\":\"operator\",\"role\":\"admin\",\"exp\":{exp}}}", Tokens.DefaultJwtSecret))),
            Is.Null);

        // Its own tokens still work: the process secret is stable for the life of the process.
        string own = Tokens.CreateAccessToken(auth.Tokens, new Tokens.Claims { Sub = "operator" });
        Assert.That(gate.SessionIdentity(Bearer(own)), Is.EqualTo(Identity.Session));
    }

    [Test]
    public void TheAlgorithmIsPinned()
    {
        (Authenticator gate, _, _) = Gate("pin");
        long exp = DateTimeOffset.UtcNow.AddHours(1).ToUnixTimeSeconds();
        string claims = $"{{\"sub\":\"operator\",\"exp\":{exp}}}";
        // Signed correctly with HS256, but the header claims another algorithm.
        Assert.That(gate.SessionIdentity(Bearer(Forge(claims, Secret, alg: "HS512"))), Is.Null);
        Assert.That(gate.SessionIdentity(Bearer(Forge(claims, Secret, alg: "hs256"))), Is.Null);
        string none = Forge(claims, Secret, alg: "none");
        Assert.That(gate.SessionIdentity(Bearer(none[..(none.LastIndexOf('.') + 1)])), Is.Null);
        Assert.That(gate.SessionIdentity(Bearer(Forge(claims, Secret))), Is.Not.Null);
    }

    /// <summary>
    /// Users mode loads the account on every request: a stale version, a disabled account or a
    /// deleted one is no session, and a pending password change is a 403 everywhere but two routes.
    /// </summary>
    [Test]
    public void UsersModeChecksTheStoreOnEveryRequest()
    {
        (Authenticator gate, AuthRuntime auth, _) = Gate("users");
        User admin = auth.Users.Create("alpha", "Alpha1234", Roles.Admin, mustChangePassword: false);
        User viewer = auth.Users.Create("vera", "Vera12345", Roles.Viewer, mustChangePassword: true);
        long exp = DateTimeOffset.UtcNow.AddHours(1).ToUnixTimeSeconds();
        string TokenFor(User u, int tv) =>
            Forge($"{{\"sub\":\"{u.Username}\",\"uid\":{u.Id},\"tv\":{tv},\"role\":\"admin\",\"exp\":{exp}}}",
                Secret);

        Identity? who = gate.SessionIdentity(Bearer(TokenFor(viewer, 1)));
        Assert.That(who, Is.Not.Null);
        Assert.That(who!.Role, Is.EqualTo(Roles.Viewer), "the role comes from the store, not the token");
        Assert.That(who.Username, Is.EqualTo("vera"));

        Assert.That(() => gate.RequireRole(Bearer(TokenFor(viewer, 1)), Roles.Viewer),
            Throws.TypeOf<ServiceException>().With.Message.EqualTo("password_change_required"));
        Assert.That(gate.RequireSessionAllowPasswordChange(Bearer(TokenFor(viewer, 1))), Is.Not.Null);
        Assert.That(() => gate.RequireRole(Bearer(TokenFor(admin, 1)), Roles.Admin), Throws.Nothing);

        Assert.That(gate.SessionIdentity(Bearer(TokenFor(admin, 2))), Is.Null, "wrong token version");
        Assert.That(gate.SessionIdentity(Bearer(Forge(
            $"{{\"sub\":\"operator\",\"role\":\"admin\",\"exp\":{exp}}}", Secret))), Is.Null, "a PIN-era token");
        Assert.That(gate.SessionIdentity(Bearer(Forge(
            $"{{\"sub\":\"alpha\",\"uid\":1.0,\"tv\":1,\"exp\":{exp}}}", Secret))), Is.Null, "uid 1.0 is not an integer");

        auth.Users.Update(viewer, isActive: false);
        Assert.That(gate.SessionIdentity(Bearer(TokenFor(viewer, 2))), Is.Null, "a disabled account");
    }

    // -- the account rules, under concurrency ---------------------------------------------

    /// <summary>
    /// Two admins demote each other at the same instant; one must survive.
    ///
    /// <para>
    /// **The interleaving is forced, not hoped for.** A plain two-thread race passes with the lock
    /// removed, because check and write take microseconds and the threads almost never overlap. So the
    /// admin count waits at a barrier AFTER it is computed: without the lock both threads count before
    /// either writes — the bug — and both demotions land; with the lock the second thread cannot reach
    /// the count, the barrier times out, and the first proceeds alone.
    /// </para>
    /// </summary>
    [Test]
    public void ConcurrentDemotionsCannotRemoveEveryAdministrator()
    {
        var users = new Users(Store());
        User first = users.Create("alpha", "Alpha1234", Roles.Admin);
        User second = users.Create("bravo", "Bravo1234", Roles.Admin);

        using var counted = new Barrier(2);
        users.AfterAdminCount = () => counted.SignalAndWait(TimeSpan.FromMilliseconds(500));

        using var start = new Barrier(2);
        var outcomes = new System.Collections.Concurrent.ConcurrentBag<string>();
        void Demote(User user)
        {
            start.SignalAndWait();
            try
            {
                users.Update(user, role: Roles.Viewer);
                outcomes.Add("demoted");
            }
            catch (UserError)
            {
                outcomes.Add("refused");
            }
        }

        var threads = new[] { new Thread(() => Demote(first)), new Thread(() => Demote(second)) };
        foreach (Thread t in threads) t.Start();
        foreach (Thread t in threads) t.Join();

        users.AfterAdminCount = null;
        Assert.That(users.CountActiveAdmins(), Is.EqualTo(1));
        Assert.That(outcomes.OrderBy(o => o), Is.EqualTo(new[] { "demoted", "refused" }));
    }

    /// <summary>
    /// Verification takes ~100 ms outside the lock. A sign-in that then saved the copy it loaded before
    /// hashing would silently overwrite a demotion made in that window.
    /// </summary>
    [Test]
    public void ASignInDoesNotUndoADemotionMadeWhileItWasHashing()
    {
        var users = new Users(Store());
        users.Create("alpha", "Alpha1234", Roles.Admin);
        User target = users.Create("worker", "Worker1234", Roles.Operator);

        bool fired = false;
        users.AfterVerify = () =>
        {
            if (!fired)
            {
                fired = true;
                users.Update(users.Get(target.Id)!, role: Roles.Viewer);
            }
        };
        User? signedIn = users.Authenticate("worker", "Worker1234");
        users.AfterVerify = null;

        Assert.That(fired, Is.True, "the race was not exercised");
        Assert.That(signedIn, Is.Not.Null);
        Assert.That(signedIn!.Role, Is.EqualTo(Roles.Viewer));
        Assert.That(signedIn.LastLoginAt, Is.Not.Null);
        Assert.That(users.Get(target.Id)!.Role, Is.EqualTo(Roles.Viewer));
    }

    /// <summary>A password reset in that same window makes the in-flight sign-in fail.</summary>
    [Test]
    public void ASignInFailsIfThePasswordChangedWhileItWasHashing()
    {
        var users = new Users(Store());
        User target = users.Create("worker", "Worker1234", Roles.Operator);
        users.AfterVerify = () => users.ResetPassword(users.Get(target.Id)!, "Other-Pass9");
        Assert.That(users.Authenticate("worker", "Worker1234"), Is.Null);
    }

    [Test]
    public void StoreReadsHandOutCopies()
    {
        FileStore db = Store();
        var users = new Users(db);
        User user = users.Create("alpha", "Alpha1234", Roles.Admin);

        User held = db.GetUser(user.Id)!;
        held.Role = Roles.Viewer;                         // an edit nobody saved
        db.FindUser("ALPHA")!.IsActive = false;
        db.AllUsers()[0].TokenVersion = 99;
        user.DisplayName = "mutated after put";

        User fresh = db.GetUser(user.Id)!;
        Assert.That(fresh.Role, Is.EqualTo(Roles.Admin));
        Assert.That(fresh.IsActive, Is.True);
        Assert.That(fresh.TokenVersion, Is.EqualTo(1));
        Assert.That(fresh.DisplayName, Is.EqualTo(""));

        AuditEntry entry = db.AppendAudit(new AuditEntry { Action = "login", Actor = "alpha" });
        entry.Actor = "someone-else";
        db.RecentAudit(10, null, null)[0].Detail = "edited";
        Assert.That(db.RecentAudit(10, null, null)[0].Actor, Is.EqualTo("alpha"));
        Assert.That(db.RecentAudit(10, null, null)[0].Detail, Is.EqualTo(""));
    }

    [Test]
    public void TheLastActiveAdministratorRules()
    {
        var users = new Users(Store());
        User admin = users.Create("alpha", "Alpha1234", Roles.Admin);
        User other = users.Create("bravo", "Bravo1234", Roles.Admin);

        Assert.That(() => users.Delete(admin, admin.Id),
            Throws.TypeOf<UserError>().With.Message.EqualTo("You cannot delete your own account"));
        users.Update(other, isActive: false);
        Assert.That(() => users.Update(admin, role: Roles.Viewer),
            Throws.TypeOf<UserError>().With.Message.EqualTo("Cannot demote the last active administrator"));
        Assert.That(() => users.Update(admin, isActive: false),
            Throws.TypeOf<UserError>().With.Message.EqualTo("Cannot deactivate the last active administrator"));
        Assert.That(() => users.Delete(admin, actingUserId: null),
            Throws.TypeOf<UserError>().With.Message.EqualTo("Cannot delete the last active administrator"));
        Assert.That(() => users.Update(admin, role: "boss"),
            Throws.TypeOf<UserError>().With.Message.EqualTo("Unknown role 'boss'"));
        Assert.That(() => users.Create("ALPHA", "Alpha1234", Roles.Viewer),
            Throws.TypeOf<UserError>().With.Message.EqualTo("User 'ALPHA' already exists"));
        Assert.That(() => users.Create("аdmin", "Alpha1234", Roles.Viewer),
            Throws.TypeOf<UserError>().With.Message.StartsWith("Username may contain only Latin letters"));

        // An authority change bumps the version; a display-name change does not.
        int tv = users.Get(other.Id)!.TokenVersion;
        users.Update(users.Get(other.Id)!, displayName: "  Браво ");
        Assert.That(users.Get(other.Id)!.TokenVersion, Is.EqualTo(tv));
        Assert.That(users.Get(other.Id)!.DisplayName, Is.EqualTo("Браво"));
        users.Update(users.Get(other.Id)!, role: Roles.Operator);
        Assert.That(users.Get(other.Id)!.TokenVersion, Is.EqualTo(tv + 1));
    }

    // -- storage -----------------------------------------------------------------------------

    [Test]
    public void UsersJsonIsReadableUtf8AndSurvivesARestart()
    {
        var users = new Users(Store());
        users.Create("ivan", "Ivan12345", Roles.Viewer, displayName: "Иван Петров");
        string text = File.ReadAllText(Path.Combine(_dir, "users.json"), Encoding.UTF8);
        Assert.That(text, Does.Contain("Иван Петров"), "Cyrillic must not be \\u-escaped");
        Assert.That(text, Does.Contain("\"password_hash\": \"$argon2id$v=19$m=65536,t=3,p=4$"));
        Assert.That(text, Does.Contain("\"token_version\": 1"));

        User? reloaded = Store().FindUser("IVAN");
        Assert.That(reloaded?.DisplayName, Is.EqualTo("Иван Петров"));
        Assert.That(Passwords.Verify(reloaded!.PasswordHash, "Ivan12345"), Is.True);
    }

    [Test]
    public void UnreadableFilesStartEmptyAndAFailedAuditWriteIsSwallowed()
    {
        File.WriteAllText(Path.Combine(_dir, "users.json"), "{ not json");
        File.WriteAllText(Path.Combine(_dir, "audit.jsonl"), "{\"id\":1,\"action\":\"login\"}\n{ broken\n");
        FileStore db = Store();
        Assert.That(db.AllUsers(), Is.Empty);
        Assert.That(db.RecentAudit(10, null, null).Select(e => e.Action), Is.EqualTo(new[] { "login" }),
            "one bad line, not a bad log");

        // The file becomes unwritable: the action must still succeed.
        File.Delete(Path.Combine(_dir, "audit.jsonl"));
        Directory.CreateDirectory(Path.Combine(_dir, "audit.jsonl"));
        Assert.That(() => Audit.Record(db, "login", actor: "pin"), Throws.Nothing);
        Assert.That(db.RecentAudit(10, "login", "PI").Count, Is.EqualTo(1));
    }

    // -- the throttle ------------------------------------------------------------------------

    private sealed class FakeClock
    {
        public double Now;
        public double Read() => Now;
    }

    [Test]
    public void ThrottleLocksOneAccountFromOneAddress()
    {
        var clock = new FakeClock { Now = 1000 };
        var throttle = new LoginThrottle(3, 120, clock.Read);
        for (int i = 0; i < 3; i++)
        {
            Assert.That(throttle.BlockedFor("admin", "10.0.0.1"), Is.Zero);
            throttle.NoteFailure(" Admin ", "10.0.0.1");       // trimmed and case-folded
            clock.Now += 10;
        }
        // Oldest failure at 1000, now 1030: 120 - 30 = 90 s left.
        Assert.That(throttle.BlockedFor("admin", "10.0.0.1"), Is.EqualTo(90));
        Assert.That(throttle.BlockedFor("bob", "10.0.0.1"), Is.Zero, "another account");
        Assert.That(throttle.BlockedFor("admin", "10.0.0.2"), Is.Zero, "another address");

        clock.Now = 1000 + 119.5;
        Assert.That(throttle.BlockedFor("admin", "10.0.0.1"), Is.EqualTo(1), "never 0 while refused");
        clock.Now = 1000 + 120;
        Assert.That(throttle.BlockedFor("admin", "10.0.0.1"), Is.Zero, "the oldest left the window");
    }

    [Test]
    public void RotatingUsernamesDoesNotEscapeTheAddressLockout()
    {
        var clock = new FakeClock { Now = 0 };
        var throttle = new LoginThrottle(3, 120, clock.Read);
        for (int i = 0; i < 9; i++)
        {
            throttle.NoteFailure($"spray{i}", "10.0.0.1");
        }
        Assert.That(throttle.BlockedFor("never-tried", "10.0.0.1"), Is.GreaterThan(0));
        Assert.That(throttle.BlockedFor("never-tried", "10.0.0.2"), Is.Zero);
    }

    [Test]
    public void ASuccessClearsOnlyTheAccountCounter()
    {
        var clock = new FakeClock { Now = 0 };
        var throttle = new LoginThrottle(3, 120, clock.Read);
        for (int i = 0; i < 3; i++)
        {
            throttle.NoteFailure("alice", "10.0.0.1");
        }
        Assert.That(throttle.BlockedFor("alice", "10.0.0.1"), Is.GreaterThan(0));
        throttle.Clear("alice", "10.0.0.1");
        Assert.That(throttle.BlockedFor("alice", "10.0.0.1"), Is.Zero);

        // The three failures still count against the address: six more reach 3 x 3.
        for (int i = 0; i < 5; i++)
        {
            throttle.NoteFailure($"other{i}", "10.0.0.1");
        }
        Assert.That(throttle.BlockedFor("zed", "10.0.0.1"), Is.Zero, "8 of 9");
        throttle.NoteFailure("other5", "10.0.0.1");
        Assert.That(throttle.BlockedFor("zed", "10.0.0.1"), Is.GreaterThan(0),
            "a success must not have reset the address-wide counter");
    }

    [Test]
    public void ModeResolutionNeverThrows()
    {
        Assert.That(AuthMode.Resolve(null, "files"), Is.EqualTo(("pin", (string?)null)));
        Assert.That(AuthMode.Resolve("  USERS ", "files"), Is.EqualTo(("users", (string?)null)));
        (string mode, string? reason) = AuthMode.Resolve("bogus", "files");
        Assert.That(mode, Is.EqualTo("pin"));
        Assert.That(reason, Is.EqualTo(
            "AUTH_MODE='bogus' is not one of pin, users — falling back to PIN authentication"));
        (mode, reason) = AuthMode.Resolve("users", "sql");
        Assert.That(mode, Is.EqualTo("pin"));
        Assert.That(reason, Does.StartWith("AUTH_MODE=users is implemented for the temporary file store only"));
        Assert.That(PyRepr.Quote("it's"), Is.EqualTo("\"it's\""));
    }
}
