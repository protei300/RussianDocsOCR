using System.Diagnostics;

namespace RussianDocs.Service.Auth;

/// <summary>
/// Failed-login throttling, in memory, for one process.
///
/// <para>
/// Honest about what it is: the service is pinned to a single process, so a dictionary is the whole
/// mechanism. A deployment behind several instances needs shared state — Redis, or the reverse
/// proxy's own rate limiting — and <c>docs/auth.md</c> says so.
/// </para>
///
/// <para>
/// **Two counters, and both are needed** (ports/AUTH.md §6):
/// </para>
/// <list type="bullet">
/// <item><b>per (identity, address)</b>, limit <c>LOGIN_MAX_ATTEMPTS</c>. Locking by account alone
/// would let anyone lock a known user out by failing on purpose from anywhere;</item>
/// <item><b>per address across every identity</b>, under the key identity <c>*</c> (not a valid
/// username, so it cannot collide), limit three times that. Without it the first counter is defeated
/// by trying a different username each time — which is exactly what password spraying is. The first
/// Python version had only the first counter.</item>
/// </list>
///
/// <para>
/// The address is <c>HttpContext.Connection.RemoteIpAddress</c> and NEVER <c>X-Forwarded-For</c>: a
/// header the client writes cannot be the key that limits the client. It lives here, in memory, for the
/// life of the process, and never reaches the audit log. The clock is MONOTONIC
/// (<see cref="Stopwatch"/>), so a wall-clock step cannot lift or extend a lockout. Port of the
/// throttle in <c>service/core/auth.py</c>.
/// </para>
/// </summary>
public sealed class LoginThrottle(int maxAttempts, int windowSeconds, Func<double>? clock = null)
{
    /// <summary>The address-wide counter's identity. Not a valid username — see Users.</summary>
    public const string AnyIdentity = "*";

    /// <summary>
    /// Failures from one address across ALL identities before it is blocked, as a multiple of the
    /// per-account limit. Higher than that limit because several people share an office NAT; lower
    /// than unlimited because otherwise rotating usernames is a free pass.
    /// </summary>
    public const int AddressLimitFactor = 3;

    /// <summary>Above this many keys, a failure also sweeps keys with nothing left in the window.</summary>
    private const int SweepThreshold = 10_000;

    private readonly Func<double> _now = clock ?? (() => Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency);
    private readonly object _gate = new();
    private readonly Dictionary<(string Identity, string Address), List<double>> _attempts = [];

    private static (string, string) Key(string identity, string address) =>
        ((identity ?? "").Trim().ToLowerInvariant(), string.IsNullOrEmpty(address) ? "-" : address);

    /// <summary>
    /// Seconds remaining in a lockout, or 0 when the caller may try.
    ///
    /// <para>
    /// <c>n = max(1, floor(window − (now − oldest failure in window)))</c>, the larger of the two
    /// counters' answers — so a <c>Retry-After</c> is never 0 while the request is still refused.
    /// </para>
    /// </summary>
    public int BlockedFor(string identity, string address)
    {
        double now = _now();
        lock (_gate)
        {
            return Math.Max(
                Blocked(Key(identity, address), maxAttempts, now),
                Blocked(Key(AnyIdentity, address), maxAttempts * AddressLimitFactor, now));
        }
    }

    private int Blocked((string, string) key, int limit, double now)
    {
        if (!_attempts.TryGetValue(key, out List<double>? times))
        {
            return 0;
        }
        List<double> recent = times.Where(t => now - t < windowSeconds).ToList();
        if (recent.Count < limit)
        {
            return 0;
        }
        return Math.Max(1, (int)Math.Floor(windowSeconds - (now - recent[0])));
    }

    /// <summary>A failure counts against BOTH counters.</summary>
    public void NoteFailure(string identity, string address)
    {
        double now = _now();
        lock (_gate)
        {
            foreach ((string, string) key in new[] { Key(identity, address), Key(AnyIdentity, address) })
            {
                List<double> recent = _attempts.TryGetValue(key, out List<double>? times)
                    ? times.Where(t => now - t < windowSeconds).ToList()
                    : [];
                recent.Add(now);
                _attempts[key] = recent;
            }
            // Opportunistic sweep: without it the map grows once per distinct (identity, address)
            // pair for the life of the process — and a spraying run makes those cheaply.
            if (_attempts.Count > SweepThreshold)
            {
                foreach ((string, string) stale in _attempts
                             .Where(e => !e.Value.Any(t => now - t < windowSeconds))
                             .Select(e => e.Key).ToList())
                {
                    _attempts.Remove(stale);
                }
            }
        }
    }

    /// <summary>
    /// After a successful sign-in, so one typo does not linger.
    ///
    /// <para>
    /// **The identity's counter only, never the address-wide one**: otherwise an attacker holding one
    /// valid account could reset their budget by signing in between guesses at everyone else's.
    /// </para>
    /// </summary>
    public void Clear(string identity, string address)
    {
        lock (_gate)
        {
            _attempts.Remove(Key(identity, address));
        }
    }

    /// <summary>Test seam: how many keys the map holds.</summary>
    internal int KeyCount
    {
        get
        {
            lock (_gate)
            {
                return _attempts.Count;
            }
        }
    }
}
