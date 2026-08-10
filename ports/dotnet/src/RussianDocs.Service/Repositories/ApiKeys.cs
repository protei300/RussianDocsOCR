using System.Security.Cryptography;
using System.Text;
using RussianDocs.Service.Auth;
using RussianDocs.Service.Model;
using RussianDocs.Service.Store;

namespace RussianDocs.Service.Repositories;

/// <summary>
/// The repository for API keys. Signatures copied from <c>service/repositories/api_keys.py</c>.
/// </summary>
public static class ApiKeys
{
    /// <summary>
    /// Reserved for the environment key so it never collides with a stored one.
    /// </summary>
    public const int DefaultKeyId = 0;

    /// <summary>
    /// The environment-provided default key is SYNTHESISED AT STARTUP, NOT STORED.
    ///
    /// <para>
    /// That keeps one awkward case honest: runtime-created keys live in the ephemeral store and vanish
    /// on restart, so if the default were merely stored too, a restart could leave the API with no
    /// working credential at all. Deriving it from the environment on every boot means it is always
    /// present — and deleting it is refused rather than silently undone by the next restart.
    /// </para>
    /// </summary>
    private static (ApiKey Key, bool Generated) DefaultKey(Tokens.Config cfg)
    {
        (string raw, bool generated) = Tokens.ResolveDefaultKey(cfg);
        return (new ApiKey
        {
            Id = DefaultKeyId,
            Label = generated ? "Default (generated at startup)" : "Default (environment)",
            Prefix = Tokens.Prefix(raw),
            KeyHash = Tokens.HashApiKey(raw),
            IsDefault = true,
        }, generated);
    }

    /// <summary>Every usable key, default first.</summary>
    public static List<ApiKey> All(IDocumentStore db, Tokens.Config cfg)
    {
        (ApiKey def, _) = DefaultKey(cfg);
        // OrderBy is stable, which matters: two keys created in the same clock tick must not swap
        // places between requests.
        List<ApiKey> stored = db.AllApiKeys()
            .OrderBy(k => k.CreatedAt ?? DateTime.MaxValue)
            .ThenBy(k => k.Id)
            .ToList();
        stored.Insert(0, def);
        return stored;
    }

    /// <summary>Mints a key. Returns the record and the PLAINTEXT, which is shown once.</summary>
    public static (ApiKey Record, string Plaintext) Create(IDocumentStore db, string label)
    {
        string raw = Tokens.GenerateApiKey();
        string name = label.Trim();
        if (name.Length == 0)
        {
            name = "Unnamed key";
        }
        int id = db.NextApiKeyId();
        if (id == DefaultKeyId)
        {
            // Never shadow the environment key: id 0 is reserved, and a stored key holding it would
            // make the default unreachable through the API.
            id = 1;
        }
        var record = new ApiKey
        {
            Id = id,
            Label = name,
            Prefix = Tokens.Prefix(raw),
            KeyHash = Tokens.HashApiKey(raw),
            IsDefault = false,
            CreatedAt = Document.UtcNow(),
        };
        db.PutApiKey(record);
        return (record, raw);
    }

    public static bool Delete(IDocumentStore db, int id) => db.DropApiKey(id);

    /// <summary>
    /// Matches a presented key against every known hash, in CONSTANT TIME.
    ///
    /// <para>
    /// A constant-time compare per candidate rather than a dictionary lookup: ordinary equality on a
    /// digest leaks how much of it matched through timing. Returning early on a match is fine — what
    /// must not vary with the secret is the comparison itself; the NUMBER of configured keys is not
    /// secret, and the list is tiny anyway.
    /// </para>
    /// </summary>
    public static ApiKey? Verify(IDocumentStore db, Tokens.Config cfg, string candidate)
    {
        if (candidate.Length == 0)
        {
            return null;
        }
        byte[] digest = Encoding.UTF8.GetBytes(Tokens.HashApiKey(candidate));
        foreach (ApiKey key in All(db, cfg))
        {
            if (CryptographicOperations.FixedTimeEquals(
                    digest, Encoding.UTF8.GetBytes(key.KeyHash)))
            {
                return key;
            }
        }
        return null;
    }

    /// <summary>Records last use. The environment key is not persisted, so it is skipped.</summary>
    public static void Touch(IDocumentStore db, ApiKey key)
    {
        if (key.IsDefault)
        {
            return;
        }
        key.LastUsedAt = Document.UtcNow();
        db.PutApiKey(key);
    }

    /// <summary>
    /// The list for the UI.
    ///
    /// <para>
    /// **The GENERATED default is returned IN FULL**: it exists only in this process's memory, so
    /// masking it would make it unusable — the operator would have no way to learn a key the service
    /// invented. A key supplied via <c>DEFAULT_API_KEY</c> stays masked, because whoever set it
    /// already has it and echoing a configured secret back into a browser is gratuitous.
    /// </para>
    /// </summary>
    public static List<Dictionary<string, object?>> Public(IDocumentStore db, Tokens.Config cfg)
    {
        (string raw, bool generated) = Tokens.ResolveDefaultKey(cfg);
        var output = new List<Dictionary<string, object?>>();
        foreach (ApiKey key in All(db, cfg))
        {
            Dictionary<string, object?> entry = key.Public();
            if (key.IsDefault)
            {
                entry["is_generated"] = generated;
                if (generated)
                {
                    entry["key"] = raw;
                }
            }
            output.Add(entry);
        }
        return output;
    }
}
