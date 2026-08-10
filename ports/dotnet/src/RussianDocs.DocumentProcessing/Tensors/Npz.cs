using System.IO.Compression;

namespace RussianDocs.DocumentProcessing.Tensors;

/// <summary>
/// Reads a <c>.npz</c> — a zip of `.npy` members.
///
/// <para>
/// One caller: <c>DocTypeAngles</c>'s <c>centers.npz</c>, holding <c>labels</c>, <c>centers</c> and
/// <c>max_distance</c>. It replaced a pickle, which was both a code-execution vector and fragile
/// across NumPy versions — worth remembering before anyone proposes a "more convenient" format.
/// </para>
/// </summary>
public static class Npz
{
    /// <summary>Loads every member, keyed by name WITHOUT the <c>.npy</c> suffix.</summary>
    public static Dictionary<string, NdArray> Load(string path)
    {
        var result = new Dictionary<string, NdArray>(StringComparer.Ordinal);
        using ZipArchive zip = ZipFile.OpenRead(path);
        foreach (ZipArchiveEntry entry in zip.Entries)
        {
            // Read fully into memory first: the zip stream is forward-only and non-seekable, while
            // the .npy reader needs to read a header and then a payload of a size the header names.
            using var buffer = new MemoryStream();
            using (Stream member = entry.Open())
            {
                member.CopyTo(buffer);
            }

            string name = entry.Name.EndsWith(".npy", StringComparison.OrdinalIgnoreCase)
                ? entry.Name[..^4]
                : entry.Name;
            result[name] = Npy.Parse(buffer.ToArray(), $"{path}!{entry.Name}");
        }
        return result;
    }
}
