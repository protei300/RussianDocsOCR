using System.Text;
using RussianDocs.DocumentProcessing.Imaging;
using RussianDocs.Service.Store;
using Image = RussianDocs.DocumentProcessing.Imaging.Image;

namespace RussianDocs.Service.Repositories;

/// <summary>
/// Binary artifacts: the uploaded original, the rendered canvas, the thumbnail.
///
/// <para>
/// **THIS LAYER STAYS ON THE FILESYSTEM EVEN AFTER A SQL MIGRATION.** Multi-megabyte PNGs do not
/// belong in a database — in a real deployment this file grows an S3 implementation, not a BLOB
/// column. That is why it is separate from <see cref="Documents"/> rather than folded into it.
/// </para>
/// </summary>
public static class Artifacts
{
    /// <summary>
    /// The formats accepted, keyed by the bytes that actually identify them.
    ///
    /// <para>
    /// SNIFFED rather than trusting the client's Content-Type, which is attacker-controlled and
    /// routinely wrong even when it is not.
    /// </para>
    /// </summary>
    private static readonly (byte[] Prefix, string Ext, string Media)[] Magic =
    [
        ([0xff, 0xd8, 0xff], ".jpg", "image/jpeg"),
        (Encoding.Latin1.GetBytes("\x89PNG\r\n\x1a\n"), ".png", "image/png"),
        (Encoding.Latin1.GetBytes("BM"), ".bmp", "image/bmp"),
        (Encoding.Latin1.GetBytes("II*\0"), ".tif", "image/tiff"),
        (Encoding.Latin1.GetBytes("MM\0*"), ".tif", "image/tiff"),
    ];

    /// <summary>
    /// The extension and media type for a supported image, or <c>null</c>.
    ///
    /// <para>
    /// WEBP needs a two-part check — 'RIFF' at 0 and 'WEBP' at 8 — which is why it is not in the
    /// table.
    /// </para>
    /// </summary>
    public static (string Ext, string Media)? SniffImage(ReadOnlySpan<byte> data)
    {
        foreach ((byte[] prefix, string ext, string media) in Magic)
        {
            if (data.StartsWith(prefix))
            {
                return (ext, media);
            }
        }
        if (data.Length >= 12 &&
            data[..4].SequenceEqual("RIFF"u8) &&
            data[8..12].SequenceEqual("WEBP"u8))
        {
            return (".webp", "image/webp");
        }
        return null;
    }

    /// <summary>Detected separately so the error can say WHY. Users will try PDFs.</summary>
    public static bool IsPdf(ReadOnlySpan<byte> data) => data.StartsWith("%PDF"u8);

    /// <summary>The artifact directory, created.</summary>
    public static string DocDir(IDocumentStore db, int id)
    {
        string dir = db.DocDir(id);
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>
    /// Stores the upload byte-for-byte under a FIXED name.
    ///
    /// <para>
    /// The client's filename is kept on the record for display only and never touches the filesystem
    /// — so it cannot be a path-traversal vector no matter what it contains.
    /// </para>
    /// </summary>
    public static string SaveOriginal(IDocumentStore db, int id, byte[] data, string ext)
    {
        string path = Path.Combine(DocDir(db, id), "original" + ext);
        FileStore.AtomicWriteBytes(path, data);
        return path;
    }

    /// <summary>
    /// The upload's width and height, or <c>null</c> if it cannot be decoded.
    ///
    /// <para>
    /// Done SYNCHRONOUSLY at upload time so an undecodable file becomes an immediate, actionable 422
    /// instead of a mysterious failed job minutes later.
    /// </para>
    ///
    /// <para>
    /// <see cref="Io.TryDecodeSize"/> rather than a full decode: the colour conversion a full decode
    /// owes the pipeline is a second pass over the image, and nothing here reads a pixel. In the Go
    /// port that was measurable, not theoretical — ~72 ms per upload against ~22 ms.
    /// </para>
    /// </summary>
    public static (int Width, int Height)? DecodeDimensions(byte[] data) =>
        Io.TryDecodeSize(data, out int w, out int h) ? (w, h) : null;

    /// <summary>
    /// Writes the corrected canvas as PNG and returns its dimensions.
    ///
    /// <para>
    /// The canvas is RGB and the encoder expects BGR. Skipping the conversion swaps red and blue in
    /// every displayed document — and the result looks plausible enough on a passport that it can ship
    /// unnoticed. Hence the explicit conversion inside <see cref="Io.WritePngFromRgb"/> and the
    /// regression test asserting a known-red pixel stays red.
    /// </para>
    /// </summary>
    public static (string Path, int Width, int Height) SaveCanvas(IDocumentStore db, int id,
        Image rgb)
    {
        string path = Path.Combine(DocDir(db, id), "canvas.png");
        Io.WritePngFromRgb(path, rgb);
        return (path, rgb.Width, rgb.Height);
    }

    /// <summary>
    /// Writes a small JPEG for the list page.
    ///
    /// <para>
    /// Without it the log page pulls full canvases for every visible row on each three-second poll —
    /// megabytes per refresh for images rendered at 56 px wide.
    /// </para>
    /// </summary>
    public static string SaveThumbnail(IDocumentStore db, int id, Image rgb, int width)
    {
        string dir = DocDir(db, id);
        if (width <= 0)
        {
            width = 96;
        }
        int height = Math.Max(1, (rgb.Height * width + rgb.Width / 2) / rgb.Width);
        using Image small = Io.Resize(rgb, width, height, Interpolation.Area);

        string path = Path.Combine(dir, "thumb.jpg");
        Io.WriteJpegFromRgb(path, small, 80);
        return path;
    }

    /// <summary>The path and media type for "original", "canvas" or "thumb".</summary>
    public static (string Path, string Media)? OpenArtifact(IDocumentStore db, int id, string kind)
    {
        string dir = db.DocDir(id);
        switch (kind)
        {
            case "canvas":
                // PNG for anything this service rendered; JPEG for the pre-computed seed fixtures,
                // which trade exactness for a committable repository footprint.
                foreach ((string name, string media) in new[]
                         {
                             ("canvas.png", "image/png"), ("canvas.jpg", "image/jpeg"),
                         })
                {
                    string candidate = Path.Combine(dir, name);
                    if (File.Exists(candidate))
                    {
                        return (candidate, media);
                    }
                }
                return null;

            case "thumb":
                string thumb = Path.Combine(dir, "thumb.jpg");
                if (File.Exists(thumb))
                {
                    return (thumb, "image/jpeg");
                }
                // Falls back to the full canvas rather than 404ing: a missing thumbnail is a
                // performance problem, not a missing document.
                return OpenArtifact(db, id, "canvas");

            case "original":
                if (!Directory.Exists(dir))
                {
                    return null;
                }
                string[] matches = Directory.GetFiles(dir, "original.*");
                Array.Sort(matches, StringComparer.Ordinal);
                foreach (string candidate in matches)
                {
                    if (Path.GetExtension(candidate) == ".tmp")
                    {
                        continue;
                    }
                    byte[] head = new byte[16];
                    int read;
                    try
                    {
                        using FileStream stream = File.OpenRead(candidate);
                        read = stream.Read(head, 0, head.Length);
                    }
                    catch (IOException)
                    {
                        continue;
                    }
                    return SniffImage(head.AsSpan(0, read)) is { } sniffed
                        ? (candidate, sniffed.Media)
                        : (candidate, "application/octet-stream");
                }
                return null;

            default:
                return null;
        }
    }
}
