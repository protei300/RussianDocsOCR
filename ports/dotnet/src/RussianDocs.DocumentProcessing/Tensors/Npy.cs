using System.Globalization;
using System.Text;

namespace RussianDocs.DocumentProcessing.Tensors;

/// <summary>
/// Reader and writer for the `.npy` subset defined in <c>conformance/spec/npy-subset.md</c>.
///
/// <para>
/// Every port needs this regardless of the harness, because <c>DocTypeAngles</c> ships its centroids
/// as <c>centers.npz</c> — a zip of three `.npy` members. The harness reuses it rather than adding a
/// second serialisation format.
/// </para>
///
/// <para>
/// **The header is a Python literal, not JSON.** It uses single quotes, <c>True</c>/<c>False</c>,
/// and a trailing comma in one-element tuples (<c>'shape': (3,)</c>). A JSON parser appears to work
/// on the common cases and then fails on exactly those, which is why this parses the literal
/// directly.
/// </para>
/// </summary>
public static class Npy
{
    private static readonly byte[] Magic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

    public static NdArray Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Read(stream, path);
    }

    public static NdArray Parse(byte[] blob, string origin = "<memory>")
    {
        using var stream = new MemoryStream(blob, writable: false);
        return Read(stream, origin);
    }

    public static NdArray Read(Stream stream, string origin)
    {
        var magic = ReadExactly(stream, 6, origin);
        if (!magic.AsSpan().SequenceEqual(Magic))
        {
            throw new InvalidDataException($"npy: {origin}: not a .npy file");
        }

        var version = ReadExactly(stream, 2, origin);
        if (version[0] != 1 || version[1] != 0)
        {
            // 2.0 and 3.0 differ only in header length and encoding, and nothing in this project
            // writes them. Refusing is better than half-supporting: a 2.0 file read as 1.0 gives a
            // plausible-looking shape from the wrong bytes.
            throw new InvalidDataException(
                $"npy: {origin}: version {version[0]}.{version[1]}, only 1.0 is supported");
        }

        int headerLen = BitConverter.ToUInt16(ReadExactly(stream, 2, origin));
        string header = Encoding.ASCII.GetString(ReadExactly(stream, headerLen, origin));

        (Dtype dtype, int itemSize) = ParseDescr(Field(header, "descr", origin), origin);
        if (Field(header, "fortran_order", origin) is not "False")
        {
            throw new InvalidDataException($"npy: {origin}: fortran_order must be False");
        }
        int[] shape = ParseShape(Field(header, "shape", origin), origin);

        long expected = (long)NdArray.Count(shape) * itemSize;
        var data = ReadExactly(stream, checked((int)expected), origin);
        return new NdArray(data, shape, dtype, itemSize);
    }

    public static void Save(string path, NdArray array)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var stream = File.Create(path);
        Write(stream, array);
    }

    public static void Write(Stream stream, NdArray array)
    {
        string descr = DescrOf(array);
        // The trailing comma on a one-element tuple is required by the format, not decoration:
        // `(3)` is the integer 3 in Python, `(3,)` is a tuple. NumPy writes the comma and a reader
        // that expects it will reject a file written without one.
        string shape = array.Shape.Length == 1
            ? $"({array.Shape[0]},)"
            : $"({string.Join(", ", array.Shape)}{(array.Shape.Length == 0 ? "" : ",")})";

        string header = $"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}";

        // NumPy pads the header so that magic + version + length + header is a multiple of 64,
        // aligning the payload. Readers that ignore the padding still work, but writing it keeps
        // the files byte-comparable with the reference's own output.
        int prefix = Magic.Length + 2 + 2;
        int padded = ((prefix + header.Length + 1 + 63) / 64) * 64;
        header = header.PadRight(padded - prefix - 1) + "\n";

        stream.Write(Magic);
        stream.WriteByte(1);
        stream.WriteByte(0);
        stream.Write(BitConverter.GetBytes((ushort)header.Length));
        stream.Write(Encoding.ASCII.GetBytes(header));
        stream.Write(array.Data);
    }

    private static string DescrOf(NdArray a) => a.Dtype switch
    {
        Dtype.Float32 => "<f4",
        Dtype.Float64 => "<f8",
        Dtype.UInt8 => "|u1",
        Dtype.Int64 => "<i8",
        Dtype.Unicode => $"<U{a.ItemSize / 4}",
        _ => throw new InvalidDataException($"npy: cannot write dtype {a.Dtype}"),
    };

    private static (Dtype, int) ParseDescr(string descr, string origin) => descr switch
    {
        "<f4" or "=f4" or "f4" => (Dtype.Float32, 4),
        "<f8" or "=f8" or "f8" => (Dtype.Float64, 8),
        "|u1" or "u1" or "B" => (Dtype.UInt8, 1),
        "<i8" or "=i8" or "i8" => (Dtype.Int64, 8),
        _ when descr.StartsWith("<U", StringComparison.Ordinal) =>
            (Dtype.Unicode, 4 * int.Parse(descr[2..], CultureInfo.InvariantCulture)),
        // Big-endian is refused rather than byte-swapped: nothing here produces it, and a silent
        // swap would hide a genuinely wrong file.
        _ => throw new InvalidDataException($"npy: {origin}: unsupported dtype '{descr}'"),
    };

    /// <summary>Pulls one value out of the Python-literal header by key.</summary>
    private static string Field(string header, string key, string origin)
    {
        string needle = $"'{key}':";
        int at = header.IndexOf(needle, StringComparison.Ordinal);
        if (at < 0)
        {
            throw new InvalidDataException($"npy: {origin}: header has no '{key}'");
        }
        int start = at + needle.Length;
        while (start < header.Length && header[start] == ' ')
        {
            start++;
        }

        if (header[start] == '\'')
        {
            int end = header.IndexOf('\'', start + 1);
            return header[(start + 1)..end];
        }
        if (header[start] == '(')
        {
            int end = header.IndexOf(')', start);
            return header[start..(end + 1)];
        }
        int stop = start;
        while (stop < header.Length && header[stop] is not (',' or '}' or ' '))
        {
            stop++;
        }
        return header[start..stop];
    }

    private static int[] ParseShape(string tuple, string origin)
    {
        string inner = tuple.Trim('(', ')').Trim();
        if (inner.Length == 0)
        {
            return []; // '()' is a scalar: one element
        }
        var parts = inner.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var dims = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!int.TryParse(parts[i], NumberStyles.Integer, CultureInfo.InvariantCulture, out dims[i]))
            {
                throw new InvalidDataException($"npy: {origin}: bad shape '{tuple}'");
            }
        }
        return dims;
    }

    private static byte[] ReadExactly(Stream stream, int count, string origin)
    {
        var buffer = new byte[count];
        int read = 0;
        while (read < count)
        {
            int n = stream.Read(buffer, read, count - read);
            if (n <= 0)
            {
                throw new InvalidDataException(
                    $"npy: {origin}: truncated — wanted {count} bytes, got {read}");
            }
            read += n;
        }
        return buffer;
    }
}
