using System.Runtime.InteropServices;

namespace RussianDocs.DocumentProcessing.Tensors;

/// <summary>The element types the `.npy` subset allows. Names match the NumPy descriptors.</summary>
public enum Dtype
{
    /// <summary><c>&lt;f4</c> — float32. What every model input and output uses.</summary>
    Float32,
    /// <summary><c>&lt;f8</c> — float64. Appears in intermediates, never in a model tensor.</summary>
    Float64,
    /// <summary><c>|u1</c> — uint8. Images and the OCR input.</summary>
    UInt8,
    /// <summary><c>&lt;i8</c> — int64. Label indices.</summary>
    Int64,
    /// <summary><c>&lt;U&lt;n&gt;</c> — fixed-width UTF-32 strings. Only <c>centers.npz</c> uses it.</summary>
    Unicode,
}

/// <summary>
/// A dense, C-contiguous N-dimensional array — the port's stand-in for <c>numpy.ndarray</c>.
///
/// <para>
/// Bytes rather than a typed array, deliberately. The dtype is not known until a `.npy` header or a
/// `model.json` has been read, so a generic <c>NdArray&lt;T&gt;</c> would have to be constructed
/// through reflection at every load site; all three ports concluded the same thing and settled on a
/// byte payload plus a dtype tag, with typed spans for access. Keeping that identical across
/// languages is worth more here than type safety at the seams.
/// </para>
///
/// <para>
/// **C-contiguous only.** Fortran order is an error, not a mode: the reference never produces it,
/// and silently accepting it would let a transposed tensor reach a model and be graded as a numeric
/// divergence instead of a shape bug.
/// </para>
/// </summary>
public sealed class NdArray
{
    /// <summary>Raw little-endian payload, C-contiguous.</summary>
    public byte[] Data { get; }

    /// <summary>Dimensions. Empty means a scalar — one element, not zero.</summary>
    public int[] Shape { get; }

    public Dtype Dtype { get; }

    /// <summary>
    /// Bytes per element. For <see cref="Dtype.Unicode"/> this is <c>4 * n</c>, because NumPy stores
    /// <c>&lt;U&lt;n&gt;</c> as fixed-width UTF-32 padded with NULs — the trap that turns a naive
    /// byte-slicing label decoder into a list of empty strings.
    /// </summary>
    public int ItemSize { get; }

    public NdArray(byte[] data, int[] shape, Dtype dtype, int itemSize)
    {
        int count = Count(shape);
        long expected = (long)count * itemSize;
        if (data.LongLength != expected)
        {
            throw new ArgumentException(
                $"tensor: payload is {data.LongLength} bytes, shape {Describe(shape)} of " +
                $"{itemSize}-byte items needs {expected}");
        }
        Data = data;
        Shape = shape;
        Dtype = dtype;
        ItemSize = itemSize;
    }

    /// <summary>Total element count. A zero-length shape is a SCALAR: one element.</summary>
    public static int Count(int[] shape)
    {
        int n = 1;
        foreach (int d in shape)
        {
            if (d < 0)
            {
                throw new ArgumentException($"tensor: negative dimension in {Describe(shape)}");
            }
            n *= d;
        }
        return n;
    }

    public int Length => Count(Shape);

    public static NdArray FromFloat32(float[] values, params int[] shape)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(bytes);
        return new NdArray(bytes, shape, Dtype.Float32, sizeof(float));
    }

    public static NdArray FromUInt8(byte[] values, params int[] shape) =>
        new(values, shape, Dtype.UInt8, 1);

    /// <summary>float32 view. Throws on any other dtype rather than reinterpreting.</summary>
    public ReadOnlySpan<float> AsFloat32() =>
        Dtype == Dtype.Float32
            ? MemoryMarshal.Cast<byte, float>(Data)
            : throw new InvalidOperationException($"tensor: dtype is {Dtype}, not Float32");

    public ReadOnlySpan<double> AsFloat64() =>
        Dtype == Dtype.Float64
            ? MemoryMarshal.Cast<byte, double>(Data)
            : throw new InvalidOperationException($"tensor: dtype is {Dtype}, not Float64");

    public ReadOnlySpan<long> AsInt64() =>
        Dtype == Dtype.Int64
            ? MemoryMarshal.Cast<byte, long>(Data)
            : throw new InvalidOperationException($"tensor: dtype is {Dtype}, not Int64");

    public ReadOnlySpan<byte> AsUInt8() =>
        Dtype == Dtype.UInt8
            ? Data
            : throw new InvalidOperationException($"tensor: dtype is {Dtype}, not UInt8");

    /// <summary>
    /// Decodes a <c>&lt;U&lt;n&gt;</c> array to strings.
    ///
    /// <para>
    /// NumPy stores these as fixed-width UTF-32LE, NUL-padded to <c>n</c> code points. Slicing the
    /// bytes naively yields empty strings, which is how a label array turns into nine blanks that
    /// then match nothing — a failure that reads like a broken model rather than a broken decoder.
    /// </para>
    /// </summary>
    public string[] AsUnicode()
    {
        if (Dtype != Dtype.Unicode)
        {
            throw new InvalidOperationException($"tensor: dtype is {Dtype}, not Unicode");
        }
        int codePoints = ItemSize / 4;
        var result = new string[Length];
        for (int i = 0; i < result.Length; i++)
        {
            var chars = new List<char>(codePoints);
            for (int c = 0; c < codePoints; c++)
            {
                int offset = (i * codePoints + c) * 4;
                uint cp = BitConverter.ToUInt32(Data, offset);
                if (cp == 0)
                {
                    break; // NUL padding: the string ends here, the field does not
                }
                chars.AddRange(char.ConvertFromUtf32((int)cp));
            }
            result[i] = new string(chars.ToArray());
        }
        return result;
    }

    public static string Describe(int[] shape) => $"({string.Join(", ", shape)})";

    public override string ToString() => $"NdArray{Describe(Shape)} {Dtype}";
}
