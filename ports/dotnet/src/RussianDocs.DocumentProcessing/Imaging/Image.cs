using OpenCvSharp;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Imaging;

/// <summary>
/// An owned image. Wraps an OpenCV <c>Mat</c> and must be disposed.
///
/// <para>
/// **The wrapper exists for ownership, not for abstraction.** A raw <c>Mat</c> holds unmanaged
/// memory that the GC will not reclaim on any useful timescale, and the Go port proved what that
/// costs: one code path that read a result and returned without releasing it leaked 12.7 MB per
/// document, unbounded, 663 MB to 6932 MB over 460 documents — with the conformance suite green
/// throughout, because the CLI processes one document per process. Wrapping it in
/// <see cref="IDisposable"/> lets the compiler and the analysers help, which is one of the few
/// places .NET is genuinely better placed than Go here.
/// </para>
///
/// <para>
/// Every method that returns an <see cref="Image"/> transfers OWNERSHIP to the caller. Methods that
/// take one only borrow it. Where the reference aliases an array and lets the GC sort it out, this
/// port clones — see the unsplit-field fallback in word splitting, where a borrowed Mat inside a
/// list the caller disposes is a double free that only shows up in bulk.
/// </para>
/// </summary>
public sealed class Image : IDisposable
{
    private Mat _mat;
    private bool _disposed;

    private Image(Mat mat) => _mat = mat;

    /// <summary>Takes ownership of an existing Mat.</summary>
    public static Image Wrap(Mat mat) => new(mat);

    /// <summary>
    /// The underlying Mat, BORROWED. Callers must not dispose it, and must not keep it past the
    /// lifetime of this <see cref="Image"/>.
    /// </summary>
    public Mat Mat => _disposed
        ? throw new ObjectDisposedException(nameof(Image))
        : _mat;

    public int Width => Mat.Cols;
    public int Height => Mat.Rows;
    public int Channels => Mat.Channels();
    public bool IsEmpty => _disposed || _mat.Empty();

    /// <summary>An independent copy. The caller owns the result.</summary>
    public Image Clone() => new(Mat.Clone());

    /// <summary>
    /// Detaches the Mat and hands it over, leaving this instance disposed.
    ///
    /// <para>
    /// The analogue of the Go port's <c>Results.TakeCanvas</c>, and it exists for the same reason:
    /// exactly one image has to outlive a pipeline run — the canvas the service stores — while every
    /// intermediate must be released immediately. Without a way to say that, the only options are
    /// disposing what the caller still needs or disposing nothing, and the Go port shipped the
    /// second one.
    /// </para>
    /// </summary>
    public Mat Take()
    {
        var mat = Mat;
        _disposed = true;
        _mat = null!;
        return mat;
    }

    /// <summary>
    /// Copies the pixels into an <see cref="NdArray"/> shaped <c>(H, W, C)</c>, uint8.
    ///
    /// <para>
    /// Used only by the probe: this is how an image stage becomes a comparable payload. A copy, not
    /// a view, because <c>Mat</c> rows can be padded — <c>Step()</c> is not necessarily
    /// <c>Cols * ElemSize()</c>, and reading the buffer as if it were contiguous yields an image
    /// with a diagonal skew that looks like a warp bug.
    /// </para>
    /// </summary>
    public NdArray ToArray()
    {
        Mat mat = Mat;
        if (mat.Type().Depth != MatType.CV_8U)
        {
            throw new InvalidOperationException($"imaging: ToArray expects 8-bit, got {mat.Type()}");
        }

        int h = mat.Rows, w = mat.Cols, c = mat.Channels();
        var data = new byte[(long)h * w * c];
        int rowBytes = w * c;
        for (int y = 0; y < h; y++)
        {
            Marshal.Copy(mat.Ptr(y), data, y * rowBytes, rowBytes);
        }
        return NdArray.FromUInt8(data, h, w, c);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _mat?.Dispose();
        _mat = null!;
    }
}

internal static class Marshal
{
    /// <summary>
    /// Thin alias so the copy above reads the same as the Go port's row loop. Kept separate because
    /// System.Runtime.InteropServices.Marshal.Copy has many overloads and the IntPtr one is easy to
    /// mis-resolve.
    /// </summary>
    internal static void Copy(IntPtr source, byte[] destination, int startIndex, int length) =>
        System.Runtime.InteropServices.Marshal.Copy(source, destination, startIndex, length);
}
