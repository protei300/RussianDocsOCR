using OpenCvSharp;
using RussianDocs.DocumentProcessing.Tensors;

namespace RussianDocs.DocumentProcessing.Imaging;

/// <summary>Interpolation flags, named so call sites say which one the reference used.</summary>
public enum Interpolation
{
    /// <summary><c>cv2.INTER_LINEAR</c> — OpenCV's default and what every resize here uses.</summary>
    Linear = (int)InterpolationFlags.Linear,
    /// <summary><c>cv2.INTER_AREA</c> — only the thumbnail.</summary>
    Area = (int)InterpolationFlags.Area,
}

/// <summary>
/// Decode, resize and the other pixel-level primitives.
///
/// <para>
/// **Everything goes through OpenCV, never through a .NET imaging library.** Measured in the spike:
/// OpenCV and a non-OpenCV JPEG decoder disagree by up to 14 LSB on 58-83 % of pixels, because
/// libjpeg-turbo's IDCT differs from other implementations — enough that "1e-3 on numeric outputs"
/// is unreachable before inference even starts. The same applies to resize: <c>cv2.resize</c> with
/// INTER_LINEAR computes in FIXED POINT with 11-bit coefficients, not in float, and cannot be
/// reproduced by hand.
/// </para>
/// </summary>
public static class Io
{
    /// <summary>
    /// Decodes to RGB.
    ///
    /// <para>
    /// This is <c>BasePreprocessing.__call__</c> and the first half of <c>Pipeline._prepare_image</c>:
    /// <c>imdecode(IMREAD_COLOR)</c> gives BGR, and the pipeline works in RGB. Format is sniffed from
    /// the content, not from the extension — <c>tests/images/OCRv2</c> contains a file named
    /// <c>.png</c> that is not one, and cv2 never cared because it sniffs too.
    /// </para>
    /// </summary>
    public static Image DecodeRgb(byte[] data)
    {
        Mat bgr = Cv2.ImDecode(data, ImreadModes.Color);
        if (bgr is null || bgr.Empty())
        {
            bgr?.Dispose();
            throw new InvalidDataException($"imaging: decoded an empty image ({data.Length} bytes)");
        }
        using (bgr)
        {
            var rgb = new Mat();
            Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);
            return Image.Wrap(rgb);
        }
    }

    /// <summary>
    /// Width and height of an encoded image, or <c>false</c> if it cannot be decoded.
    ///
    /// <para>
    /// Still a FULL decode, on purpose: the caller uses this to reject an undecodable upload with an
    /// immediate, actionable error rather than letting it become a mysterious failed job, and only a
    /// real decode proves decodability. What it skips is the BGR-to-RGB conversion that
    /// <see cref="DecodeRgb"/> owes the pipeline and that a caller wanting two integers does not —
    /// a second full pass over the image, roughly 36 MB of pointless copying on a phone photo.
    /// </para>
    /// </summary>
    public static bool TryDecodeSize(byte[] data, out int width, out int height)
    {
        width = height = 0;
        Mat? mat = null;
        try
        {
            mat = Cv2.ImDecode(data, ImreadModes.Color);
            if (mat is null || mat.Empty())
            {
                return false;
            }
            width = mat.Cols;
            height = mat.Rows;
            return true;
        }
        catch (OpenCVException)
        {
            return false;
        }
        finally
        {
            mat?.Dispose();
        }
    }

    public static Image LoadRgb(string path) => DecodeRgb(File.ReadAllBytes(path));

    /// <summary>
    /// Resizes to an exact size.
    ///
    /// <para>
    /// Argument order follows <c>cv2.resize</c>'s <c>dsize=(w, h)</c>, NOT numpy's <c>(h, w)</c>. The
    /// shipped model input sizes are square, which hides an axis swap — the conformance suite
    /// therefore includes a deliberately non-square resize.
    /// </para>
    /// </summary>
    public static Image Resize(Image src, int width, int height, Interpolation interp)
    {
        var dst = new Mat();
        Cv2.Resize(src.Mat, dst, new Size(width, height), 0, 0, (InterpolationFlags)interp);
        return Image.Wrap(dst);
    }

    /// <summary>
    /// Shrinks so the longest side is at most <paramref name="imgSize"/>.
    ///
    /// <para>
    /// The second half of <c>Pipeline._prepare_image</c>: <c>ratio = max(max(h, w) / img_size, 1)</c>
    /// — so it only ever SHRINKS — then <c>int(w // ratio), int(h // ratio)</c>.
    /// </para>
    ///
    /// <para>
    /// The floor divisions go through <see cref="PyNum.FloorDivInt"/>, and that is load-bearing: for
    /// 2999x1777 the correct answer is 1499, while <c>(int)Math.Floor(2999 / ratio)</c> gives 1500.
    /// A canvas one pixel wider shifts every box downstream, and the failure surfaces at a stage far
    /// from its cause. A consequence worth stating because it surprises: this does NOT guarantee the
    /// long side equals <paramref name="imgSize"/>.
    /// </para>
    /// </summary>
    public static Image FitToLongestSide(Image src, int imgSize)
    {
        int h = src.Height, w = src.Width;
        double ratio = Math.Max((double)Math.Max(h, w) / imgSize, 1.0);
        if (ratio == 1.0)
        {
            // No resize at all when it already fits — not a resize by 1.0, which would still run
            // the interpolator and could perturb pixels.
            return src.Clone();
        }
        int newW = PyNum.FloorDivInt(w, ratio);
        int newH = PyNum.FloorDivInt(h, ratio);
        return Resize(src, newW, newH, Interpolation.Linear);
    }

    public static Image ToBgr(Image src)
    {
        var dst = new Mat();
        Cv2.CvtColor(src.Mat, dst, ColorConversionCodes.RGB2BGR);
        return Image.Wrap(dst);
    }

    public static Image ToGray(Image src)
    {
        var dst = new Mat();
        Cv2.CvtColor(src.Mat, dst, ColorConversionCodes.RGB2GRAY);
        return Image.Wrap(dst);
    }

    public static Image CopyMakeBorderConstant(
        Image src, int top, int bottom, int left, int right, byte r, byte g, byte b)
    {
        var dst = new Mat();
        Cv2.CopyMakeBorder(src.Mat, dst, top, bottom, left, right, BorderTypes.Constant,
            new Scalar(r, g, b));
        return Image.Wrap(dst);
    }

    public static Image NewFilled(int height, int width, byte r, byte g, byte b) =>
        Image.Wrap(new Mat(height, width, MatType.CV_8UC3, new Scalar(r, g, b)));

    /// <summary>Writes a PNG. The input is RGB; the encoder expects BGR, so it converts first.</summary>
    public static void WritePngFromRgb(string path, Image rgb)
    {
        using Image bgr = ToBgr(rgb);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        if (!Cv2.ImWrite(path, bgr.Mat))
        {
            throw new IOException($"imaging: could not write {path}");
        }
    }

    /// <summary>
    /// Writes a JPEG at the given quality. The input is RGB; the encoder expects BGR.
    ///
    /// <para>
    /// Used only for the list-page thumbnail, where a lossy 96-px-wide image is the point. Nothing the
    /// conformance suite compares goes through here.
    /// </para>
    /// </summary>
    public static void WriteJpegFromRgb(string path, Image rgb, int quality)
    {
        using Image bgr = ToBgr(rgb);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        if (!Cv2.ImWrite(path, bgr.Mat, new ImageEncodingParam(ImwriteFlags.JpegQuality, quality)))
        {
            throw new IOException($"imaging: could not write {path}");
        }
    }
}
